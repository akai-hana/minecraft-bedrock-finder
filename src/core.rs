use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use smallvec::SmallVec;
use rayon::prelude::*;

use crate::gpu::{GpuContext, GpuBlock};

// Constants

const FALLBACK_LO: u64 = (-7_046_029_254_386_353_131_i64) as u64;
const FALLBACK_HI: u64 = 7_640_891_576_956_012_809_u64;

// Number of spiral positions examined between cancellation-flag checks.
//
// The search loop runs par_iter().find_first() over SEARCH_BATCH_SIZE / 8 groups
// of 8 positions each.  Rayon distributes the groups across all available cores;
// find_first cancels remaining workers as soon as the earliest match is confirmed.
//
// At ~100 M positions/s per core with 8 cores that is ~800 M positions/s, so
// 2^20 (~ 1 M) positions per batch ~ 1 ms per batch - short enough that the
// cancel button feels instant.  Must be a multiple of 8 (SIMD group size).
const SEARCH_BATCH_SIZE: i64 = 1 << 20; // must be a multiple of 8
const _: () = assert!(SEARCH_BATCH_SIZE % 8 == 0, "SEARCH_BATCH_SIZE must be a multiple of 8");

// Number of consecutive spiral groups each Rayon task processes in one shot.
//
// Within a chunk the running (l, j, x, z) spiral state is carried from group
// to group by `fill_group_from_state`, which requires **no** f64::sqrt.  Only
// the very first group of each chunk calls `spiral_coords_with_state` (one
// sqrt); every subsequent group in the chunk is derived with comparisons and
// integer adds alone.
//
// For GROUPS_PER_CHUNK = 1024 this reduces sqrt calls from 1 per 8 positions
// to 1 per 8192 positions (a 1024x reduction). On a typical 3 GHz machine
// vsqrtsd costs ~20 cycles, saving roughly 20 cycles/group relative to the
// ~30-50 cycle hot-path cost after an early-exit on block 0.
//
// CHUNKS_PER_BATCH = GROUPS_PER_BATCH / GROUPS_PER_CHUNK = 128, which is
// ample parallelism for machines with <= 32 cores.
//
// Must evenly divide GROUPS_PER_BATCH (= SEARCH_BATCH_SIZE / 8).
pub const GROUPS_PER_CHUNK: i64 = 1024;
const GROUPS_PER_BATCH: i64 = SEARCH_BATCH_SIZE / 8;
pub const CHUNKS_PER_BATCH: i64 = GROUPS_PER_BATCH / GROUPS_PER_CHUNK;
const _: () = assert!(
    GROUPS_PER_BATCH % GROUPS_PER_CHUNK == 0,
    "GROUPS_PER_BATCH must be divisible by GROUPS_PER_CHUNK",
);

/// Compute how many chunks to dispatch in a single `into_par_iter().find_first()` call.
///
/// Two goals are addressed together:
///
/// **(a) Rayon overhead** - each `find_first` call pays for task-graph setup,
/// work-stealing initialisation, and join synchronisation. Dispatching 16x more
/// chunks per call amortises that cost while keeping each chunk small
/// (~8192 positions, a few microseconds), so cancel responsiveness is unaffected.
///
/// **(b) Load balance on high-core-count machines** - with `CHUNKS_PER_BATCH = 128`
/// machines with > 32 cores get fewer than 4 tasks/core, leaving workers idle at
/// the tail of each `find_first`. The formula `max(16 * CHUNKS_PER_BATCH,
/// nextpow2(4 * num_threads))` restores >= 4 tasks/core on any machine without
/// affecting the common <= 32-core case.
///
/// Called once at the start of `run_search` so the thread-pool query is not
/// repeated inside the search loop.
fn compute_super_batch_chunks() -> i64 {
    let threads = rayon::current_num_threads() as i64;
    // 4 chunks/thread: enough for work-stealing to keep every core busy.
    let for_balance  = threads.saturating_mul(4);
    // 16x the original static batch size to amortise Rayon setup cost.
    let for_batching = CHUNKS_PER_BATCH * 16;
    let target = for_balance.max(for_batching);
    // Round up to the nearest power of two (executes once at startup).
    (target as u64).next_power_of_two() as i64
}

// Bedrock type

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BedrockType { Floor, Roof }

impl BedrockType {
    /// The Y coordinate at which this bedrock layer is always solid.
    /// Floor: -64 (bottom). Roof: 128 (top). For Roof, min() > max(); see
    /// `compute_probability` for how this ordering is handled.
    pub fn min(self) -> i32 { match self { BedrockType::Floor => -64, BedrockType::Roof => 128 } }
    /// The Y coordinate at which this bedrock layer is always air.
    /// Floor: -59. Roof: 123. The names reflect always-solid vs always-air, not ordering.
    pub fn max(self) -> i32 { match self { BedrockType::Floor => -59, BedrockType::Roof => 123 } }
}

// Xoroshiro128++ RNG (mirrors Xoroshiro128PlusPlusRandomImpl)

#[inline(always)]
fn xoroshiro_next(s: &mut (u64, u64)) -> u64 {
    let (s0, s1) = *s;
    let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);
    let s1 = s1 ^ s0;
    s.0 = s0.rotate_left(49) ^ s1 ^ (s1 << 21);
    s.1 = s1.rotate_left(28);
    result
}

#[inline(always)]
fn guard_zero(state: (u64, u64)) -> (u64, u64) {
    if (state.0 | state.1) == 0 { (FALLBACK_LO, FALLBACK_HI) } else { state }
}

// SplitMix64 (mirrors RandomSeed.nextSplitMix64Int)

#[inline]
fn split_mix64(mut seed: u64) -> u64 {
    seed = (seed ^ (seed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    seed = (seed ^ (seed >> 27)).wrapping_mul(0x94D049BB133111EB);
    seed ^ (seed >> 31)
}

fn create_xoroshiro_seed(seed: i64) -> (u64, u64) {
    let l = (seed as u64) ^ 0x6A09E667F3BCC909_u64;
    let m = l.wrapping_add(FALLBACK_LO);
    guard_zero((split_mix64(l), split_mix64(m)))
}


// Probability (mirrors BedrockReader.computeProbability)

pub fn compute_probability(y: i32, bt: BedrockType) -> f64 {
    let (min, max) = (bt.min(), bt.max());
    match bt {
        BedrockType::Floor => {
            if y == min     { 2.0 }
            else if y > max { -1.0 }
            else { 1.0 - (y - min) as f64 / (max - min) as f64 }
        }
        BedrockType::Roof => {
            if y == min     { 2.0 }
            else if y < max { -1.0 }
            else { 1.0 - (y - max) as f64 / (min - max) as f64 }
            // = 1 - (y - 123) / 5
        }
    }
}

// Deriver-seed derivation (mirrors BedrockReader constructor)

fn compute_deriver_seeds(seed: i64, bt: BedrockType) -> (i64, i64) {
    // 1. Base random from world seed (goes through SplitMix64)
    let mut state = create_xoroshiro_seed(seed);

    // 2. createRandomDeriver() on AbstractRandom -> consume 2 outputs
    let d1_lo = xoroshiro_next(&mut state) as i64;
    let d1_hi = xoroshiro_next(&mut state) as i64;

    // 3. createRandom(identifier_string) on RandomDeriver
    //    Uses Guava MD5 (= standard MD5, UTF-8 bytes), big-endian Longs.fromBytes.
    //    Both identifier strings are compile-time constants -> digest is compile-time
    //    constant -> no runtime MD5 computation needed.
    const FLOOR_MD5: [u8; 16] = [0xBB, 0xF7, 0x92, 0x8B, 0x7B, 0xF1, 0xD2, 0x85,
    0xC4, 0xDC, 0x7C, 0xF9, 0x0E, 0x1B, 0x3B, 0x94]; // md5("minecraft:bedrock_floor")
    const ROOF_MD5:  [u8; 16] = [0x8E, 0xBD, 0x4A, 0x1D, 0x13, 0x1D, 0x71, 0xCC,
    0xC9, 0x84, 0xCF, 0xBB, 0x68, 0x4A, 0x26, 0xC4]; // md5("minecraft:bedrock_roof")
    let bs = match bt {
        BedrockType::Floor => FLOOR_MD5,
        BedrockType::Roof  => ROOF_MD5,
    };
    let l = i64::from_be_bytes([bs[0], bs[1], bs[2], bs[3], bs[4], bs[5], bs[6], bs[7]]);
    let m = i64::from_be_bytes([bs[8], bs[9], bs[10], bs[11], bs[12], bs[13], bs[14], bs[15]]);
    // new Xoroshiro128PlusPlusRandom(l ^ seedLo, m ^ seedHi) with a direct init and no SplitMix64
    let mut state2 = guard_zero(((l ^ d1_lo) as u64, (m ^ d1_hi) as u64));

    // 4. createRandomDeriver() on that random -> consume 2 more outputs
    let d2_lo = xoroshiro_next(&mut state2) as i64;
    let d2_hi = xoroshiro_next(&mut state2) as i64;

    (d2_lo, d2_hi)
}

// Block data

/// Precompute the integer threshold for the SIMD hot path.
///
/// The scalar check does `(result >> 40) as f32 * 2^-24 < probability`,
/// which is exactly equivalent to `(result >> 40) < probability * 2^24`
/// (i.e. `top24 < prob_threshold`) when `probability * 2^24` is not an
/// exact integer, which holds for every bedrock probability used here
/// (e.g. 0.8 * 16777216 = 13421772.8, 0.6 * 16777216 = 10066329.6).
#[inline(always)]
pub fn prob_to_threshold(probability: f64) -> u64 {
    // The SIMD integer-threshold comparison is only valid for probability in (0, 1).
    // Values outside this range (always-bedrock >= 1.0, never-bedrock <= 0.0) are
    // filtered out by run_search before reaching any SIMD kernel, but assert here
    // so a future refactor cannot silently introduce incorrect thresholds.
    debug_assert!(
        probability > 0.0 && probability < 1.0,
        "prob_to_threshold requires probability in (0, 1), got {}",
        probability
    );
    (probability * 16_777_216.0_f64) as u64
}

#[derive(Clone)]
pub struct Block {
    pub x: i32, pub y: i32, pub z: i32,
    pub should_be_bedrock: bool,
    pub probability:    f64,
    /// Precomputed integer threshold: `(probability * 2^24) as u64`.
    /// Used by the SIMD kernels to avoid float conversion in the hot path.
    pub prob_threshold: u64,
}

// 64-byte aligned allocation wrapper
//
// Vec<T> uses align_of::<T>() as its allocation alignment (4 bytes for i32,
// 8 bytes for i64, 1 byte for bool).  AVX-512 and AVX2 SIMD loads can suffer
// extra cache-line fetches when the start of an SoA array crosses a 64-byte
// boundary.  AlignedVec<T> allocates with 64-byte alignment so every field in
// `Blocks` begins on a cache-line boundary, allowing the slightly faster
// aligned load variants and guaranteeing no cache-line split on the first
// per-block iteration load.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};

struct AlignedVec<T: Copy> {
    ptr:    std::ptr::NonNull<T>,
    len:    usize,
    layout: Layout,
}

// SAFETY: AlignedVec<T> owns its allocation and provides no interior mutability.
unsafe impl<T: Copy + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedVec<T> {}

impl<T: Copy> AlignedVec<T> {
    /// Allocate a 64-byte-aligned buffer with room for `cap` elements.
    ///
    /// Combined with `push`, this lets callers write directly into the final
    /// aligned buffer in a single pass - eliminating the intermediate `Vec`
    /// alloc->fill->dealloc round-trips that a copy-from-slice approach would
    /// otherwise require.
    fn with_capacity(cap: usize) -> Self {
        if cap == 0 {
            return Self {
                ptr:    std::ptr::NonNull::dangling(),
                len:    0,
                layout: Layout::new::<u8>(),
            };
        }
        let size  = cap * std::mem::size_of::<T>();
        let align = 64_usize.max(std::mem::align_of::<T>());
        let layout = Layout::from_size_align(size, align)
            .expect("AlignedVec: invalid layout");
        // SAFETY: layout is non-zero-size (cap > 0 and size_of::<T>() >= 1).
        let raw = unsafe { alloc(layout) } as *mut T;
        if raw.is_null() { handle_alloc_error(layout); }
        Self {
            ptr:    std::ptr::NonNull::new(raw).unwrap(),
            len:    0,
            layout,
        }
    }

    /// Append one element.
    ///
    /// # Panics
    /// Panics (in debug) if the buffer is already full (len == capacity).
    /// Capacity is fixed at construction; `AlignedVec` does not grow.
    #[inline(always)]
    fn push(&mut self, val: T) {
        let cap = self.layout.size() / std::mem::size_of::<T>();
        debug_assert!(self.len < cap, "AlignedVec::push: buffer full");
        // SAFETY: `self.len < cap`, so `self.ptr + self.len` is within the
        // allocation and valid for one write.
        unsafe { self.ptr.as_ptr().add(self.len).write(val); }
        self.len += 1;
    }
}

impl<T: Copy> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        if self.len == 0 { return &[]; }
        // SAFETY: ptr is valid for `len` reads, properly aligned, and live.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.len == 0 { return; }
        // T: Copy implies no destructor - skip drop_in_place entirely.
        // SAFETY: we own the allocation; elements are trivially-copyable (no Drop).
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, self.layout); }
    }
}

// Structure-of-Arrays layout for the SIMD hot path
//
// Structure-of-Arrays layout: each field lives in its own contiguous,
// 64-byte-aligned array. The hot SIMD loop touches `bx_hash_term`,
// `bz_hash_term`, `y`, `prob_threshold`, and `sbb_xor` - five independent
// streams that stay cache-line-local. Raw `x`/`z` coordinates are folded into
// the pre-computed hash terms at construction time, and `probability: f64` is
// converted to the integer `prob_threshold` and not retained.
//
// Conversion from `Vec<Block>` (used for building, sorting, and rotating)
// happens once in `run_search`, just before the search loop begins.
struct Blocks {
    y:            AlignedVec<i32>,
    /// Per-block XOR mask used to negate the bedrock-mask in non-bedrock
    /// blocks without a branch.
    ///
    /// Encoding: `0x00` when `should_be_bedrock` is true (pass = bedrock),
    /// `0xFF` when false (pass = non-bedrock, i.e. `bedrock_mask ^ 0xFF`).
    ///
    /// In the SIMD kernels: `active &= bedrock_mask ^ sbb_xor[i]`.
    /// The existing `active` initialisation (0xFF / 0x0F) naturally masks off
    /// the upper bits produced by the XOR in the 4-lane AVX2 path, so no
    /// extra `& 0x0F` is needed.
    ///
    /// In the scalar path: `let sbb = sbb_xor[i] == 0; if sbb != is_bed { ... }`.
    sbb_xor:      AlignedVec<u8>,
    /// Integer threshold `(probability * 2^24) as u64`.  Used by both the
    /// SIMD hot loop and the scalar confirmation path; the raw `f64`
    /// probability is not stored here.
    prob_threshold:    AlignedVec<u64>,
    /// Precomputed `(z as i64).wrapping_mul(116_129_781)`.
    ///
    /// The z-coordinate path keeps full 64-bit precision (no i32-first cast),
    /// matching the Java `(long)z * 116129781L` idiom.  Used by the AVX-512
    /// kernel to hoist the oz * K_z multiplication out of the block loop.
    bz_hash_term:      AlignedVec<i64>,
    /// Precomputed `bx.wrapping_mul(3_129_871)` in i32 wrapping arithmetic.
    ///
    /// The x-coordinate path uses the Java `(long)(x * 3129871)` idiom:
    /// multiply in i32 (wrapping), then sign-extend to i64.  Because wrapping
    /// integer multiplication distributes over addition in Z/2^32Z
    /// / `(ox + bx) * K == ox * K + bx * K (mod 2^32)` - the ox * K term can be
    /// hoisted out of the block loop in i32 space, and bx * K is added inside.
    /// The sum is sign-extended to i64 afterwards, exactly matching the scalar.
    ///
    /// Used by the AVX-512 and AVX2 kernels to replace a per-block mullo with
    /// a cheaper add in the hot loop.
    bx_hash_term:      AlignedVec<i32>,
    /// `Some(y)` when every block in this set shares the same Y layer
    /// (e.g. all constraints come from y = -60).  Used by the hot-path kernels
    /// to hoist the per-block y broadcast **outside** the block loop.
    ///
    /// The compiler cannot prove loop-invariance from the SoA slice alone
    /// / every `y_s[i]` access looks like a distinct load - so the hoist must be
    /// explicit.  When `None`, kernels fall back to loading `y_s[i]` per
    /// iteration as before.
    uniform_y: Option<i32>,
    /// `Some(t)` when every block shares the same `prob_threshold`.
    ///
    /// This is always true when `uniform_y` is `Some`, because the threshold
    /// depends only on the Y layer and bedrock type, not on block position.
    /// For example, all y = -60 Floor blocks have threshold
    /// `(0.2 x 2^24) as u64 = 3_355_443` regardless of `should_be_bedrock`.
    /// Allows the SIMD threshold-broadcast vector to be computed once before
    /// the block loop instead of once per block.
    uniform_threshold: Option<u64>,
}

impl Blocks {
    fn from_vec(v: Vec<Block>) -> Self {
        // Collect all five fields in a single pass over `v` to avoid reading the
        // same data multiple times. Pre-allocate each AlignedVec with the exact
        // capacity needed and push directly, so no intermediate allocations or
        // copies are required.
        let n = v.len();
        let mut ys   = AlignedVec::<i32>::with_capacity(n);
        let mut sbbs = AlignedVec::<u8>::with_capacity(n);
        let mut thrs = AlignedVec::<u64>::with_capacity(n);
        let mut bzs  = AlignedVec::<i64>::with_capacity(n);
        let mut bxs  = AlignedVec::<i32>::with_capacity(n);
        let mut uniform_y:   Option<i32> = v.first().map(|b| b.y);
        let mut uniform_thr: Option<u64> = v.first().map(|b| b.prob_threshold);
        for b in &v {
            ys.push(b.y);
            // sbb_xor encoding: 0x00 = should be bedrock, 0xFF = should be non-bedrock.
            // SIMD kernels XOR the bedrock mask with this value to negate it
            // branchlessly; the scalar path recovers `sbb = sbb_xor == 0`.
            sbbs.push(if b.should_be_bedrock { 0x00u8 } else { 0xFFu8 });
            thrs.push(b.prob_threshold);
            // bz_hash_term: precomputed (bz as i64) * 116_129_781.
            // Used by the AVX-512 and AVX2 kernels to hoist oz*K_z out of the block loop.
            bzs.push((b.z as i64).wrapping_mul(116_129_781_i64));
            // bx_hash_term: precomputed bx.wrapping_mul(3_129_871) in i32 space.
            // Wrapping i32 multiplication distributes over addition in Z/2^32Z, so
            // (ox + bx) * K == ox*K + bx*K (mod 2^32).  The ox*K multiply is hoisted
            // once before the block loop; inside the loop only a cheap i32 add is needed.
            bxs.push(b.x.wrapping_mul(3_129_871_i32));
            // Track uniformity incrementally - no second pass needed.
            if uniform_y   == Some(b.y)             {} else { uniform_y   = None; }
            if uniform_thr == Some(b.prob_threshold) {} else { uniform_thr = None; }
        }
        // Verify that all SoA fields have the same length. These asserts fire once
        // at construction time, not inside the hot-path kernels, so in-kernel
        // checks can be debug_assert_eq! (compiled away in release builds).
        assert_eq!(n, ys.len(),   "Blocks field length mismatch: y");
        assert_eq!(n, sbbs.len(), "Blocks field length mismatch: sbb_xor");
        assert_eq!(n, thrs.len(), "Blocks field length mismatch: prob_threshold");
        assert_eq!(n, bzs.len(),  "Blocks field length mismatch: bz_hash_term");
        assert_eq!(n, bxs.len(),  "Blocks field length mismatch: bx_hash_term");

        Self {
            y:              ys,
            sbb_xor:        sbbs,
            prob_threshold: thrs,
            bz_hash_term:   bzs,
            bx_hash_term:   bxs,
            uniform_y,
            uniform_threshold: uniform_thr,
        }
    }
}

fn clamp01(v: f64) -> f64 { v.clamp(0.0, 1.0) }

/// Scalar formation check with pre-computed position terms (optimisation 2).
///
/// Accepts `ox_i32_term` and `oz_i64_term` already computed by the caller,
/// eliminating the two multiplications that would otherwise be re-executed for
/// every rotation when `search_all_rotations` is active on a non-SIMD machine.
#[inline(always)]
fn check_formation_with_terms(
    ox_i32_term: i32,
    oz_i64_term: i64,
    dlo: i64,
    dhi: i64,
    blocks: &Blocks,
) -> bool {
    debug_assert_ne!(dhi, 0, "deriver hi seed must be non-zero");
    let bx_s  = &blocks.bx_hash_term[..];
    let bz_s  = &blocks.bz_hash_term[..];
    let y_s   = &blocks.y[..];
    let thr_s = &blocks.prob_threshold[..];
    let sbb_s = &blocks.sbb_xor[..];
    let n     = bx_s.len();
    debug_assert_eq!(n, bz_s.len(),  "Blocks field length mismatch: bz_hash_term");
    debug_assert_eq!(n, y_s.len(),   "Blocks field length mismatch: y");
    debug_assert_eq!(n, thr_s.len(), "Blocks field length mismatch: prob_threshold");
    debug_assert_eq!(n, sbb_s.len(), "Blocks field length mismatch: sbb_xor");
    // Split on whether y and threshold are loop-invariant so the compiler can
    // emit tight loops without Option overhead. `$i` is a macro parameter
    // (not a loop variable inside the macro) so it shares hygiene context
    // with the `for i in 0..n` loop in each match arm.
    macro_rules! inner_body {
        ($i:expr, $y_i64:expr, $thr_i:expr) => {{
            let term_x = ox_i32_term.wrapping_add(bx_s[$i]) as i64;
            let term_z = oz_i64_term.wrapping_add(bz_s[$i]);
            let mut l  = term_x ^ term_z ^ ($y_i64);
            let inner  = l.wrapping_mul(42_317_861_i64).wrapping_add(11_i64);
            l = l.wrapping_mul(inner);
            let hash   = l >> 16;
            let s0     = (hash ^ dlo) as u64;
            let s1     = dhi as u64;
            let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);
            let is_bed = (result >> 40) < ($thr_i);
            let sbb    = sbb_s[$i] == 0;
            if sbb != is_bed { return false; }
        }};
    }
    match (blocks.uniform_y, blocks.uniform_threshold) {
        (Some(y), Some(thr)) => {
            let y_i64 = y as i64;
            for i in 0..n { inner_body!(i, y_i64, thr); }
        }
        (Some(y), None) => {
            let y_i64 = y as i64;
            for i in 0..n { inner_body!(i, y_i64, thr_s[i]); }
        }
        (None, Some(thr)) => {
            for i in 0..n { inner_body!(i, y_s[i] as i64, thr); }
        }
        (None, None) => {
            for i in 0..n { inner_body!(i, y_s[i] as i64, thr_s[i]); }
        }
    }
    true
}

// SIMD kernel (AVX2, x86-64 only)
//
// Processes 4 (ox, oz) pairs simultaneously using 256-bit AVX2 registers
// (4 * 64-bit lanes). The math_hash kernel (a few multiplies and XORs),
// xoroshiro128++ step (rotate, add, XOR), and threshold comparison all map
// cleanly to SIMD with no data dependencies between lanes. The scalar path
// is kept as a fallback for non-AVX2 hardware and for hit confirmation.
//
// Key design choices
//
// - mullo_epi64  AVX2 has no 64-bit lane multiply; we emulate it as:
//                a*b = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32
//                (the 2^64 term vanishes in the low 64 bits).
//
// - term_z       z fits in i32 and 116_129_781 fits in i32, so
//                _mm256_mul_epi32, which uses only the low 32 bits of each
//                64-bit lane, gives the exact same result as the scalar
//                (z as i64).wrapping_mul(116_129_781_i64).
//
// - hash >> 16   Arithmetic vs logical right-shift is irrelevant here: the
//                result is immediately XORd with dlo and reinterpreted as u64,
//                so only the bit pattern matters.  _mm256_srli_epi64 suffices.
//
// - guard_zero   Handled with a per-lane _mm256_cmpeq_epi64 + _mm256_blendv_epi8.
//                In practice dhi != 0 so the branch never fires, but it is
//                correct when it does.
//
// - float cmp    Done in f64 (via _mm256_cvtps_pd) to match the scalar
//                `(f as f64) < probability` exactly.
//
// - early exit   check_formation_x4 ANDs per-block 4-bit masks and breaks as
//                soon as active == 0, mirroring .all() short-circuiting.
//
// - spiral order find_first operates over SEARCH_BATCH_SIZE/8 groups.  Once a
//                group is found, a scalar scan of up to 8 positions pinpoints
//                the exact first match in spiral order.

#[cfg(target_arch = "x86_64")]
mod simd_avx2 {
    // NOTE: `use ::core::arch::...` uses an absolute path to the standard library's
    // `core` crate, bypassing the parent module also named `core`.
    use ::core::arch::x86_64::*;

    // Helpers

    /// Low 64 bits of a 4-lane 64-bit integer multiply.
    ///
    /// AVX2 has no `mullo_epi64`, so we use:
    ///   a*b (low 64) = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mullo_epi64(a: __m256i, b: __m256i) -> __m256i {
        let a_hi  = _mm256_srli_epi64(a, 32);
        let b_hi  = _mm256_srli_epi64(b, 32);
        let lo_lo = _mm256_mul_epu32(a, b);
        let cross = _mm256_slli_epi64(
            _mm256_add_epi64(
                _mm256_mul_epu32(a_hi, b),
                _mm256_mul_epu32(a,    b_hi),
            ),
            32,
        );
        _mm256_add_epi64(lo_lo, cross)
    }

    /// Low 64 bits of a 4-lane 64*32-bit multiply.
    ///
    /// Faster than `mullo_epi64` when the high 32 bits of every lane in `b`
    /// are zero (e.g. a scalar constant < 2^32). The `a_lo*b_hi` and
    /// `a_hi*b_hi` cross-products vanish, saving one `mul_epu32` and one
    /// `add_epi64`.
    ///
    /// # Safety
    /// Requires AVX2. Every lane in `b` must have its high 32 bits zeroed.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mullo_epi64_small_b(a: __m256i, b: __m256i) -> __m256i {
        let a_hi  = _mm256_srli_epi64(a, 32);
        let lo_lo = _mm256_mul_epu32(a, b);                // a_lo * b_lo
        let cross = _mm256_slli_epi64(_mm256_mul_epu32(a_hi, b), 32); // a_hi * b_lo << 32
        _mm256_add_epi64(lo_lo, cross)
    }

    /// Rotate each 64-bit lane left by 17 bits.
    ///
    /// # Safety
    /// Caller must guarantee AVX2 is available. All intrinsics here are safe
    /// under `target_feature_11` within this `#[target_feature(enable = "avx2")]` fn.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rotl17_epi64(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_slli_epi64(x, 17), _mm256_srli_epi64(x, 47))
    }

    // Core SIMD kernel

    /// Compute `math_hash` for 4 lanes with a pre-computed z-term.
    ///
    /// `term_z` is a `__m256i` of 4 * i64 values already carrying
    /// `(oz as i64) * K_z + bz_hash_term`, allowing the caller to hoist the
    /// expensive `cvtepi32_epi64` + multiply out of the block loop and replace
    /// it with a single cheap `add_epi64`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn math_hash_x4_precomputed(x_term_i32: __m128i, y: i32, term_z: __m256i) -> __m256i {
        unsafe {
            let term_x = _mm256_cvtepi32_epi64(x_term_i32);
            let y64    = _mm256_set1_epi64x(y as i64);
            let mut l  = _mm256_xor_si256(_mm256_xor_si256(term_x, term_z), y64);

            let inner = _mm256_add_epi64(
                // 42_317_861 < 2^32, so the cheap small-b form saves one mul_epu32
                // and one add_epi64 vs the general mullo_epi64.
                mullo_epi64_small_b(l, _mm256_set1_epi64x(42_317_861_i64)),
                _mm256_set1_epi64x(11_i64),
            );
            l = mullo_epi64(l, inner);

            let logical   = _mm256_srli_epi64(l, 16);
            let sign_mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), l);
            let top16     = _mm256_and_si256(
                sign_mask,
                _mm256_set1_epi64x(0xFFFF_0000_0000_0000_u64 as i64),
            );
            _mm256_or_si256(logical, top16)
        }
    }

    /// Returns a 4-bit mask: bit `i` is set if position `i` passes the
    /// `is_bedrock` test for the given block parameters.
    ///
    /// Accepts a pre-computed `term_z` (`__m256i` of 4 * i64) carrying
    /// `(oz as i64) * K_z + bz_hash_term`, avoiding the per-block
    /// `cvtepi32_epi64` + multiply inside the hash kernel.
    ///
    /// `sbb_xor` is the precomputed XOR mask from `Blocks::sbb_xor`:
    ///   0x00 -> position is bedrock = pass
    ///   0xFF -> position is non-bedrock = pass (inverts the mask)
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn is_bedrock_x4_precomputed(
        dlo:        i64,
        dhi:        i64,
        x_term_i32: __m128i,  // 4 * i32: (ox+bx)*K_x in i32 wrapping space
        y:          i32,
        term_z:     __m256i,  // 4 * i64: oz*K_z + bz_hash_term (pre-computed)
        prob_threshold: u64,
        sbb_xor:    u8,       // 0x00 = bedrock expected, 0xFF = non-bedrock expected
    ) -> u8 {
        unsafe {
            let hash = math_hash_x4_precomputed(x_term_i32, y, term_z);
            let s0   = _mm256_xor_si256(hash, _mm256_set1_epi64x(dlo));
            let s1   = _mm256_set1_epi64x(dhi);

            debug_assert_ne!(dhi, 0, "deriver hi seed must be non-zero");

            let sum    = _mm256_add_epi64(s0, s1);
            let result = _mm256_add_epi64(rotl17_epi64(sum), s0);

            let top24    = _mm256_srli_epi64(result, 40);
            let thresh_v = _mm256_set1_epi64x(prob_threshold as i64);
            let cmp      = _mm256_cmpgt_epi64(thresh_v, top24);
            let bedrock_mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u8;

            // Branchless inversion via precomputed XOR mask (optimisation 12).
            // active_lo/hi are initialised to 0x0F, so the AND with `active`
            // in the caller naturally clears the upper bits produced by
            // XOR-ing with 0xFF - no extra `& 0x0F` needed.
            bedrock_mask ^ sbb_xor
        }
    }

    /// Fill `xs` and `zs` with an 8-element arithmetic progression for a uniform-leg
    /// group using AVX2 intrinsics.
    ///
    /// `(x_g, z_g)` is the first position of the group; `(dx, dz)` is the constant
    /// step direction (one of `(+/-1, 0)` or `(0, +/-1)`).  The four cases are matched
    /// explicitly so the compiler constant-folds each branch into a minimal sequence.
    ///
    /// Calling this instead of `fill_group_from_state` is valid only when the entire
    /// chunk lies within a single spiral leg (the `uniform` fast-path guard in
    /// `run_search` ensures this).
    ///
    /// # Safety
    /// Requires AVX2.  Caller must have verified `is_x86_feature_detected!("avx2")`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn fill_group_uniform(
        x_g: i32, z_g: i32,
        dx: i32, dz: i32,
        xs: &mut [i32; 8], zs: &mut [i32; 8],
    ) {
        // SAFETY: caller guarantees AVX2 is available (enforced by #[target_feature]).
        unsafe {
            // Constant offsets [0, 1, 2, 3, 4, 5, 6, 7] - one vector for all four cases.
            // The compiler hoists this out of the enclosing group loop because it is
            // a compile-time constant built from immediate operands.
            let steps = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let x_base = _mm256_set1_epi32(x_g);
            let z_base = _mm256_set1_epi32(z_g);
            let (xs_v, zs_v) = match (dx, dz) {
                ( 1,  0) => (_mm256_add_epi32(x_base, steps), z_base),
                (-1,  0) => (_mm256_sub_epi32(x_base, steps), z_base),
                ( 0,  1) => (x_base, _mm256_add_epi32(z_base, steps)),
                ( 0, -1) => (x_base, _mm256_sub_epi32(z_base, steps)),
                _        => unreachable!("dx/dz must be 0 or +/-1"),
            };
            _mm256_storeu_si256(xs.as_mut_ptr() as *mut __m256i, xs_v);
            _mm256_storeu_si256(zs.as_mut_ptr() as *mut __m256i, zs_v);
        }
    }

    /// Precompute the position-dependent hash terms for 8 spiral positions (AVX2 path).
    ///
    /// Returns `(ox_lo_term, ox_hi_term, oz_lo_z_term, oz_hi_z_term)` - the hoisted
    /// x and z multiplications for each 4-wide half that depend only on `(xs, zs)`.
    /// When multiple rotations are checked for the same group, call this once and
    /// forward the results to `check_formation_x8_avx2_with_terms` to avoid recomputing
    /// the same four SIMD multiplications for every rotation.
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn position_terms_x8_avx2(
        positions_x: &[i32; 8],
        positions_z: &[i32; 8],
    ) -> (__m128i, __m128i, __m256i, __m256i) {
        // SAFETY: caller guarantees AVX2 is available.
        unsafe {
            let ox_lo = _mm_loadu_si128(positions_x[..4].as_ptr() as *const __m128i);
            let oz_lo = _mm_loadu_si128(positions_z[..4].as_ptr() as *const __m128i);
            let ox_hi = _mm_loadu_si128(positions_x[4..].as_ptr() as *const __m128i);
            let oz_hi = _mm_loadu_si128(positions_z[4..].as_ptr() as *const __m128i);

            let k_x = _mm_set1_epi32(3_129_871_i32);
            let ox_lo_term = _mm_mullo_epi32(ox_lo, k_x);
            let ox_hi_term = _mm_mullo_epi32(ox_hi, k_x);

            // k_z = 116_129_781 < 2^32 -> high 32 bits of every lane are zero.
            // Use the cheaper small-b form (saves one mul_epu32 + one add_epi64
            // per call, 6 instructions total across both calls) - the same saving
            // already applied to k = 42_317_861 in math_hash_x4_precomputed.
            let k_z = _mm256_set1_epi64x(116_129_781_i64);
            let oz_lo_i64    = _mm256_cvtepi32_epi64(oz_lo);
            let oz_hi_i64    = _mm256_cvtepi32_epi64(oz_hi);
            let oz_lo_z_term = mullo_epi64_small_b(oz_lo_i64, k_z);
            let oz_hi_z_term = mullo_epi64_small_b(oz_hi_i64, k_z);

            (ox_lo_term, ox_hi_term, oz_lo_z_term, oz_hi_z_term)
        }
    }

    /// Inner AVX2 formation check with pre-computed position terms (optimisation 2).
    ///
    /// Accepts the four hoisted terms from `position_terms_x8_avx2`, eliminating the
    /// four SIMD multiplications that would otherwise be re-executed for every rotation
    /// when `search_all_rotations` is active.
    ///
    /// # Safety
    /// Requires AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn check_formation_x8_avx2_with_terms(
        ox_lo_term:   __m128i, // 4 * i32: ox[0..4]*K_x, wrapping
        ox_hi_term:   __m128i, // 4 * i32: ox[4..8]*K_x, wrapping
        oz_lo_z_term: __m256i, // 4 * i64: oz[0..4]*K_z
        oz_hi_z_term: __m256i, // 4 * i64: oz[4..8]*K_z
        dlo:    i64,
        dhi:    i64,
        blocks: &super::Blocks,
    ) -> u8 {
        // SAFETY: caller guarantees AVX2 is available.
        unsafe {
            let mut active_lo: u8 = 0x0F;
            let mut active_hi: u8 = 0x0F;

            let bx_s  = &blocks.bx_hash_term[..];
            let bz_s  = &blocks.bz_hash_term[..];
            let y_s   = &blocks.y[..];
            let thr_s = &blocks.prob_threshold[..];
            let sbb_s = &blocks.sbb_xor[..];
            let n     = bx_s.len();
            debug_assert_eq!(n, bz_s.len(),  "Blocks field length mismatch: bz_hash_term");
            debug_assert_eq!(n, y_s.len(),   "Blocks field length mismatch: y");
            debug_assert_eq!(n, thr_s.len(), "Blocks field length mismatch: prob_threshold");
            debug_assert_eq!(n, sbb_s.len(), "Blocks field length mismatch: sbb_xor");

            // Split on y/threshold uniformity so _mm256_set1 broadcasts are hoisted
            // outside the block loop. `$i` is a macro parameter so it shares hygiene
            // context with the `for i in 0..n` in each match arm.
            macro_rules! block_body {
                ($i:expr, $y_expr:expr, $thr_expr:expr) => {{
                    let bx_k  = _mm_set1_epi32(bx_s[$i]);
                    let x_lo  = _mm_add_epi32(ox_lo_term, bx_k);
                    let x_hi  = _mm_add_epi32(ox_hi_term, bx_k);

                    let bz_term   = _mm256_set1_epi64x(bz_s[$i]);
                    let term_z_lo = _mm256_add_epi64(oz_lo_z_term, bz_term);
                    let term_z_hi = _mm256_add_epi64(oz_hi_z_term, bz_term);

                    let sbb = sbb_s[$i];

                    active_lo &= is_bedrock_x4_precomputed(dlo, dhi, x_lo, $y_expr, term_z_lo, $thr_expr, sbb);
                    active_hi &= is_bedrock_x4_precomputed(dlo, dhi, x_hi, $y_expr, term_z_hi, $thr_expr, sbb);

                    if (active_lo | active_hi) == 0 { return 0; }
                }};
            }
            match (blocks.uniform_y, blocks.uniform_threshold) {
                (Some(y), Some(thr)) => {
                    for i in 0..n { block_body!(i, y, thr); }
                }
                (Some(y), None) => {
                    for i in 0..n { block_body!(i, y, thr_s[i]); }
                }
                (None, Some(thr)) => {
                    for i in 0..n { block_body!(i, y_s[i], thr); }
                }
                (None, None) => {
                    for i in 0..n { block_body!(i, y_s[i], thr_s[i]); }
                }
            }

            active_lo | (active_hi << 4)
        }
    }

}

// SIMD kernel (AVX-512, x86-64 only)
//
// Processes 8 (ox, oz) pairs simultaneously using AVX-512F/DQ 512-bit registers
// (8 * 64-bit lanes). Advantages over the AVX2 path:
//
//  - mullo_epi64   native single instruction (_mm512_mullo_epi64, AVX-512DQ).
//  - mask output   _mm512_cmpgt_epu64_mask returns a u8 k-register directly.
//  - pack          _mm512_cvtepi64_epi32 packs 8 * i64 -> 8 * i32 in one step.
//  - guard_zero    omitted (dhi is never zero in practice).
//
// On Ice Lake / Zen 4 and newer this gives a theoretical 2x throughput
// improvement over the AVX2 path for the inner kernel.
//
// Detection: avx512f + avx512dq.  Groups are always 8 positions wide.

#[cfg(target_arch = "x86_64")]
mod simd_avx512 {
    // NOTE: `use ::core::arch::...` uses an absolute path to the standard library's
    // `core` crate, bypassing the parent module also named `core`.
    use ::core::arch::x86_64::*;

    /// Rotate each 64-bit lane left by 17 bits (AVX-512F).
    ///
    /// # Safety
    /// Caller must guarantee AVX-512F is available. All intrinsics here are safe
    /// under `target_feature_11` within this `#[target_feature(enable = "avx512f")]` fn.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn rotl17_epi64(x: __m512i) -> __m512i {
        _mm512_rol_epi64(x, 17)
    }

    /// Compute `math_hash(x[i], y, z[i])` for i in 0..8 simultaneously.
    ///
    /// `term_x` is a `__m512i` holding the already-computed per-lane x hash
    /// term (8 * i64): `(ox[lane] + bx).wrapping_mul(3_129_871) as i64`,
    /// i.e. the i32 multiply has already been done and sign-extended.
    ///
    /// `term_z` is a `__m512i` holding `(oz[lane] as i64) * 116_129_781 + bz_hash_term[block]`.
    /// The z-term is safe to hoist and add because z uses a true i64 multiply.
    ///
    /// `y` is broadcast as a 64-bit constant across all 8 lanes.
    ///
    /// # Safety
    /// Caller must guarantee AVX-512F, AVX-512DQ, and AVX2 are available. All
    /// intrinsics here are safe under `target_feature_11` within this fn; no
    /// raw-pointer or user-unsafe-fn operations are present.
    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    #[inline]
    unsafe fn math_hash_x8(term_x: __m512i, y: i32, term_z: __m512i) -> __m512i {
        let y64   = _mm512_set1_epi64(y as i64);
        let mut l = _mm512_xor_si512(_mm512_xor_si512(term_x, term_z), y64);

        // l^2 * K + l * 11  =  l * (l * K + 11) - two mullo_epi64 instead of three.
        // Wrapping 64-bit multiplication distributes over addition (mod 2^64),
        // so the result is identical to the original three-multiply form.
        let inner = _mm512_add_epi64(
            _mm512_mullo_epi64(l, _mm512_set1_epi64(42_317_861_i64)),
            _mm512_set1_epi64(11_i64),
        );
        l = _mm512_mullo_epi64(l, inner);

        // l >> 16 (arithmetic / signed).  MUST match the scalar `l >> 16` on i64.
        // AVX-512F provides _mm512_srai_epi64 natively; no emulation needed.
        _mm512_srai_epi64(l, 16)
    }

    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    #[inline]
    unsafe fn is_bedrock_x8(
        dlo:            i64,
        dhi:            i64,
        term_x:         __m512i,  // 8 * i64: (ox+bx)*K_x as i32, sign-extended
        y:              i32,
        term_z:         __m512i,  // 8 * i64: oz*K_z + bz_hash_term per lane
        prob_threshold: u64,
        sbb_xor:        u8,       // 0x00 = bedrock expected, 0xFF = non-bedrock expected
    ) -> u8 {
        // SAFETY: caller guarantees AVX-512F, AVX-512DQ, and AVX2 are available
        // (enforced by #[target_feature]).
        unsafe {
            let hash = math_hash_x8(term_x, y, term_z);
            let s0   = _mm512_xor_si512(hash, _mm512_set1_epi64(dlo));
            let s1   = _mm512_set1_epi64(dhi);

            // guard_zero omitted: dhi is derived from MD5 ^ xoroshiro output;
            // it is never zero in practice.
            debug_assert_ne!(dhi, 0, "deriver hi seed must be non-zero");

            // xoroshiro128++ single step: result = (s0 + s1).rotate_left(17) + s0
            let sum    = _mm512_add_epi64(s0, s1);
            let result = _mm512_add_epi64(rotl17_epi64(sum), s0);

            // Integer threshold comparison replaces the float conversion chain.
            //
            // Original: (result >> 40) as f32 * 2^-24 < probability
            // Equivalent: (result >> 40) < probability * 2^24  (= prob_threshold)
            //
            // This holds exactly because probability * 2^24 is never an integer
            // for the bedrock probabilities used here (e.g. 0.8 * 16777216 = 13421772.8).
            //
            // _mm512_cmpgt_epu64_mask(thresh, top24): bit i set when thresh > top24[i]
            //   i.e. top24[i] < thresh  ==  position i is bedrock.
            let top24: __m512i  = _mm512_srli_epi64(result, 40);
            let thresh: __m512i = _mm512_set1_epi64(prob_threshold as i64);
            let bedrock_mask: u8 = _mm512_cmpgt_epu64_mask(thresh, top24);

            // Branchless inversion via precomputed XOR mask (optimisation 12).
            // sbb_xor == 0x00 -> bedrock expected -> pass mask = bedrock_mask.
            // sbb_xor == 0xFF -> non-bedrock expected -> pass mask = !bedrock_mask.
            bedrock_mask ^ sbb_xor
        }
    }

    /// Precompute the position-dependent hash terms for 8 spiral positions.
    ///
    /// Returns `(ox_term_i32: __m256i, oz_term_v: __m512i)` - the hoisted x and z
    /// multiplications that depend only on `(xs, zs)`, not on any block list.
    /// When multiple rotations are checked for the same group of 8 positions,
    /// calling this once and forwarding the results to `check_formation_x8_with_terms`
    /// avoids recomputing the same two multiplications for every rotation.
    ///
    /// # Safety
    /// Requires AVX-512F, AVX-512DQ, and AVX2.
    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    #[inline]
    pub unsafe fn position_terms_x8(
        positions_x: &[i32; 8],
        positions_z: &[i32; 8],
    ) -> (__m256i, __m512i) {
        // SAFETY: caller guarantees AVX-512F, AVX-512DQ, and AVX2 are available.
        unsafe {
            let ox_v = _mm256_loadu_si256(positions_x.as_ptr() as *const __m256i);
            let oz_v = _mm256_loadu_si256(positions_z.as_ptr() as *const __m256i);
            let oz64      = _mm512_cvtepi32_epi64(oz_v);
            let oz_term_v = _mm512_mullo_epi64(oz64, _mm512_set1_epi64(116_129_781_i64));
            let ox_term_i32 = _mm256_mullo_epi32(ox_v, _mm256_set1_epi32(3_129_871_i32));
            (ox_term_i32, oz_term_v)
        }
    }

    /// Inner formation check that accepts pre-computed position terms (optimisation 2).
    ///
    /// Accepts `ox_term_i32` and `oz_term_v` already produced by `position_terms_x8`,
    /// eliminating the two SIMD multiplications that would otherwise be re-executed
    /// for every rotation when `search_all_rotations` is active.
    ///
    /// # Safety
    /// Requires AVX-512F + AVX-512DQ + AVX2.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    pub unsafe fn check_formation_x8_with_terms(
        ox_term_i32: __m256i, // 8 * i32: ox*K_x in wrapping i32 space, pre-computed
        oz_term_v:   __m512i, // 8 * i64: oz*K_z, pre-computed
        dlo:    i64,
        dhi:    i64,
        blocks: &super::Blocks,
    ) -> u8 {
        // SAFETY: caller guarantees AVX-512F, AVX-512DQ, and AVX2 are available.
        unsafe {
            let mut active: u8 = 0xFF; // bits 0-7 all set = all lanes in play

            // Optimisation 11: extract all five SoA slices before the loop so LLVM
            // can prove they are all of equal length `n` and hoist the five
            // independent bounds checks out of the loop entirely.
            let bx_s  = &blocks.bx_hash_term[..];
            let bz_s  = &blocks.bz_hash_term[..];
            let y_s   = &blocks.y[..];
            let thr_s = &blocks.prob_threshold[..];
            let sbb_s = &blocks.sbb_xor[..];
            let n     = bx_s.len();
            debug_assert_eq!(n, bz_s.len(),  "Blocks field length mismatch: bz_hash_term");
            debug_assert_eq!(n, y_s.len(),   "Blocks field length mismatch: y");
            debug_assert_eq!(n, thr_s.len(), "Blocks field length mismatch: prob_threshold");
            debug_assert_eq!(n, sbb_s.len(), "Blocks field length mismatch: sbb_xor");

            // Split on y/threshold uniformity so _mm512_set1_epi64 broadcasts are
            // hoisted outside the block loop. `$i` is a macro parameter so it shares
            // hygiene context with the `for i in 0..n` in each match arm.
            macro_rules! block_body {
                ($i:expr, $y_expr:expr, $thr_expr:expr) => {{
                    let abs_x_i32 = _mm256_add_epi32(ox_term_i32, _mm256_set1_epi32(bx_s[$i]));
                    let term_x    = _mm512_cvtepi32_epi64(abs_x_i32);
                    let term_z    = _mm512_add_epi64(oz_term_v, _mm512_set1_epi64(bz_s[$i]));

                    let passed = is_bedrock_x8(dlo, dhi, term_x, $y_expr, term_z, $thr_expr, sbb_s[$i]);
                    active &= passed;
                    if active == 0 { return 0; }
                }};
            }
            match (blocks.uniform_y, blocks.uniform_threshold) {
                (Some(y), Some(thr)) => {
                    for i in 0..n { block_body!(i, y, thr); }
                }
                (Some(y), None) => {
                    for i in 0..n { block_body!(i, y, thr_s[i]); }
                }
                (None, Some(thr)) => {
                    for i in 0..n { block_body!(i, y_s[i], thr); }
                }
                (None, None) => {
                    for i in 0..n { block_body!(i, y_s[i], thr_s[i]); }
                }
            }
            active
        }
    }

}

// SIMD dispatch level
//
// Detected once at startup in run_search and captured by the parallel closure,
// eliminating repeated is_x86_feature_detected! calls (each a function call +
// atomic load + branch) on every group iteration.

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdLevel {
    /// AVX-512F + AVX-512DQ: process 8 positions per group.
    Avx512,
    /// AVX2: process 4 positions per group.
    Avx2,
    /// No SIMD: scalar one-at-a-time fallback.
    Scalar,
}

fn detect_simd() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

// Parallel search dispatch
//
// simd is determined once at the start of run_search (see detect_simd) so
// the branch inside the par_iter closure is always perfectly predicted and
// the compiler can constant-fold it in inlined call sites.
//
// rayon's find_first guarantees the earliest (spiral-order) match, cancels
// other workers on a hit, and never wastes threads on empty ranges.  After a
// SIMD group hit, a scalar scan of up to 8 positions pinpoints the exact first
// match in spiral order.

// O(1) spiral coordinate formula
//
// The spiral follows: 1R, 1U, 2L, 2D, 3R, 3U, 4L, 4D, ...
// (Right = +x, Up = +z, Left = -x, Down = -z)
//
// Decomposition into shells and sides gives a closed-form (x, z) for any
// index k without simulating prior positions:
//
//   k = 0            -> (start_x, start_z)
//   k >= 1            -> shell L = floor((1 + sqrt(k)) / 2),
//                       offset j = k - (4L^2 - 4L + 1) within the shell,
//                       then one of four legs (Up / Left / Down / Right).
//
// Shell L occupies indices [4L^2-4L+1, 4L^2+4L] and has 8L positions:
//   Leg 0 Up   (2L-1 steps): dx = L,  dz = -(L-1)+j
//   Leg 1 Left (2L   steps): dx = L-(j-(2L-1)),  dz = L
//   Leg 2 Down (2L   steps): dx = -L, dz = L-(j-(4L-1))
//   Leg 3 Right(2L+1 steps): dx = -L+(j-(6L-1)), dz = -L
//
// This matches the exact spiral convention in the original streaming loop.

/// Translate shell index `l` and intra-shell offset `j` to a (dx, dz) displacement.
/// Extracted from `spiral_coords` so it can be reused by `fill_group`.
#[inline(always)]
fn coords_from_lj(l: i64, j: i64) -> (i64, i64) {
    if j < 2*l - 1 {
        // Leg 0: Up (+z).  Starts at (L, -(L-1)).
        (l, -(l-1) + j)
    } else if j < 4*l - 1 {
        // Leg 1: Left (-x).  Starts at (L, L).
        let o = j - (2*l - 1);
        (l - o, l)
    } else if j < 6*l - 1 {
        // Leg 2: Down (-z).  Starts at (-L, L).
        let o = j - (4*l - 1);
        (-l, l - o)
    } else {
        // Leg 3: Right (+x).  Starts at (-L, -L).
        let o = j - (6*l - 1);
        (-l + o, -l)
    }
}

/// Like `spiral_coords` but also returns the shell `l`, intra-shell offset `j`,
/// and the current step direction `(dx, dz)` together with the first `j` of the
/// next leg (`next_leg_j`), so that `fill_group_from_state` can initialise its
/// threaded direction state without an extra `leg_state` call.
///
/// Returns `(x, z, l, j, dx, dz, next_leg_j)`.  For `k == 0` the sentinel
/// `(start_x, start_z, 0, -1, 0, 1, 0)` is returned; `fill_group_from_state`
/// detects the `l == 0` case via the `j >= 8*l` test (0 >= 0 is true) and
/// correctly advances to shell 1, overwriting dx/dz/next_leg_j.
#[inline(always)]
fn spiral_coords_with_state(k: i64, start_x: i32, start_z: i32) -> (i32, i32, i64, i64, i32, i32, i64) {
    if k == 0 { return (start_x, start_z, 0, -1, 0, 1, 0); }

    let l = ((1.0 + (k as f64).sqrt()) * 0.5) as i64;
    let l = if 4*l*l + 4*l < k { l + 1 } else if 4*l*l - 4*l + 1 > k { l - 1 } else { l };
    let j = k - (4*l*l - 4*l + 1);
    let (dx, dz) = coords_from_lj(l, j);
    let (sdx, sdz, next_leg_j) = leg_state(l, j);
    (start_x + dx as i32, start_z + dz as i32, l, j, sdx, sdz, next_leg_j)
}


/// Derive the current step direction `(dx, dz)` and the first `j` value of the
/// next leg (`next_leg_j`) from the shell/offset pair `(l, j)`.
///
/// Called once per chunk to initialise the threaded direction state, and again
/// only on the rare leg-transition branch inside `fill_group_from_state`.
/// Eliminates the repeated multiplications that `step_direction` performed on
/// every single step.
#[inline(always)]
fn leg_state(l: i64, j: i64) -> (i32, i32, i64) {
    if      j <= 2*l - 1 { ( 0,  1, 2*l) }
    else if j <= 4*l - 1 { (-1,  0, 4*l) }
    else if j <= 6*l - 1 { ( 0, -1, 6*l) }
    else                 { ( 1,  0, 8*l) }
}

/// Fill `xs` and `zs` with the 8 spiral positions of the current group, then
/// advance `(*x, *z, *l, *j)` to position 0 of the **next** group - with no
/// `f64::sqrt` call.
///
/// # Entry contract
/// On entry `(*x, *z)` must be position 0 of the current group and `(*l, *j)`
/// its shell/intra-shell-offset, exactly as produced by
/// `spiral_coords_with_state`.  (The k = 0 sentinel `l = 0, j = -1` is also
/// accepted; the first step correctly advances to shell 1.)
///
/// # Exit contract
/// On exit `(*x, *z, *l, *j)` is the entry state for the immediately
/// following group.  Calling this function repeatedly in a loop steps through
/// consecutive groups with one `spiral_coords_with_state` call (one sqrt) per
/// entire chunk rather than one per group.
///
/// # Implementation
/// The body is identical to the inner loop of `fill_group` (positions 1-7),
/// followed by one additional step that advances to position 0 of the next
/// group.  No sqrt is needed because positions within a shell are a constant
/// unit step apart, and shell transitions are detected by the cheap integer
/// test `j >= 8 * l`.
#[inline(always)]
fn fill_group_from_state(
    x: &mut i32, z: &mut i32,
    l: &mut i64, j: &mut i64,
    dx: &mut i32, dz: &mut i32,       // current step direction (threaded as state)
    next_leg_j: &mut i64,             // first j of the next leg (threaded as state)
    xs: &mut [i32; 8], zs: &mut [i32; 8],
    start_x: i32, start_z: i32,
) {
    xs[0] = *x;
    zs[0] = *z;

    // Compute shell_end = 8 * *l once at entry and refresh only on shell transitions,
    // avoiding a multiplication on every iteration.
    let mut shell_end = 8 * *l;

    for i in 1..8 {
        *j += 1;
        if *j >= shell_end {
            // Shell boundary (also handles the k == 0 sentinel where l == 0).
            *l += 1;
            shell_end = 8 * *l;  // update once on shell transition
            *j  = 0;
            xs[i] = start_x + *l as i32;
            zs[i] = start_z - (*l - 1) as i32;
            // New shell always starts on Leg 0 (Up, +z); next leg boundary at 2l.
            (*dx, *dz, *next_leg_j) = (0, 1, 2 * *l);
        } else {
            // Common path: one comparison, no multiplication.
            if *j >= *next_leg_j {
                (*dx, *dz, *next_leg_j) = leg_state(*l, *j);
            }
            xs[i] = xs[i - 1] + *dx;
            zs[i] = zs[i - 1] + *dz;
        }
    }

    // One more step: advance state to position 0 of the next group.
    *j += 1;
    if *j >= shell_end {
        *l += 1;
        *j  = 0;
        *x = start_x + *l as i32;
        *z = start_z - (*l - 1) as i32;
        (*dx, *dz, *next_leg_j) = (0, 1, 2 * *l);
    } else {
        if *j >= *next_leg_j {
            (*dx, *dz, *next_leg_j) = leg_state(*l, *j);
        }
        *x = xs[7] + *dx;
        *z = zs[7] + *dz;
    }
}

// Generic chunk-batch helper.
//
// Making this a generic function lets the compiler produce separate machine-code
// paths for each SIMD level, with `uniform_fill` and `check_group` fully inlined
// and no dead branches in the hot loop.
//
// `uniform_fill(x, z, dx, dz, xs, zs)` is called on the fast path when the
// entire chunk lies within one spiral leg. `check_group(xs, zs)` returns true
// if any position in the group matches. `cancel` is checked at the start of
// each chunk closure; returning false drains workers immediately, bounding
// cancel latency to one chunk duration regardless of batch size.

#[inline(always)]
fn run_chunk_batch<Fill, Check>(
    batch_base:   i64,
    total_chunks: i64,   // chunks to dispatch in this single find_first call
    start_x:    i32,
    start_z:    i32,
    cancel:     &AtomicBool,
    uniform_fill: &Fill,
    check_group:  &Check,
) -> Option<i64>
where
    Fill:  Fn(i32, i32, i32, i32, &mut [i32; 8], &mut [i32; 8]) + Sync,
    Check: Fn(&[i32; 8], &[i32; 8]) -> bool + Sync,
    {
        (0..total_chunks).into_par_iter().find_first(|&ci| {
            // Fast-path exit: if the user cancelled (or another super-batch found a
            // result and set the flag), skip all work in this chunk and return false.
            // All workers drain within one chunk (~8 192 positions, a few us), so
            // find_first returns None and the outer loop hits the cancel check.
            if cancel.load(Ordering::Relaxed) { return false; }

            let chunk_base_group = batch_base + ci * GROUPS_PER_CHUNK;
            let base_k = chunk_base_group * 8;

            let (mut x, mut z, mut l, mut j, mut dx, mut dz, mut next_leg_j) =
                spiral_coords_with_state(base_k, start_x, start_z);

            let mut xs = [0i32; 8];
            let mut zs = [0i32; 8];

            let chunk_end_j = j + GROUPS_PER_CHUNK as i64 * 8;
            let uniform = l > 0 && chunk_end_j < next_leg_j && chunk_end_j < 8 * l;

            if uniform {
                // Fast path: entire chunk lies within one leg - direction is constant.
                // Branch hoisted out of the loop so the inner loop contains only the
                // uniform fill, with no dead code or repeated branch evaluation.
                for _ in 0..GROUPS_PER_CHUNK {
                    uniform_fill(x, z, dx, dz, &mut xs, &mut zs);
                    x += 8 * dx;
                    z += 8 * dz;
                    if check_group(&xs, &zs) { return true; }
                }
            } else {
                for _ in 0..GROUPS_PER_CHUNK {
                    fill_group_from_state(
                        &mut x, &mut z, &mut l, &mut j,
                        &mut dx, &mut dz, &mut next_leg_j,
                        &mut xs, &mut zs,
                        start_x, start_z,
                    );
                    if check_group(&xs, &zs) { return true; }
                }
            }
            false
        })
    }

// Block-level rotation helpers

/// Rotate a set of relative block offsets by `times_cw` quarter-turns clockwise,
/// then normalise so the minimum X and Z coordinates are both 0.
///
/// Rotation formulae (standard 2-D, with X east and Z south):
///   0    -> ( x,  z)
///   1x CW -> (-z,  x)
///   2x CW -> (-x, -z)
///   3x CW -> ( z, -x)
fn rotate_blocks(blocks: &[Block], times_cw: u8) -> Vec<Block> {
    if blocks.is_empty() { return vec![]; }
    // Single pass: build the output Vec<Block> directly, then fix up coordinates
    // in a second pass - eliminating the intermediate Vec<(i32, i32)> and the zip.
    let mut result: Vec<Block> = Vec::with_capacity(blocks.len());
    let mut min_x = i32::MAX;
    let mut min_z = i32::MAX;
    for b in blocks {
        let (tx, tz) = match times_cw % 4 {
            0 => ( b.x,  b.z),
            1 => (-b.z,  b.x),
            2 => (-b.x, -b.z),
            3 => ( b.z, -b.x),
            _ => unreachable!(),
        };
        if tx < min_x { min_x = tx; }
        if tz < min_z { min_z = tz; }
        result.push(Block { x: tx, z: tz, ..*b });
    }
    for b in &mut result {
        b.x -= min_x;
        b.z -= min_z;
    }
    result
}

/// Canonical signature for deduplication: a sorted list of (x, y, z, is_bedrock)
/// tuples. An inline-capacity-64 SmallVec keeps this on the stack and avoids
/// heap allocations during startup deduplication.
fn blocks_signature(blocks: &[Block]) -> SmallVec<[(i32, i32, i32, bool); 64]> {
    let mut sig: SmallVec<[(i32, i32, i32, bool); 64]> = blocks
        .iter()
        .map(|b| (b.x, b.y, b.z, b.should_be_bedrock))
        .collect();
    sig.sort_unstable();
    sig
}

/// Return up to 4 distinct rotations of `blocks` (fewer if the pattern has
/// rotational symmetry, e.g. a symmetric 2-rotation pattern yields only 2).
pub fn generate_rotations(blocks: Vec<Block>) -> Vec<Vec<Block>> {
    let mut seen: Vec<SmallVec<[(i32, i32, i32, bool); 64]>> = Vec::with_capacity(4);
    let mut rotations: Vec<Vec<Block>> = Vec::with_capacity(4);
    for r in 0..4u8 {
        let rotated = rotate_blocks(&blocks, r);
        let sig = blocks_signature(&rotated);
        if !seen.contains(&sig) {
            seen.push(sig);
            rotations.push(rotated);
        }
    }
    rotations
}

/// Format an area label for a precomputed spiral shell number `l`
/// (the spiral's bounding square at shell `l` has side `2l+1`).
///
/// The shell `l` for spiral index `k` is `floor((1 + sqrt(k)) / 2)`, but
/// computing it via `sqrt` on every progress callback is wasteful since
/// `k` (== `batch_start_group * 8`) only grows monotonically by
/// `GROUPS_PER_BATCH * 8` each step. `Message::SearchProgress` instead
/// tracks `l` incrementally with integer arithmetic and calls this function
/// purely to format the result.
pub fn area_label_from_l(l: i64) -> String {
    let side = (2 * l + 1) as u64; // positions per axis
                                   // Round to the nearest friendly unit
    let fmt_n = |n: u64| -> String {
        if n >= 1_000_000 { format!("{}M", (n + 500_000) / 1_000_000) }
        else if n >= 1_000 { format!("{}k", (n + 500) / 1_000) }
        else               { format!("{}", n) }
    };
    let s = fmt_n(side);
    format!("{} x {}", s, s)
}

// Wraps the original spiral search loop with a cancellation flag checked once
// per chunk.  Returns Ok(Some((x, z, resume_group))) on success where
// `resume_group` is the spiral group index to pass back in as `start_group`
// on a subsequent call in order to keep searching *past* this match (used to
// find duplicate/N-th occurrences of the same pattern), Ok(None) if
// cancelled, or Err if the block constraints are impossible.
pub fn run_search(
    seed: i64,
    start_x: i32,
    start_z: i32,
    bt: BedrockType,
    // One block-set per rotation to test (1 entry for a single-orientation
    // search, up to 4 when `search_all_rotations` is enabled). The spiral is
    // traversed exactly once; every position is checked against *all*
    // rotations before moving on, so the cost of the merged search is
    // independent of how many rotations are supplied (aside from the
    // per-block formation checks themselves).
    rotations: Vec<Vec<Block>>,
    cancel: Arc<AtomicBool>,
    // Called after each batch with the spiral index at the end of that batch.
    // Pass `None` if progress reporting is not needed.
    progress_cb: Option<&dyn Fn(i64)>,
    // Optional GPU context. When Some, the GPU handles the coarse search;
    // the CPU scalar walk is used only for hit confirmation. When None,
    // the existing SIMD/scalar CPU path is used exclusively.
    gpu_ctx: Option<Arc<GpuContext>>,
    // Spiral group index to begin searching from. Pass 0 to search from the
    // configured center outward as usual; pass the `resume_group` returned by
    // a previous call to continue past a match already found (so repeated
    // calls walk the spiral forward and surface successive, non-overlapping
    // occurrences instead of re-finding the same one).
    start_group: i64,
) -> Result<Option<(i32, i32, i64)>, String> {
    if rotations.is_empty() { return Ok(Some((start_x, start_z, start_group + 1))); }

    // Validate, sort, and filter each rotation's block list independently.
    // A rotation rearranges block *positions* but the per-block validity
    // rules (and the probability-based ordering that maximises early-exit
    // effectiveness) apply the same way to each rotation.
    let mut blocks_per_rotation: Vec<Vec<Block>> = Vec::with_capacity(rotations.len());
    for mut blocks in rotations {
        for b in &blocks {
            if b.probability >= 1.0 && !b.should_be_bedrock {
                return Err(format!(
                        "Block ({},{},{}) is always bedrock but declared as air. No solution exists.",
                        b.x, b.y, b.z));
            }
            if b.probability <= 0.0 && b.should_be_bedrock {
                return Err(format!(
                        "Block ({},{},{}) is never bedrock but declared as bedrock. No solution exists.",
                        b.x, b.y, b.z));
            }
        }

        blocks.sort_by_cached_key(|b| {
            let key = if b.should_be_bedrock { 1.0 - clamp01(b.probability) } else { clamp01(b.probability) };
            std::cmp::Reverse(key.to_bits())
        });

        let blocks: Vec<Block> = blocks.into_iter().filter(|b| {
            let p = clamp01(b.probability);
            if b.should_be_bedrock { p < 1.0 } else { p > 0.0 }
        }).collect();

        // If any single rotation has no remaining constraints, every position
        // (including the start) trivially satisfies it.
        if blocks.is_empty() { return Ok(Some((start_x, start_z, start_group + 1))); }

        blocks_per_rotation.push(blocks);
    }

    // Build the GPU block list from ALL rotations' blocks (written once per
    // run_search call, before the hot search loop begins).
    //
    // All rotation-sets are stored back-to-back in a single flat Vec:
    //   [rot0 blocks ..., rot1 blocks ..., rot2 blocks ..., rot3 blocks ...]
    //
    // The GPU kernel checks every rotation-set and reports a hit when any of
    // them is satisfied, mirroring the CPU's `blocks_list.iter().any` logic.
    let gpu_rotation_count = blocks_per_rotation.len() as u32;
    let gpu_blocks: Option<Vec<GpuBlock>> = gpu_ctx.as_ref().map(|_| {
        blocks_per_rotation.iter().flat_map(|rotation| {
            rotation.iter().map(|b| {
                let bz_k = (b.z as i64).wrapping_mul(116_129_781);
                GpuBlock {
                    bx_k:              (b.x as u32).wrapping_mul(3_129_871),
                    by:                b.y,
                    bz_k_lo:           (bz_k as u64 & 0xFFFF_FFFF) as u32,
                    bz_k_hi:           ((bz_k as u64) >> 32) as u32,
                    prob_threshold:    b.prob_threshold as u32,
                    should_be_bedrock: b.should_be_bedrock as u32,
                    _pad:              [0; 2],
                }
            })}).collect()
    });

    let (dlo, dhi) = compute_deriver_seeds(seed, bt);
    let simd       = detect_simd();

    // Convert AoS to SoA here, once per rotation, before the hot search loop.
    // From this point on the SIMD kernels only touch separate contiguous
    // arrays; the `probability: f64` field never appears in the hot path.
    let blocks_list: Vec<Blocks> = blocks_per_rotation.into_iter().map(Blocks::from_vec).collect();

    // Parallel search over the spiral using rayon's find_first.
    //
    // The spiral is divided into groups of 8 positions.  spiral_coords(k, ...)
    // maps any index k to its (x, z) in O(1) with no shared mutable state,
    // so every rayon worker can compute its assigned groups independently.
    //
    // find_first guarantees the *earliest* spiral-order match is returned even
    // when multiple threads discover candidates simultaneously, and cancels
    // remaining workers once the winner is confirmed.
    //
    // `super_batch_chunks` chunks are dispatched in a single `find_first` call
    // to amortise Rayon setup overhead and ensure >= 4 chunks/thread on high-
    // core-count machines. Cancellation is checked at super-batch boundaries.

    // Compute once - queries the Rayon thread pool (cheap after initialisation).
    let super_batch_chunks     = compute_super_batch_chunks();
    let groups_per_super_batch = super_batch_chunks * GROUPS_PER_CHUNK;

    // Define scalar closures outside the loop so their captures are clearly
    // fixed for the lifetime of the search.

    // Scalar uniform-fill closure (used on non-AVX2 path).
    let scalar_fill = |x: i32, z: i32, dx: i32, dz: i32,
    xs: &mut [i32; 8], zs: &mut [i32; 8]| {
        for i in 0..8i32 {
            xs[i as usize] = x + i * dx;
            zs[i as usize] = z + i * dz;
        }
    };
    // Compute ox/oz hash terms once per position, then check each rotation via
    // check_formation_with_terms to avoid redundant multiplications.
    let scalar_check = |xs: &[i32; 8], zs: &[i32; 8]| {
        xs.iter().zip(zs.iter()).any(|(&cx, &cz)| {
            let ox_i32_term = cx.wrapping_mul(3_129_871_i32);
            let oz_i64_term = (cz as i64).wrapping_mul(116_129_781_i64);
            blocks_list.iter().any(|blocks| {
                check_formation_with_terms(ox_i32_term, oz_i64_term, dlo, dhi, blocks)
            })
        })
    };

    let mut batch_start_group: i64 = start_group;

    // Write block data once - it's constant for this entire search.
    if let (Some(ctx), Some(gblocks)) = (gpu_ctx.as_ref(), gpu_blocks.as_deref()) {
        ctx.write_blocks(gblocks);
    }

    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // Snapshot for closure capture (i64 is Copy).
        let batch_base = batch_start_group;

        // SIMD dispatch is resolved once here; run_chunk_batch is monomorphised
        // per path so dead SIMD branches are fully eliminated. Position-dependent
        // hash terms (ox*K_x, oz*K_z) are computed once per group of 8 positions
        // and forwarded to each rotation check, eliminating 3/4 of those
        // multiplications when all 4 rotations are active.

        // Named wrappers are needed because #[target_feature] cannot be applied to
        // closures in stable Rust. The #[inline] wrappers below let the compiler
        // inline the SIMD kernels through the closure boundary.
        #[cfg(target_arch = "x86_64")]
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn avx2_fill(
            x: i32, z: i32, dx: i32, dz: i32,
            xs: &mut [i32; 8], zs: &mut [i32; 8],
        ) {
            // SAFETY: caller guarantees AVX2 is available.
            unsafe { simd_avx2::fill_group_uniform(x, z, dx, dz, xs, zs); }
        }

        // Wrappers for position-term computation, hoisted outside the per-rotation
        // block loop to avoid recomputing the same terms for each rotation.

        #[cfg(target_arch = "x86_64")]
        #[inline]
        #[target_feature(enable = "avx512f,avx512dq,avx2")]
        unsafe fn avx512_position_terms(
            xs: &[i32; 8], zs: &[i32; 8],
        ) -> (::core::arch::x86_64::__m256i, ::core::arch::x86_64::__m512i) {
            unsafe { simd_avx512::position_terms_x8(xs, zs) }
        }

        #[cfg(target_arch = "x86_64")]
        #[inline]
        #[target_feature(enable = "avx512f,avx512dq,avx2")]
        unsafe fn avx512_check_with_terms(
            ox_term: ::core::arch::x86_64::__m256i,
            oz_term: ::core::arch::x86_64::__m512i,
            dlo: i64, dhi: i64, blocks: &Blocks,
        ) -> bool {
            unsafe { simd_avx512::check_formation_x8_with_terms(ox_term, oz_term, dlo, dhi, blocks) != 0 }
        }

        #[cfg(target_arch = "x86_64")]
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn avx2_position_terms(
            xs: &[i32; 8], zs: &[i32; 8],
        ) -> (::core::arch::x86_64::__m128i, ::core::arch::x86_64::__m128i,
        ::core::arch::x86_64::__m256i, ::core::arch::x86_64::__m256i) {
            unsafe { simd_avx2::position_terms_x8_avx2(xs, zs) }
        }

        #[cfg(target_arch = "x86_64")]
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn avx2_check_with_terms(
            oxlo: ::core::arch::x86_64::__m128i,
            oxhi: ::core::arch::x86_64::__m128i,
            ozlo: ::core::arch::x86_64::__m256i,
            ozhi: ::core::arch::x86_64::__m256i,
            dlo: i64, dhi: i64, blocks: &Blocks,
        ) -> bool {
            unsafe { simd_avx2::check_formation_x8_avx2_with_terms(oxlo, oxhi, ozlo, ozhi, dlo, dhi, blocks) != 0 }
        }

        let found_chunk: Option<i64> = if let (Some(ctx), Some(gblocks)) =
            (gpu_ctx.as_ref(), gpu_blocks.as_deref())
        {
            // GPU path
            // The entire super-batch is processed in a single dispatch.
            // Positions are computed inside the shader (closed-form spiral),
            // so the CPU cost here is just a 48-byte uniform write + 4-byte
            // readback - roughly 10-50 us regardless of batch size.
            if cancel.load(Ordering::Relaxed) {
                None
            } else {
                ctx.search_batch(
                    batch_base,
                    super_batch_chunks as usize,
                    start_x, start_z,
                    dlo, dhi,
                    gblocks,
                    gpu_rotation_count,
                )
            }
        } else {
            // CPU path (SIMD / scalar)
            match simd {
                #[cfg(target_arch = "x86_64")]
                SimdLevel::Avx512 => run_chunk_batch(
                    batch_base, super_batch_chunks, start_x, start_z, &cancel,
                    &|x: i32, z: i32, dx: i32, dz: i32,
                    xs: &mut [i32; 8], zs: &mut [i32; 8]| {
                        // SAFETY: AVX2 was verified by detect_simd (AVX-512 => AVX2).
                        unsafe { avx2_fill(x, z, dx, dz, xs, zs); }
                    },
                    &|xs: &[i32; 8], zs: &[i32; 8]| {
                        // Compute position terms once, then check each rotation.
                        // SAFETY: AVX-512F+DQ verified by detect_simd.
                        let (ox_term, oz_term) = unsafe { avx512_position_terms(xs, zs) };
                        blocks_list.iter().any(|blocks| unsafe {
                            avx512_check_with_terms(ox_term, oz_term, dlo, dhi, blocks)
                        })
                    },
                ),
                #[cfg(target_arch = "x86_64")]
                SimdLevel::Avx2 => run_chunk_batch(
                    batch_base, super_batch_chunks, start_x, start_z, &cancel,
                    &|x: i32, z: i32, dx: i32, dz: i32,
                    xs: &mut [i32; 8], zs: &mut [i32; 8]| {
                        // SAFETY: AVX2 verified by detect_simd.
                        unsafe { avx2_fill(x, z, dx, dz, xs, zs); }
                    },
                    &|xs: &[i32; 8], zs: &[i32; 8]| {
                        // Compute position terms once, then check each rotation.
                        // SAFETY: AVX2 verified by detect_simd.
                        let (oxlo, oxhi, ozlo, ozhi) = unsafe { avx2_position_terms(xs, zs) };
                        blocks_list.iter().any(|blocks| unsafe {
                            avx2_check_with_terms(oxlo, oxhi, ozlo, ozhi, dlo, dhi, blocks)
                        })
                    },
                ),
                _ => run_chunk_batch(batch_base, super_batch_chunks, start_x, start_z, &cancel, &scalar_fill, &scalar_check),
            }
        };

        if let Some(ci) = found_chunk {
            // Re-derive the winning chunk's spiral state (one sqrt, cold path)
            // and walk it group-by-group to find the exact matching group, then
            // position-by-position to find the exact first match in spiral order.
            let chunk_base_group = batch_start_group + ci * GROUPS_PER_CHUNK;
            let base_k = chunk_base_group * 8;
            let (mut x, mut z, mut l, mut j, mut dx, mut dz, mut next_leg_j) =
                spiral_coords_with_state(base_k, start_x, start_z);
            let mut xs = [0i32; 8];
            let mut zs = [0i32; 8];
            for g in 0..GROUPS_PER_CHUNK {
                fill_group_from_state(
                    &mut x, &mut z, &mut l, &mut j,
                    &mut dx, &mut dz, &mut next_leg_j,
                    &mut xs, &mut zs,
                    start_x, start_z,
                );
                let hit = xs.iter().zip(zs.iter())
                    .any(|(&cx, &cz)| {
                        let ox_t = cx.wrapping_mul(3_129_871_i32);
                        let oz_t = (cz as i64).wrapping_mul(116_129_781_i64);
                        blocks_list.iter().any(|blocks| check_formation_with_terms(ox_t, oz_t, dlo, dhi, blocks))
                    });
                if hit {
                    // xs and zs are already populated by fill_group_from_state above;
                    // no need to recompute positions via spiral_coords.
                    for p in 0..8usize {
                        let ox_t = xs[p].wrapping_mul(3_129_871_i32);
                        let oz_t = (zs[p] as i64).wrapping_mul(116_129_781_i64);
                        if blocks_list.iter().any(|blocks| check_formation_with_terms(ox_t, oz_t, dlo, dhi, blocks)) {
                            // Resume point for a subsequent call: the next
                            // spiral group after this one. Walking forward by
                            // a whole group (rather than the exact position)
                            // means a second match within the same 8-position
                            // group would be skipped, but that is astronomically
                            // unlikely for any pattern worth searching for.
                            let resume_group = chunk_base_group + g + 1;
                            return Ok(Some((xs[p], zs[p], resume_group)));
                        }
                    }
                    // The scalar group scan confirmed a hit above but no individual
                    // position passed - impossible; indicates a SIMD kernel bug.
                    unreachable!(
                        "SIMD reported a match in chunk {} group {} but scalar \
                         confirmation found no matching position - SIMD kernel bug",
                         ci, g
                    );
                }
            }
            // The SIMD kernel reported a hit in this chunk but the scalar walk
            // found nothing across all groups - spurious SIMD false positive.
            unreachable!(
                "SIMD reported a match in chunk {} but scalar walk found no \
                 matching group - SIMD kernel bug",
                 ci
            );
        }

        batch_start_group += groups_per_super_batch;

        // Report progress: the spiral index at the end of this batch.
        if let Some(cb) = progress_cb {
            cb(batch_start_group * 8);
        }
    }
}
