/// bedrockformation
/// Rust port of the Minecraft Bedrock Formation Finder.
///
/// Usage: bedrockformation <seed> <x:z> <floor|roof> [x,y,z:bedrock ...]
/// Example: bedrockformation 124352345 0:0 floor 0,-63,0:1 1,-62,0:1 0,-63,1:0

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use rayon::prelude::*;

use iced::{
    Application, Command, Element, Event, Length, Settings, Subscription, Theme, Alignment,
    executor, theme,
    keyboard::{self, Key},
    event,
    widget::{button, checkbox, container, horizontal_rule, mouse_area, radio, row, scrollable, text, text_input, Column, Row, Space},
    window,
};

// Constants 

const FLOAT_MULT: f32 = 5.960_464_5e-8_f32; // 2^-24

const FALLBACK_LO: u64 = (-7_046_029_254_386_353_131_i64) as u64;
const FALLBACK_HI: u64 = 7_640_891_576_956_012_809_u64;

// MUST be a multiple of 8: AVX-512 processes groups of 8, AVX2 groups of 4.
// (32_768 satisfies both; it is 2^15.)
//
// With the SoA layout the chunk holds two i32 arrays, so the total buffer size
// is 2 * 4 * CHUNK_SIZE bytes.  At the previous value of 262_144 that was 2 MB,
// well past the L2 cache on most CPUs (typically 256 KB to 1 MB).
//
// 32_768 -> 2 * 4 * 32_768 = 256 KB, comfortably within L2 on virtually all
// modern CPUs, so each chunk stays hot in cache throughout the spiral-fill and
// SIMD scan passes.  The spiral-fill loop is cheap enough that the extra rayon
// task overhead from smaller chunks is negligible.
const CHUNK_SIZE: usize = 32_768;
const _: () = assert!(CHUNK_SIZE % 8 == 0, "CHUNK_SIZE must be a multiple of 8 (AVX-512 requirement)");

// Bedrock type 

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BedrockType { Floor, Roof }

impl BedrockType {
    fn identifier(self) -> &'static str {
        match self {
            BedrockType::Floor => "minecraft:bedrock_floor",
            BedrockType::Roof  => "minecraft:bedrock_roof",
        }
    }
    fn min(self) -> i32 { match self { BedrockType::Floor => -64, BedrockType::Roof => 128 } }
    fn max(self) -> i32 { match self { BedrockType::Floor => -59, BedrockType::Roof => 123 } }
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

// Hash (mirrors MathHelper.hashCode) 
//
// CRITICAL: Java evaluates `(long)(x * 3129871)` as:
//   1. multiply x (int) by 3129871 (int) -> int result (32-bit, wrapping)
//   2. cast that int to long (sign-extend)
//
// Contrast with `(long)z * 116129781L`:
//   1. cast z to long first
//   2. multiply as 64-bit
// So z stays 64-bit; only x uses the 32-bit-first idiom.

#[inline(always)]
fn math_hash(x: i32, y: i32, z: i32) -> i64 {
    let term_x = x.wrapping_mul(3_129_871) as i64;
    let term_z = (z as i64).wrapping_mul(116_129_781_i64);
    let mut l = term_x ^ term_z ^ (y as i64);
    l = l.wrapping_mul(l)
        .wrapping_mul(42_317_861_i64)
        .wrapping_add(l.wrapping_mul(11_i64));
    l >> 16
}

// Core bedrock check (mirrors BedrockReader.inlinedIsBedrock) 

// Stable-Rust approximation of std::intrinsics::unlikely.
// #[cold] tells LLVM this call site is rarely reached; the branch leading here
// is weighted as cold even after inlining, steering branch-prediction hints
// and keeping the hot path in the instruction cache.
#[cold] #[inline(always)] fn cold_true()  -> bool { true  }
#[cold] #[inline(always)] fn cold_false() -> bool { false }

#[inline(always)]
fn is_bedrock(dlo: i64, dhi: i64, x: i32, y: i32, z: i32, probability: f64) -> bool {
    if probability >= 1.0 { return cold_true();  }
    if probability <= 0.0 { return cold_false(); }

    let hash = math_hash(x, y, z);
    let s0 = (hash ^ dlo) as u64;
    let s1 = dhi as u64;

    let (s0, s1) = guard_zero((s0, s1));

    // Single xoroshiro128++ step (read-only, no state struct needed)
    let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);
    // nextFloat(): top 24 bits * 2^-24
    let f = (result >> 40) as f32 * FLOAT_MULT;
    (f as f64) < probability
}

// Probability (mirrors BedrockReader.computeProbability) 

fn compute_probability(y: i32, bt: BedrockType) -> f64 {
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
    let digest = md5::compute(bt.identifier().as_bytes());
    let bs = digest.0;
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
fn prob_to_threshold(probability: f64) -> u64 {
    (probability * 16_777_216.0_f64) as u64
}

struct Block {
    x: i32, y: i32, z: i32,
    should_be_bedrock: bool,
    probability:    f64,
    /// Precomputed integer threshold: `(probability * 2^24) as u64`.
    /// Used by the SIMD kernels to avoid float conversion in the hot path.
    prob_threshold: u64,
}

// Structure-of-Arrays layout for the SIMD hot path 
//
// The original AoS `Block` is 32 bytes (x i32, y i32, z i32, bool + 3-byte
// padding, f64, u64).  When the SIMD inner loop iterates over blocks it pulls
// in the entire struct, carrying the unused `probability: f64` through every
// cache-line fetch.
//
// With SoA, each field lives in its own contiguous array.  The hot SIMD loop
// only touches `x`, `z`, `y`, `prob_threshold`, and `should_be_bedrock`:
// five independent streams that each stay cache-line-local.  The `probability`
// array is only accessed by the scalar confirmation path (at most once per
// chunk hit), so it never pollutes the SIMD working set.
//
// Conversion from the AoS `Vec<Block>` (used for building, sorting, and
// rotating) happens once in `run_search`, just before the search loop begins.
struct Blocks {
    x:                 Vec<i32>,
    y:                 Vec<i32>,
    z:                 Vec<i32>,
    should_be_bedrock: Vec<bool>,
    /// Scalar `probability` retained for the confirmation path in
    /// `check_formation`; never read by the SIMD kernels.
    probability:       Vec<f64>,
    /// Integer threshold `(probability * 2^24) as u64`; the only
    /// floating-point-related field accessed in the SIMD hot loop.
    prob_threshold:    Vec<u64>,
}

impl Blocks {
    fn from_vec(v: Vec<Block>) -> Self {
        Self {
            x:                 v.iter().map(|b| b.x).collect(),
            y:                 v.iter().map(|b| b.y).collect(),
            z:                 v.iter().map(|b| b.z).collect(),
            should_be_bedrock: v.iter().map(|b| b.should_be_bedrock).collect(),
            probability:       v.iter().map(|b| b.probability).collect(),
            prob_threshold:    v.iter().map(|b| b.prob_threshold).collect(),
        }
    }

    #[inline(always)] fn len(&self) -> usize { self.x.len() }
}

// Formation check (mirrors Main.checkFormation) 

#[inline(always)]
fn check_formation(ox: i32, oz: i32, dlo: i64, dhi: i64, blocks: &Blocks) -> bool {
    // Scalar confirmation path: runs at most once per SIMD group hit, so the
    // indexed SoA accesses here are fine, as we are not in the hot loop.
    blocks.should_be_bedrock.iter()
        .zip(blocks.x.iter())
        .zip(blocks.y.iter())
        .zip(blocks.z.iter())
        .zip(blocks.probability.iter())
        .all(|((((sbb, bx), by), bz), prob)| {
            *sbb == is_bedrock(dlo, dhi, ox + bx, *by, oz + bz, *prob)
        })
}

fn clamp01(v: f64) -> f64 { v.clamp(0.0, 1.0) }

// SIMD kernel (AVX2, x86-64 only) 
//
// SIMD batch processing:
//
// The entire inner kernel; math_hash (a few multiplies and XORs),
// xoroshiro128++ (rotate, add, XOR), and float extraction; maps cleanly to
// SIMD with no data dependencies between lanes.
//
// We process 4 (ox, oz) pairs simultaneously using AVX2 256-bit registers
// (4 * 64-bit lanes).  The scalar path is preserved as a fallback for non-AVX2
// hardware and for the exact-position confirmation scan after a SIMD group hit.
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
// - spiral order find_first operates over CHUNK_SIZE/4 groups.  Once a group
//                is found, a scalar scan of up to 4 positions pinpoints the
//                exact first match in spiral order.

#[cfg(target_arch = "x86_64")]
mod simd_avx2 {
    use core::arch::x86_64::*;

    // Helpers 

    /// Low 64 bits of 4-lane 64-bit integer multiply.
    ///
    /// AVX2 has no `mullo_epi64`; we use the identity:
    ///   a*b (low 64) = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32
    ///
    /// `_mm256_mul_epu32` multiplies the low 32 bits of each 64-bit lane
    /// (unsigned), producing an unsigned 64-bit result per lane.
    ///
    /// # Safety
    /// Caller must guarantee AVX2 is available. The arithmetic intrinsics here are
    /// safe to call within this `#[target_feature(enable = "avx2")]` function via
    /// `target_feature_11`; no raw-pointer or user-unsafe-fn operations are present.
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

    /// Compute `math_hash(x[i], y, z[i])` for i in 0..4 simultaneously.
    ///
    /// `x_vec` / `z_vec` are `__m128i` holding 4 * i32 (lane 0 = lowest address).
    /// `y` is the same for all lanes and is broadcast to a 64-bit constant.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn math_hash_x4(x_vec: __m128i, y: i32, z_vec: __m128i) -> __m256i {
        // SAFETY: caller guarantees AVX2 is available (enforced by #[target_feature]).
        unsafe {
            // term_x = x.wrapping_mul(3_129_871) as i64
            // _mm_mullo_epi32 gives the low 32 bits of the i32 product; sign-extend
            // to i64 via _mm256_cvtepi32_epi64, matching Java's (int)(x*3129871) cast.
            let term_x = _mm256_cvtepi32_epi64(
                _mm_mullo_epi32(x_vec, _mm_set1_epi32(3_129_871_i32)),
            );

            // term_z = (z as i64).wrapping_mul(116_129_781)
            // Both z and 116_129_781 fit in i32, so _mm256_mul_epi32, which uses
            // only the low 32 bits of each 64-bit lane (signed * signed -> 64-bit),
            // gives the exact same result as the scalar i64 multiply.
            let z64    = _mm256_cvtepi32_epi64(z_vec);
            let term_z = _mm256_mul_epi32(z64, _mm256_set1_epi64x(116_129_781_i64));

            // l = term_x ^ term_z ^ (y as i64)
            let y64   = _mm256_set1_epi64x(y as i64);
            let mut l = _mm256_xor_si256(_mm256_xor_si256(term_x, term_z), y64);

            // l = l.wrapping_mul(l).wrapping_mul(42_317_861) + l.wrapping_mul(11)
            let l_sq   = mullo_epi64(l, l);
            let l_sq_k = mullo_epi64(l_sq, _mm256_set1_epi64x(42_317_861_i64));
            let l_11   = mullo_epi64(l,    _mm256_set1_epi64x(11_i64));
            l = _mm256_add_epi64(l_sq_k, l_11);

            // l >> 16 (arithmetic / signed).  MUST match the scalar `l >> 16` on i64.
            // When l is negative the top 16 bits are 0xFFFF, not 0x0000; using srli
            // (logical) here produces a wrong hash for any negative l.
            // AVX2 has no _mm256_srai_epi64, so we emulate it:
            //   arithmetic_sra(x, 16) = srli(x, 16) | (sign_mask & 0xFFFF000000000000)
            // where sign_mask is all-ones in lanes where x < 0, all-zeros otherwise.
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
    /// Assumes `probability in (0, 1)`: trivially-guaranteed blocks are
    /// filtered out before the search starts and never reach this path.
    ///
    /// Uses the precomputed integer threshold to avoid float conversion.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn is_bedrock_x4(
        dlo:               i64,
        dhi:               i64,
        x_vec:             __m128i,  // 4 * i32: absolute X coords
        y:                 i32,
        z_vec:             __m128i,  // 4 * i32: absolute Z coords
        prob_threshold:    u64,
        should_be_bedrock: bool,
    ) -> u8 {
        // SAFETY: caller guarantees AVX2 is available (enforced by #[target_feature]).
        unsafe {
            // s0 = (math_hash(x, y, z) ^ dlo) as u64  (per lane)
            // s1 = dhi as u64  (same for every lane, broadcast)
            let hash = math_hash_x4(x_vec, y, z_vec);
            let s0   = _mm256_xor_si256(hash, _mm256_set1_epi64x(dlo));
            let s1   = _mm256_set1_epi64x(dhi);

            // guard_zero removed from SIMD hot path: dhi is derived from MD5 ^ xoroshiro
            // output, making zero cryptographically impossible in practice.  The scalar
            // is_bedrock retains the guard for correctness on theoretical edge cases.
            // Removing the or/cmpeq/blendv trio saves 3 instructions per block per group.
            debug_assert_ne!(dhi, 0, "deriver hi seed must be non-zero");

            // xoroshiro128++ single step: result = (s0 + s1).rotate_left(17) + s0
            let sum    = _mm256_add_epi64(s0, s1);
            let result = _mm256_add_epi64(rotl17_epi64(sum), s0);

            // Integer threshold comparison replaces the float conversion chain.
            //
            // Original: (result >> 40) as f32 * 2^-24 < probability
            // Equivalent: (result >> 40) < probability * 2^24  (= prob_threshold)
            //
            // _mm256_cmpgt_epi64(thresh_v, top24): each lane is all-1s when
            //   thresh > top24[i] (signed, but both values are in [0, 2^24) so
            //   signed == unsigned here). Sign-bit of all-1s lane is 1.
            // _mm256_movemask_pd then extracts one bit per 64-bit lane -> 4-bit mask.
            let top24    = _mm256_srli_epi64(result, 40);
            let thresh_v = _mm256_set1_epi64x(prob_threshold as i64);
            let cmp      = _mm256_cmpgt_epi64(thresh_v, top24);
            let bedrock_mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u8;

            // Reconcile with what the block expects.
            if should_be_bedrock { bedrock_mask } else { !bedrock_mask & 0x0F }
        }
    }

    /// Returns a 4-bit mask of the positions (within a group of 4) that match
    /// the entire formation.  Exits early as soon as no lanes remain active,
    /// mirroring the scalar `.all()` short-circuit.
    ///
    /// All blocks must have `probability in (0, 1)` (trivially-guaranteed blocks
    /// are filtered before the search begins).
    ///
    /// # Safety
    /// Requires AVX2.  Caller must have verified `is_x86_feature_detected!("avx2")`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn check_formation_x4(
        positions_x: &[i32], // exactly 4 entries (contiguous i32 array)
        positions_z: &[i32], // exactly 4 entries (contiguous i32 array)
        dlo:    i64,
        dhi:    i64,
        blocks: &super::Blocks,
    ) -> u8 {
        debug_assert_eq!(positions_x.len(), 4);
        debug_assert_eq!(positions_z.len(), 4);

        // SAFETY: caller guarantees AVX2 is available (enforced by #[target_feature]).
        unsafe {
            // Single 128-bit load replaces 4 scalar moves from _mm_set_epi32.
            // Both slices are contiguous i32 arrays, so loadu_si128 is valid and
            // places element [0] in lane 0, which is the same lane ordering as before.
            let ox_v = _mm_loadu_si128(positions_x.as_ptr() as *const __m128i);
            let oz_v = _mm_loadu_si128(positions_z.as_ptr() as *const __m128i);

            let mut active: u8 = 0x0F; // bits 0-3 all set = all lanes in play

            // SoA hot loop: each array access stays within its own cache-line
            // stream.  No AoS struct padding is dragged into cache per iteration.
            for i in 0..blocks.len() {
                // Absolute coordinates for this block's offset
                let x_v = _mm_add_epi32(ox_v, _mm_set1_epi32(blocks.x[i]));
                let z_v = _mm_add_epi32(oz_v, _mm_set1_epi32(blocks.z[i]));

                let passed = is_bedrock_x4(dlo, dhi, x_v, blocks.y[i], z_v, blocks.prob_threshold[i], blocks.should_be_bedrock[i]);
                active &= passed;
                if active == 0 { return 0; } // no lane can match; exit immediately
            }
            active
        }
    }
}

// SIMD kernel (AVX-512, x86-64 only) 
//
// Processes 8 (ox, oz) pairs simultaneously using AVX-512F/DQ 512-bit registers
// (8 * 64-bit lanes).  Advantages over the AVX2 path:
//
// - mullo_epi64  _mm512_mullo_epi64 is a *native* single instruction (AVX-512DQ).
//               No emulation loop required.
//
// - float cmp   _mm512_cmp_pd_mask returns a u8 k-register mask directly,
//               eliminating the movemask step.
//
// - cvtepi64->32 _mm512_cvtepi64_epi32 packs 8 * i64 -> 8 * i32 in one instruction,
//               replacing the pack_lo32 permute.
//
// - guard_zero  Omitted (same reasoning as AVX2).
//
// On Ice Lake / Zen 4 and newer this is a theoretical 2* throughput improvement
// over the AVX2 path for the inner kernel.
//
// Detection: avx512f + avx512dq.  CHUNK_SIZE (262_144) is already a multiple of 8.

#[cfg(target_arch = "x86_64")]
mod simd_avx512 {
    use core::arch::x86_64::*;

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
    /// `x_vec` / `z_vec` are `__m256i` holding 8 * i32.
    /// `y` is broadcast as a 64-bit constant across all 8 lanes.
    ///
    /// # Safety
    /// Caller must guarantee AVX-512F, AVX-512DQ, and AVX2 are available. All
    /// intrinsics here are safe under `target_feature_11` within this fn; no
    /// raw-pointer or user-unsafe-fn operations are present.
    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    #[inline]
    unsafe fn math_hash_x8(x_vec: __m256i, y: i32, z_vec: __m256i) -> __m512i {
        // term_x = (x.wrapping_mul(3_129_871) as i32) sign-extended to i64
        // _mm256_mullo_epi32: 8 * i32 low-multiply (AVX2)
        // _mm512_cvtepi32_epi64: sign-extend 8 * i32 -> 8 * i64 (AVX-512F)
        let x_mul  = _mm256_mullo_epi32(x_vec, _mm256_set1_epi32(3_129_871_i32));
        let term_x = _mm512_cvtepi32_epi64(x_mul);

        // term_z = (z as i64).wrapping_mul(116_129_781)
        // _mm512_mullo_epi64 is a single native instruction (AVX-512DQ)
        let z64    = _mm512_cvtepi32_epi64(z_vec);
        let term_z = _mm512_mullo_epi64(z64, _mm512_set1_epi64(116_129_781_i64));

        let y64   = _mm512_set1_epi64(y as i64);
        let mut l = _mm512_xor_si512(_mm512_xor_si512(term_x, term_z), y64);

        // l = l*l*42_317_861 + l*11
        let l_sq   = _mm512_mullo_epi64(l, l);
        let l_sq_k = _mm512_mullo_epi64(l_sq, _mm512_set1_epi64(42_317_861_i64));
        let l_11   = _mm512_mullo_epi64(l,    _mm512_set1_epi64(11_i64));
        l = _mm512_add_epi64(l_sq_k, l_11);

        // l >> 16 (arithmetic / signed).  MUST match the scalar `l >> 16` on i64.
        // AVX-512F provides _mm512_srai_epi64 natively; no emulation needed.
        _mm512_srai_epi64(l, 16)
    }

    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    #[inline]
    unsafe fn is_bedrock_x8(
        dlo:               i64,
        dhi:               i64,
        x_vec:             __m256i,  // 8 * i32: absolute X coords
        y:                 i32,
        z_vec:             __m256i,  // 8 * i32: absolute Z coords
        prob_threshold:    u64,
        should_be_bedrock: bool,
    ) -> u8 {
        // SAFETY: caller guarantees AVX-512F, AVX-512DQ, and AVX2 are available
        // (enforced by #[target_feature]).
        unsafe {
            let hash = math_hash_x8(x_vec, y, z_vec);
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
            //   i.e. top24[i] < thresh  ≡  position i is bedrock.
            let top24: __m512i  = _mm512_srli_epi64(result, 40);
            let thresh: __m512i = _mm512_set1_epi64(prob_threshold as i64);
            let bedrock_mask: u8 = _mm512_cmpgt_epu64_mask(thresh, top24);

            if should_be_bedrock { bedrock_mask } else { !bedrock_mask }
        }
    }

    /// Returns an 8-bit mask of the positions (within a group of 8) that match
    /// the entire formation.  Exits early as soon as no lanes remain active.
    ///
    /// # Safety
    /// Requires AVX-512F + AVX-512DQ.  Caller must have verified feature support.
    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    pub unsafe fn check_formation_x8(
        positions_x: &[i32], // exactly 8 entries (contiguous i32 array)
        positions_z: &[i32], // exactly 8 entries (contiguous i32 array)
        dlo:    i64,
        dhi:    i64,
        blocks: &super::Blocks,
    ) -> u8 {
        debug_assert_eq!(positions_x.len(), 8);
        debug_assert_eq!(positions_z.len(), 8);

        // SAFETY: caller guarantees AVX-512F, AVX-512DQ, and AVX2 are available
        // (enforced by #[target_feature]).
        unsafe {
            // Single 256-bit load replaces 8 scalar moves from _mm256_set_epi32.
            // Both slices are contiguous i32 arrays, so loadu_si256 is valid and
            // places element [0] in lane 0, which is the same lane ordering as before.
            let ox_v = _mm256_loadu_si256(positions_x.as_ptr() as *const __m256i);
            let oz_v = _mm256_loadu_si256(positions_z.as_ptr() as *const __m256i);

            let mut active: u8 = 0xFF; // bits 0-7 all set = all lanes in play

            // SoA hot loop: prob_threshold, should_be_bedrock, x, y, z each live in
            // separate contiguous arrays; no AoS padding or unused f64 in the stream.
            for i in 0..blocks.len() {
                let x_v = _mm256_add_epi32(ox_v, _mm256_set1_epi32(blocks.x[i]));
                let z_v = _mm256_add_epi32(oz_v, _mm256_set1_epi32(blocks.z[i]));

                let passed = is_bedrock_x8(dlo, dhi, x_v, blocks.y[i], z_v, blocks.prob_threshold[i], blocks.should_be_bedrock[i]);
                active &= passed;
                if active == 0 { return 0; }
            }
            active
        }
    }
}

// SIMD dispatch level 
//
// Detected once at startup in main() and passed into search_chunk, eliminating
// repeated is_x86_feature_detected! calls (each a function call + atomic load +
// branch) on every chunk iteration.

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

// Chunk search dispatch 
//
// Dispatches to the widest available SIMD path (AVX-512 -> AVX2 -> scalar).
// `simd` is determined once at startup (see detect_simd / main) so the branch
// is always perfectly predicted and the compiler can see it as a compile-time
// constant in inlined call sites.
//
// rayon's `find_first` guarantees the earliest (spiral-order) match, cancels
// other workers on a hit, and never wastes threads on empty ranges.  After a
// SIMD group hit, a scalar scan of up to N positions pinpoints the exact first
// match in spiral order.

fn search_chunk(chunk_x: &[i32], chunk_z: &[i32], dlo: i64, dhi: i64, blocks: &Blocks, simd: SimdLevel) -> Option<usize> {
    debug_assert_eq!(chunk_x.len(), chunk_z.len());
    // All paths iterate sequentially over groups to guarantee spiral order.
    // SIMD kernels provide fast per-group candidate screening; the scalar
    // check_formation confirms the exact position within a matching group.

    #[cfg(target_arch = "x86_64")]
    if simd == SimdLevel::Avx512 {
        let n_groups = chunk_x.len() / 8;
        for g in 0..n_groups {
            let start = g * 8;
            // SAFETY: AVX-512F+DQ verified by detect_simd; slices are exactly 8 elements.
            let mask = unsafe {
                simd_avx512::check_formation_x8(
                    &chunk_x[start..start + 8],
                    &chunk_z[start..start + 8],
                    dlo, dhi, blocks,
                )
            };
            if mask != 0 {
                for j in 0..8 {
                    let cx = chunk_x[start + j];
                    let cz = chunk_z[start + j];
                    if check_formation(cx, cz, dlo, dhi, blocks) {
                        return Some(start + j);
                    }
                }
            }
        }
        return None;
    }

    #[cfg(target_arch = "x86_64")]
    if simd == SimdLevel::Avx2 {
        let n_groups = chunk_x.len() / 4;
        for g in 0..n_groups {
            let start = g * 4;
            // SAFETY: AVX2 verified by detect_simd; slices are exactly 4 elements.
            let mask = unsafe {
                simd_avx2::check_formation_x4(
                    &chunk_x[start..start + 4],
                    &chunk_z[start..start + 4],
                    dlo, dhi, blocks,
                )
            };
            if mask != 0 {
                for j in 0..4 {
                    let cx = chunk_x[start + j];
                    let cz = chunk_z[start + j];
                    if check_formation(cx, cz, dlo, dhi, blocks) {
                        return Some(start + j);
                    }
                }
            }
        }
        return None;
    }

    // Scalar fallback: sequential to guarantee spiral order.
    (0..chunk_x.len()).find(|&i| {
        let cx = chunk_x[i];
        let cz = chunk_z[i];
        check_formation(cx, cz, dlo, dhi, blocks)
    })
}

// Direction (mirrors Main.Direction) 

#[derive(Clone, Copy)]
enum Dir { Left, Right, Up, Down }

impl Dir {
    fn next(self) -> Dir {
        match self {
            Dir::Left  => Dir::Down,  Dir::Right => Dir::Up,
            Dir::Up    => Dir::Left,  Dir::Down  => Dir::Right,
        }
    }
    fn step(self, x: &mut i32, z: &mut i32) {
        match self {
            Dir::Left  => *x -= 1, Dir::Right => *x += 1,
            Dir::Up    => *z += 1, Dir::Down  => *z -= 1,
        }
    }
}

// main

fn main() -> iced::Result {
    App::run(Settings {
        window: window::Settings {
            size: iced::Size::new(800.0, 650.0),
            min_size: Some(iced::Size::new(620.0, 400.0)),
            ..Default::default()
        },
        ..Default::default()
    })
}

// Block-level rotation helpers

/// Rotate a set of relative block offsets by `times_cw` quarter-turns clockwise,
/// then normalise so the minimum X and Z coordinates are both 0.
///
/// Rotation formulae (standard 2-D, with X east and Z south):
///   0º -> (x,  z)
///   1* CW  ->  (−z,  x)
///   2* CW  ->  (−x, −z)
///   3* CW  ->  ( z, −x)
fn rotate_blocks(blocks: &[Block], times_cw: u8) -> Vec<Block> {
    if blocks.is_empty() { return vec![]; }
    let transformed: Vec<(i32, i32)> = blocks.iter().map(|b| {
        match times_cw % 4 {
            0 => ( b.x,  b.z),
            1 => (-b.z,  b.x),
            2 => (-b.x, -b.z),
            3 => ( b.z, -b.x),
            _ => unreachable!(),
        }
    }).collect();
    let min_x = transformed.iter().map(|&(x, _)| x).min().unwrap();
    let min_z = transformed.iter().map(|&(_, z)| z).min().unwrap();
    blocks.iter().zip(transformed.iter()).map(|(b, &(tx, tz))| Block {
        x: tx - min_x,
        z: tz - min_z,
        y: b.y,
        should_be_bedrock: b.should_be_bedrock,
        probability:    b.probability,
        prob_threshold: b.prob_threshold,
    }).collect()
}

/// Canonical signature for deduplication: sorted list of (x, y, z, is_bedrock).
fn blocks_signature(blocks: &[Block]) -> Vec<(i32, i32, i32, bool)> {
    let mut sig: Vec<_> = blocks
        .iter()
        .map(|b| (b.x, b.y, b.z, b.should_be_bedrock))
        .collect();
    sig.sort_unstable();
    sig
}

/// Return up to 4 distinct rotations of `blocks` (fewer if the pattern has
/// rotational symmetry, e.g. a symmetric 2-rotation pattern yields only 2).
fn generate_rotations(blocks: Vec<Block>) -> Vec<Vec<Block>> {
    let mut seen: std::collections::HashSet<Vec<(i32, i32, i32, bool)>> =
        std::collections::HashSet::with_capacity(4);
    let mut rotations: Vec<Vec<Block>> = Vec::with_capacity(4);
    for r in 0..4u8 {
        let rotated = rotate_blocks(&blocks, r);
        let sig = blocks_signature(&rotated);
        if seen.insert(sig) {
            rotations.push(rotated);
        }
    }
    rotations
}

// Grid rotation helpers

/// Rotate all Y-layers 90º clockwise.
/// In the grid, col maps to X and row maps to Z, so CW means: new_col = rows−1−row, new_row = col.
/// The resulting grid has new_rows = old_cols, new_cols = old_rows.
fn rotate_grid_cw(
    cells: &[Vec<Vec<CellState>>],
    rows: usize,
    cols: usize,
) -> (Vec<Vec<Vec<CellState>>>, usize, usize) {
    let new_rows = cols;
    let new_cols = rows;
    let new_cells = cells
        .iter()
        .map(|layer| {
            let mut new_layer = vec![vec![CellState::Unknown; new_cols]; new_rows];
            for r in 0..rows {
                for c in 0..cols {
                    // CW: (r, c) -> new position (c, rows−1−r)
                    new_layer[c][rows - 1 - r] = layer[r][c];
                }
            }
            new_layer
        })
        .collect();
    (new_cells, new_rows, new_cols)
}

/// Rotate all Y-layers 90º counter-clockwise.
/// CCW: (r, c) -> new position (cols−1−c, r).
/// The resulting grid has new_rows = old_cols, new_cols = old_rows.
fn rotate_grid_ccw(
    cells: &[Vec<Vec<CellState>>],
    rows: usize,
    cols: usize,
) -> (Vec<Vec<Vec<CellState>>>, usize, usize) {
    let new_rows = cols;
    let new_cols = rows;
    let new_cells = cells
        .iter()
        .map(|layer| {
            let mut new_layer = vec![vec![CellState::Unknown; new_cols]; new_rows];
            for r in 0..rows {
                for c in 0..cols {
                    // CCW: (r, c) -> new position (cols−1−c, r)
                    new_layer[cols - 1 - c][r] = layer[r][c];
                }
            }
            new_layer
        })
        .collect();
    (new_cells, new_rows, new_cols)
}

// GUI - search runner

// Wraps the original spiral search loop with a cancellation flag checked once
// per chunk.  Returns Ok(Some((x, z))) on success, Ok(None) if cancelled, or
// Err if the block constraints are impossible.
fn run_search(
    seed: i64,
    start_x: i32,
    start_z: i32,
    bt: BedrockType,
    mut blocks: Vec<Block>,
    cancel: Arc<AtomicBool>,
    // Set by the parallel-rotation dispatcher when *another* rotation has
    // already found a match; causes this rotation to bail out early with
    // `Ok(None)` (same as a user cancellation, but without touching `cancel`).
    stop_early: Arc<AtomicBool>,
) -> Result<Option<(i32, i32)>, String> {
    if blocks.is_empty() { return Ok(Some((start_x, start_z))); }

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

    if blocks.is_empty() { return Ok(Some((start_x, start_z))); }

    let (dlo, dhi) = compute_deriver_seeds(seed, bt);
    let simd       = detect_simd();

    // Convert AoS to SoA here, once, before the hot search loop.
    // From this point on the SIMD kernels only touch separate contiguous
    // arrays; the `probability: f64` field never appears in the hot path.
    let blocks = Blocks::from_vec(blocks);

    let mut x   = start_x;
    let mut z   = start_z;
    let mut dir = Dir::Right;
    let mut steps_to_take: i32 = 1;
    let mut steps_taken: i32 = 0;
    let mut sides_until_incremental: i32 = 0;

    let mut chunk_x = vec![0i32; CHUNK_SIZE];
    let mut chunk_z = vec![0i32; CHUNK_SIZE];

    loop {
        // Check both the UI cancel flag and the inter-rotation stop signal.
        if cancel.load(Ordering::Relaxed) || stop_early.load(Ordering::Relaxed) {
            return Ok(None);
        }

        for slot in 0..CHUNK_SIZE {
            chunk_x[slot] = x;
            chunk_z[slot] = z;
            dir.step(&mut x, &mut z);
            steps_taken += 1;
            if steps_taken == steps_to_take {
                steps_taken = 0;
                dir = dir.next();
                sides_until_incremental += 1;
                if sides_until_incremental == 2 {
                    sides_until_incremental = 0;
                    steps_to_take += 1;
                }
            }
        }

        if let Some(idx) = search_chunk(&chunk_x, &chunk_z, dlo, dhi, &blocks, simd) {
            let cx = chunk_x[idx];
            let cz = chunk_z[idx];
            return Ok(Some((cx, cz)));
        }
    }
}

// GUI - types

/// State of one cell in the constraint grid.
/// Cycles Unknown -> NonBedrock -> Bedrock -> Unknown on each click.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum CellState { #[default] Unknown, NonBedrock, Bedrock }

impl CellState {
    fn next(self) -> Self {
        match self {
            CellState::Unknown    => CellState::NonBedrock,
            CellState::NonBedrock => CellState::Bedrock,
            CellState::Bedrock    => CellState::Unknown,
        }
    }
    fn prev(self) -> Self {
        match self {
            CellState::Unknown    => CellState::Bedrock,
            CellState::NonBedrock => CellState::Unknown,
            CellState::Bedrock    => CellState::NonBedrock,
        }
    }
}

/// The four Y values that can contain probabilistic bedrock for each layer type.
/// Ordered left-to-right on the tab strip: most-air end first (-60 … -63 for floor).
/// -64 (always bedrock) and -59 (always air) are excluded as redundant.
fn y_values(bt: BedrockType) -> [i32; 4] {
    match bt {
        BedrockType::Floor => [-60, -61, -62, -63],
        BedrockType::Roof  => [124, 125, 126, 127],
    }
}

/// Allocate a fresh 4-layer * rows * cols grid, all Unknown.
fn make_grid(rows: usize, cols: usize) -> Vec<Vec<Vec<CellState>>> {
    vec![vec![vec![CellState::Unknown; cols]; rows]; 4]
}

#[derive(Debug, Clone, PartialEq)]
enum SearchStatus {
    Idle,
    Searching,
    Cancelled,
    Found(i32, i32),
    Error(String),
}

struct App {
    seed:          String,
    center_x:      String,
    center_z:      String,
    bedrock_type:  BedrockType,
    // Grid dimensions (1-16 each)
    grid_cols:     usize,
    grid_rows:     usize,
    grid_cols_str: String,
    grid_rows_str: String,
    // Which Y-layer tab is active
    grid_y_idx:    usize,
    // Top-left corner offset (relative block coords)
    grid_offset_x: String,
    grid_offset_z: String,
    // [y_layer 0..4][row 0..grid_rows][col 0..grid_cols]
    grid_cells:          Vec<Vec<Vec<CellState>>>,
    /// When true the search tests all 4 rotations of the pattern at every
    /// candidate position, so the result is found regardless of which
    /// compass direction the user was facing when they captured the pattern.
    search_all_rotations: bool,
    status:        SearchStatus,
    cancel_flag:   Option<Arc<AtomicBool>>,
    /// UI zoom level: 1.0 = default, range 0.5-2.0 in steps of 0.1.
    ui_scale:      f32,
}

impl Default for App {
    fn default() -> Self {
        let cols = 8usize;
        let rows = 8usize;
        Self {
            seed:          String::new(),
            center_x:      "0".into(),
            center_z:      "0".into(),
            bedrock_type:  BedrockType::Floor,
            grid_cols:     cols,
            grid_rows:     rows,
            grid_cols_str: cols.to_string(),
            grid_rows_str: rows.to_string(),
            grid_y_idx:    0,
            grid_offset_x: "0".into(),
            grid_offset_z: "0".into(),
            grid_cells:          make_grid(rows, cols),
            search_all_rotations: false,
            status:        SearchStatus::Idle,
            cancel_flag:   None,
            ui_scale:      1.0,
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    SeedChanged(String),
    CenterXChanged(String),
    CenterZChanged(String),
    TypeChanged(BedrockType),
    GridColsChanged(String),
    GridRowsChanged(String),
    GridYChanged(usize),
    GridOffsetXChanged(String),
    GridOffsetZChanged(String),
    /// Cycle the state of cell (row, col) in the active Y-layer.
    GridCellClicked(usize, usize),
    /// Cycle the state of cell (row, col) in reverse (right-click).
    GridCellRightClicked(usize, usize),
    /// Rotate all Y-layers 90º clockwise (X->Z, Z->−X).
    RotateCW,
    /// Rotate all Y-layers 90º counter-clockwise (X->−Z, Z->X).
    RotateCCW,
    /// Toggle whether the search tries all 4 rotations of the pattern.
    ToggleAllRotations(bool),
    Search,
    Cancel,
    SearchDone(Result<Option<(i32, i32)>, String>),
    ZoomIn,
    ZoomOut,
}

// GUI - Application impl

impl Application for App {
    type Message = Message;
    type Theme   = Theme;
    type Executor = executor::Default;
    type Flags   = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        (App::default(), Command::none())
    }

    fn title(&self) -> String { String::from("Bedrock Formation Finder") }

    fn theme(&self) -> Theme { Theme::GruvboxDark }

    fn subscription(&self) -> Subscription<Message> {
        event::listen_with(|event, _| {
            if let Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                modifiers,
                ..
            }) = event
                && modifiers.control()
                && let Key::Character(c) = &key
            {
                return match c.as_str() {
                    "+" | "=" => Some(Message::ZoomIn),
                    "-"       => Some(Message::ZoomOut),
                    _         => None,
                };
            }
            None
        })
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::SeedChanged(s)    => { self.seed     = s; Command::none() }
            Message::CenterXChanged(s) => { self.center_x = s; Command::none() }
            Message::CenterZChanged(s) => { self.center_z = s; Command::none() }
            Message::TypeChanged(t) => {
                // Y values change between floor/roof, so reset the grid.
                self.bedrock_type = t;
                self.grid_cells   = make_grid(self.grid_rows, self.grid_cols);
                self.grid_y_idx   = 0;
                Command::none()
            }

            Message::GridColsChanged(s) => {
                self.grid_cols_str = s.clone();
                if let Ok(n) = s.parse::<usize>() {
                    let n = n.clamp(1, 16);
                    for layer in &mut self.grid_cells {
                        for row in &mut *layer {
                            row.resize(n, CellState::Unknown);
                        }
                    }
                    self.grid_cols = n;
                }
                Command::none()
            }
            Message::GridRowsChanged(s) => {
                self.grid_rows_str = s.clone();
                if let Ok(n) = s.parse::<usize>() {
                    let n = n.clamp(1, 16);
                    for layer in &mut self.grid_cells {
                        layer.resize(n, vec![CellState::Unknown; self.grid_cols]);
                    }
                    self.grid_rows = n;
                }
                Command::none()
            }
            Message::GridYChanged(idx)     => { self.grid_y_idx    = idx; Command::none() }
            Message::GridOffsetXChanged(s) => { self.grid_offset_x = s;   Command::none() }
            Message::GridOffsetZChanged(s) => { self.grid_offset_z = s;   Command::none() }
            Message::GridCellClicked(r, c) => {
                self.grid_cells[self.grid_y_idx][r][c] =
                    self.grid_cells[self.grid_y_idx][r][c].next();
                Command::none()
            }
            Message::GridCellRightClicked(r, c) => {
                self.grid_cells[self.grid_y_idx][r][c] =
                    self.grid_cells[self.grid_y_idx][r][c].prev();
                Command::none()
            }

            Message::RotateCW => {
                let (new_cells, new_rows, new_cols) =
                    rotate_grid_cw(&self.grid_cells, self.grid_rows, self.grid_cols);
                self.grid_cells    = new_cells;
                self.grid_rows     = new_rows;
                self.grid_cols     = new_cols;
                self.grid_rows_str = new_rows.to_string();
                self.grid_cols_str = new_cols.to_string();
                // Keep y-index in bounds (it always stays valid since we never
                // change the number of Y-layers, just rows/cols).
                Command::none()
            }

            Message::RotateCCW => {
                let (new_cells, new_rows, new_cols) =
                    rotate_grid_ccw(&self.grid_cells, self.grid_rows, self.grid_cols);
                self.grid_cells    = new_cells;
                self.grid_rows     = new_rows;
                self.grid_cols     = new_cols;
                self.grid_rows_str = new_rows.to_string();
                self.grid_cols_str = new_cols.to_string();
                Command::none()
            }

            Message::ToggleAllRotations(v) => {
                self.search_all_rotations = v;
                Command::none()
            }

            Message::Search => {
                let seed = match self.seed.parse::<i64>() {
                    Ok(s)  => s,
                    Err(_) => { self.status = SearchStatus::Error("Seed must be a 64-bit integer".into()); return Command::none(); }
                };
                let start_x = match self.center_x.parse::<i32>() {
                    Ok(v)  => v,
                    Err(_) => { self.status = SearchStatus::Error("Invalid center X".into()); return Command::none(); }
                };
                let start_z = match self.center_z.parse::<i32>() {
                    Ok(v)  => v,
                    Err(_) => { self.status = SearchStatus::Error("Invalid center Z".into()); return Command::none(); }
                };
                let offset_x = self.grid_offset_x.parse::<i32>().unwrap_or(0);
                let offset_z = self.grid_offset_z.parse::<i32>().unwrap_or(0);
                let bt  = self.bedrock_type;
                let ys  = y_values(bt);
                let mut blocks_vec: Vec<Block> = Vec::new();
                for (y_idx, &y) in ys.iter().enumerate() {
                    for row in 0..self.grid_rows {
                        for col in 0..self.grid_cols {
                            let state = self.grid_cells[y_idx][row][col];
                            if state == CellState::Unknown { continue; }
                            blocks_vec.push(Block {
                                x: offset_x + col as i32,
                                y,
                                z: offset_z + row as i32,
                                should_be_bedrock: state == CellState::Bedrock,
                                probability:    compute_probability(y, bt),
                                prob_threshold: prob_to_threshold(compute_probability(y, bt)),
                            });
                        }
                    }
                }
                let all_rotations = self.search_all_rotations;
                let cancel = Arc::new(AtomicBool::new(false));
                self.cancel_flag = Some(cancel.clone());
                self.status = SearchStatus::Searching;
                Command::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            // Build the list of block-sets to search: either just
                            // the entered pattern, or all 4 rotations of it.
                            let rotations: Vec<Vec<Block>> = if all_rotations {
                                generate_rotations(blocks_vec)
                            } else {
                                vec![blocks_vec]
                            };

                            if rotations.len() == 1 {
                                // Single rotation: no parallelism overhead; dummy
                                // stop_early that is never set.
                                run_search(
                                    seed, start_x, start_z, bt,
                                    rotations.into_iter().next().unwrap(),
                                    cancel,
                                    Arc::new(AtomicBool::new(false)),
                                )
                            } else {
                                // Multi-rotation parallel search
                                //
                                // Each rotation runs as an independent rayon task.
                                // Rayon's work-stealing scheduler distributes
                                // available cores across rotations automatically,
                                // giving ~N/R threads to each (N cores, R rotations)
                                // without the overhead of building separate pools.
                                //
                                // `stop_rotations` is a shared flag: the first
                                // rotation to find a match sets it, causing every
                                // other rotation to bail out at the next chunk
                                // boundary (≤ CHUNK_SIZE iterations later).
                                // The user-cancel flag (`cancel`) is still
                                // propagated independently so the UI Cancel button
                                // continues to work normally.
                                let stop_rotations = Arc::new(AtomicBool::new(false));

                                let results: Vec<Result<Option<(i32, i32)>, String>> =
                                    rotations
                                        .into_par_iter()
                                        .map(|rot| {
                                            // Fast-path: skip if already done.
                                            if cancel.load(Ordering::Relaxed)
                                                || stop_rotations.load(Ordering::Relaxed)
                                            {
                                                return Ok(None);
                                            }
                                            let result = run_search(
                                                seed, start_x, start_z, bt,
                                                rot,
                                                cancel.clone(),
                                                stop_rotations.clone(),
                                            );
                                            // Signal remaining rotations to stop as
                                            // soon as this one finds a match.
                                            if matches!(result, Ok(Some(_))) {
                                                stop_rotations.store(true, Ordering::Relaxed);
                                            }
                                            result
                                        })
                                        .collect();

                                // Pick the best outcome in priority order:
                                //   1. Any successful match  (Ok(Some(_)))
                                //   2. User/early cancellation (Ok(None))
                                //   3. Last impossible-pattern error (Err(_))
                                let mut last: Result<Option<(i32, i32)>, String> = Ok(None);
                                for r in results {
                                    match r {
                                        Ok(Some(_)) => return r,           // found; done
                                        Ok(None)    => { last = Ok(None); } // cancelled
                                        Err(e)      => {
                                            // Only downgrade to an error if we
                                            // have no cancellation to report yet.
                                            if matches!(last, Ok(None)) {
                                                last = Err(e);
                                            }
                                        }
                                    }
                                }
                                last
                            }
                        })
                            .await
                            .unwrap_or_else(|e| Err(format!("Thread panic: {e}")))
                    },
                    Message::SearchDone,
                )
            }

            Message::Cancel => {
                if let Some(flag) = &self.cancel_flag { flag.store(true, Ordering::Relaxed); }
                self.status = SearchStatus::Cancelled;
                Command::none()
            }

            Message::ZoomIn => {
                self.ui_scale = (self.ui_scale + 0.1).min(2.0);
                Command::none()
            }
            Message::ZoomOut => {
                self.ui_scale = (self.ui_scale - 0.1).max(0.5);
                Command::none()
            }

            Message::SearchDone(result) => {
                self.cancel_flag = None;
                self.status = match result {
                    Ok(Some((x, z))) => SearchStatus::Found(x, z),
                    Ok(None)         => SearchStatus::Cancelled,
                    Err(e)           => SearchStatus::Error(e),
                };
                Command::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let is_searching = self.status == SearchStatus::Searching;
        let s = self.ui_scale;
        // Helper: scale a fixed pixel value by the zoom factor.
        let sc = |v: f32| v * s;

        let seed_row = row![
            text("World Seed").size(sc(16.0) as u16).width(Length::Fixed(sc(130.0))),
            text_input("e.g. 124352345", &self.seed)
                .on_input(Message::SeedChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fill)
                .padding(sc(8.0) as u16),
        ].spacing(sc(12.0) as u16).align_items(Alignment::Center);

        let center_row = row![
            text("Search Center").size(sc(16.0) as u16).width(Length::Fixed(sc(130.0))),
            text("X").size(sc(16.0) as u16),
            text_input("0", &self.center_x).on_input(Message::CenterXChanged).size(sc(16.0) as u16).width(Length::Fixed(sc(90.0))).padding(sc(8.0) as u16),
            text("Z").size(sc(16.0) as u16),
            text_input("0", &self.center_z).on_input(Message::CenterZChanged).size(sc(16.0) as u16).width(Length::Fixed(sc(90.0))).padding(sc(8.0) as u16),
        ].spacing(sc(10.0) as u16).align_items(Alignment::Center);

        let type_row = row![
            text("Bedrock Layer").size(sc(16.0) as u16).width(Length::Fixed(sc(130.0))),
            radio("Floor (Y -64 to -59)", BedrockType::Floor, Some(self.bedrock_type), Message::TypeChanged).text_size(sc(16.0) as u16),
            Space::with_width(Length::Fixed(sc(20.0))),
            radio("Roof  (Y 123 to 128)", BedrockType::Roof,  Some(self.bedrock_type), Message::TypeChanged).text_size(sc(16.0) as u16),
        ].spacing(sc(10.0) as u16).align_items(Alignment::Center);

        // Grid size + offset controls
        let grid_controls = row![
            text("Grid Size").size(sc(16.0) as u16).width(Length::Fixed(sc(80.0))),
            text("Cols").size(sc(16.0) as u16),
            text_input("8", &self.grid_cols_str)
                .on_input(Message::GridColsChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(46.0)))
                .padding(sc(7.0) as u16),
            text("Rows").size(sc(16.0) as u16),
            text_input("8", &self.grid_rows_str)
                .on_input(Message::GridRowsChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(46.0)))
                .padding(sc(7.0) as u16),
            Space::with_width(Length::Fixed(sc(20.0))),
            text("Offset").size(sc(16.0) as u16).width(Length::Fixed(sc(48.0))),
            text("X").size(sc(16.0) as u16),
            text_input("0", &self.grid_offset_x)
                .on_input(Message::GridOffsetXChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(58.0)))
                .padding(sc(7.0) as u16),
            text("Z").size(sc(16.0) as u16),
            text_input("0", &self.grid_offset_z)
                .on_input(Message::GridOffsetZChanged)
                .size(sc(16.0) as u16)
                .width(Length::Fixed(sc(58.0)))
                .padding(sc(7.0) as u16),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        // Y-layer tab strip
        // Tabs marked with * have at least one non-Unknown cell.
        let ys = y_values(self.bedrock_type);
        let mut y_row: Row<'_, Message> = Row::new()
            .spacing(sc(6.0) as u16)
            .align_items(Alignment::Center)
            .push(text("Y Layer").size(sc(16.0) as u16).width(Length::Fixed(sc(70.0))));
        for (i, &y) in ys.iter().enumerate() {
            let has_data = self.grid_cells[i].iter()
                .any(|r| r.iter().any(|&c| c != CellState::Unknown));
            let label = if has_data { format!("{}*", y) } else { y.to_string() };
            let btn = if i == self.grid_y_idx {
                // Active tab: no on_press so it is not re-clickable
                button(text(label).size(sc(13.0) as u16))
                    .style(theme::Button::Primary)
                    .padding([sc(5.0) as u16, sc(10.0) as u16])
            } else {
                button(text(label).size(sc(13.0) as u16))
                    .style(theme::Button::Secondary)
                    .on_press(Message::GridYChanged(i))
                    .padding([sc(5.0) as u16, sc(10.0) as u16])
            };
            y_row = y_row.push(btn);
        }

        // Cell grid
        let mut grid_col: Column<'_, Message> = Column::new().spacing(sc(2.0) as u16);
        for row_idx in 0..self.grid_rows {
            let mut grid_row: Row<'_, Message> = Row::new().spacing(sc(2.0) as u16);
            for col_idx in 0..self.grid_cols {
                let state = self.grid_cells[self.grid_y_idx][row_idx][col_idx];
                let (label, style) = match state {
                    CellState::Unknown    => ("?", theme::Button::Secondary),
                    CellState::NonBedrock => ("O", theme::Button::Primary),
                    CellState::Bedrock    => ("X", theme::Button::Destructive),
                };
                let cell = mouse_area(
                    button(
                            container(text(label).size(sc(15.0) as u16))
                                .width(Length::Fill)
                                .height(Length::Fill)
                                .center_x()
                                .center_y()
                        )
                        .on_press(Message::GridCellClicked(row_idx, col_idx))
                        .style(style)
                        .width(Length::Fixed(sc(30.0)))
                        .height(Length::Fixed(sc(30.0)))
                        .padding(0)
                ).on_right_press(Message::GridCellRightClicked(row_idx, col_idx));
                grid_row = grid_row.push(cell);
            }
            grid_col = grid_col.push(grid_row);
        }

        let rotate_row = row![
            text("Rotate grid:").size(sc(12.0) as u16).width(Length::Fixed(sc(80.0))),
            button(text("+90º (Clockwise)").size(sc(13.0) as u16))
                .on_press(Message::RotateCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
            button(text("−90º (Counter-clockwise)").size(sc(13.0) as u16))
                .on_press(Message::RotateCCW)
                .style(theme::Button::Secondary)
                .padding([sc(4.0) as u16, sc(10.0) as u16]),
        ].spacing(sc(8.0) as u16).align_items(Alignment::Center);

        let legend = row![
            text("Click to cycle:").size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(8.0))),
            text("? Unknown").size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(12.0))),
            text("O Non-bedrock").size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(sc(12.0))),
            text("X Bedrock").size(sc(12.0) as u16),
        ].align_items(Alignment::Center);

        let all_rotations_row = row![
            checkbox(
                "Search all 4 rotations (if north direction is unknown)",
                self.search_all_rotations,
            ).on_toggle(Message::ToggleAllRotations).text_size(sc(13.0) as u16),
        ].align_items(Alignment::Center);

        let search_btn = if is_searching {
            button(text("Searching...").size(sc(16.0) as u16)).padding([sc(10.0) as u16, sc(28.0) as u16])
        } else {
            button(text("Search").size(sc(16.0) as u16)).on_press(Message::Search).padding([sc(10.0) as u16, sc(28.0) as u16])
        };
        let cancel_btn = if is_searching {
            button(text("Cancel").size(sc(16.0) as u16)).on_press(Message::Cancel).padding([sc(10.0) as u16, sc(20.0) as u16])
        } else {
            button(text("Cancel").size(sc(16.0) as u16)).padding([sc(10.0) as u16, sc(20.0) as u16])
        };

        let status_msg = match &self.status {
            SearchStatus::Idle        => text("Ready when you are.").size(sc(16.0) as u16),
            SearchStatus::Searching   => text("Looking for that juicy leaked stash...").size(sc(16.0) as u16),
            SearchStatus::Cancelled   => text("Search cancelled. :(").size(sc(16.0) as u16),
            SearchStatus::Found(x, z) => text(format!("Found formation at X: {}   Z: {}", x, z)).size(sc(18.0) as u16),
            SearchStatus::Error(e)    => text(format!("Error: {}", e)).size(sc(16.0) as u16),
        };

        let zoom_row = row![
            text(format!("Zoom: {:.0}%", self.ui_scale * 100.0)).size(sc(12.0) as u16),
            Space::with_width(Length::Fixed(8.0)),
            button(text("−").size(sc(14.0) as u16))
                .on_press(Message::ZoomOut)
                .style(theme::Button::Secondary)
                .padding([sc(3.0) as u16, sc(10.0) as u16]),
            button(text("+").size(sc(14.0) as u16))
                .on_press(Message::ZoomIn)
                .style(theme::Button::Secondary)
                .padding([sc(3.0) as u16, sc(10.0) as u16]),
        ].spacing(4).align_items(Alignment::Center);

        let content = Column::new()
            .spacing(sc(2.0) as u16)
            .padding(sc(16.0) as u16)
            .max_width(sc(760.0))
            .push(
                row![
                    text("Bedrock Formation Finder").size(sc(26.0) as u16),
                    Space::with_width(Length::Fill),
                    zoom_row,
                ].align_items(Alignment::Center)
            )
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(seed_row)
            .push(center_row)
            .push(type_row)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(grid_controls)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(y_row)
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(grid_col)
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(rotate_row)
            .push(Space::with_height(Length::Fixed(sc(4.0))))
            .push(legend)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            .push(all_rotations_row)
            .push(Space::with_height(Length::Fixed(sc(6.0))))
            // .push(horizontal_rule(1))
            // .push(Space::with_height(Length::Fixed(12.0)))
            .push(
                container(row![search_btn, cancel_btn].spacing(16).align_items(Alignment::Center))
                    .width(Length::Fill)
                    .center_x()
            )
            .push(Space::with_height(Length::Fixed(sc(8.0))))
            .push(container(status_msg).width(Length::Fill).padding([sc(8.0) as u16, sc(14.0) as u16]));

        container(scrollable(content)).width(Length::Fill).height(Length::Fill).center_x().into()
    }
}
