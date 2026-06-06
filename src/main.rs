/// bedrockformation
/// Rust port of the Minecraft Bedrock Formation Finder.
///
/// Usage: bedrockformation <seed> <x:z> <floor|roof> [x,y,z:bedrock ...]
/// Example: bedrockformation 124352345 0:0 floor 0,-63,0:1 1,-62,0:1 0,-63,1:0

use std::env;

// Constants 

const FLOAT_MULT: f32 = 5.960_464_5e-8_f32; // 2^-24

const FALLBACK_LO: u64 = (-7_046_029_254_386_353_131_i64) as u64;
const FALLBACK_HI: u64 = 7_640_891_576_956_012_809_u64;

// MUST be a multiple of 8: AVX-512 processes groups of 8, AVX2 groups of 4.
// (262_144 satisfies both; it is 2^18.)
//
// Profiling note: at 262_144 * 8 B = 2 MB, the chunk buffer overflows L2 on most
// CPUs (typically 256 KB-1 MB).  Trying 65_536 (512 KB) or 32_768 (256 KB) may
// improve throughput on cache-limited hardware while keeping rayon's find_first
// effective.  The spiral-fill loop is cheap, so smaller chunks have little overhead.
const CHUNK_SIZE: usize = 262_144;
const _: () = assert!(CHUNK_SIZE % 8 == 0, "CHUNK_SIZE must be a multiple of 8 (AVX-512 requirement)");

// Bedrock type 

#[derive(Clone, Copy, PartialEq, Eq)]
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

// Issue 6: mark #[inline(always)] so is_bedrock calls this instead of reimplementing it.
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

    // Issue 6: use guard_zero instead of reimplementing inline.
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

struct Block {
    x: i32, y: i32, z: i32,
    should_be_bedrock: bool,
    probability: f64,
}

// Formation check (mirrors Main.checkFormation) 

#[inline(always)]
fn check_formation(ox: i32, oz: i32, dlo: i64, dhi: i64, blocks: &[Block]) -> bool {
    blocks.iter().all(|b| {
        b.should_be_bedrock == is_bedrock(dlo, dhi, ox + b.x, b.y, oz + b.z, b.probability)
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
    use super::Block;

    // Helpers 

    /// Low 64 bits of 4-lane 64-bit integer multiply.
    ///
    /// AVX2 has no `mullo_epi64`; we use the identity:
    ///   a*b (low 64) = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32
    ///
    /// `_mm256_mul_epu32` multiplies the low 32 bits of each 64-bit lane
    /// (unsigned), producing an unsigned 64-bit result per lane.
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
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rotl17_epi64(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_slli_epi64(x, 17), _mm256_srli_epi64(x, 47))
    }

    /// Pack the low 32 bits of each 64-bit lane of a 256-bit register into a
    /// contiguous 128-bit register of four 32-bit integers.
    ///
    /// Input:  [A_hi:A_lo | B_hi:B_lo | C_hi:C_lo | D_hi:D_lo]  (32-bit halves)
    /// Output: [A_lo | B_lo | C_lo | D_lo]
    ///
    /// A single `vpermps` (permutevar8x32) picks the 32-bit elements at indices
    /// 0, 2, 4, 6 (the lo32 of each 64-bit lane) directly into the low 128 bits.
    /// One instruction vs. the previous extract + shuffle + unpacklo sequence.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn pack_lo32(v: __m256i) -> __m128i {
        // Element indices within the 256-bit register (32-bit granularity):
        //   0 = A_lo, 2 = B_lo, 4 = C_lo, 6 = D_lo
        // Upper four slots are don't-cares; we only keep the low 128 bits.
        let idx = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(v, idx))
    }

    // Core SIMD kernel 

    /// Compute `math_hash(x[i], y, z[i])` for i in 0..4 simultaneously.
    ///
    /// `x_vec` / `z_vec` are `__m128i` holding 4 * i32 (lane 0 = lowest address).
    /// `y` is the same for all lanes and is broadcast to a 64-bit constant.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn math_hash_x4(x_vec: __m128i, y: i32, z_vec: __m128i) -> __m256i {
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
        {
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
    /// Assumes `probability == (0, 1)`: trivially-guaranteed blocks are
    /// filtered out before the search starts and never reach this path.
    ///
    /// The float comparison is performed in f64 (via `_mm256_cvtps_pd`) to
    /// match the scalar `(f as f64) < probability` exactly.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn is_bedrock_x4(
        dlo:               i64,
        dhi:               i64,
        x_vec:             __m128i,  // 4 * i32: absolute X coords
        y:                 i32,
        z_vec:             __m128i,  // 4 * i32: absolute Z coords
        probability:       f64,
        should_be_bedrock: bool,
    ) -> u8 {
        // s0 = (math_hash(x, y, z) ^ dlo) as u64  (per lane)
        // s1 = dhi as u64  (same for every lane, broadcast)
        let hash   = math_hash_x4(x_vec, y, z_vec);
        let s0 = _mm256_xor_si256(hash, _mm256_set1_epi64x(dlo));
        let s1     = _mm256_set1_epi64x(dhi);

        // guard_zero removed from SIMD hot path: dhi is derived from MD5 ^ xoroshiro
        // output, making zero cryptographically impossible in practice.  The scalar
        // is_bedrock retains the guard for correctness on theoretical edge cases.
        // Removing the or/cmpeq/blendv trio saves 3 instructions per block per group.
        debug_assert_ne!(dhi, 0, "deriver hi seed must be non-zero");

        // xoroshiro128++ single step: result = (s0 + s1).rotate_left(17) + s0
        let sum    = _mm256_add_epi64(s0, s1);
        let result = _mm256_add_epi64(rotl17_epi64(sum), s0);

        // nextFloat() -> top 24 bits of result (bits 63:40) * 2^-24.
        // Convert to f64 for exact comparison against the f64 probability.
        //   result >> 40          : 24-bit value in low bits of each 64-bit lane
        //   pack_lo32             : 4 * u32 packed into __m128i
        //   _mm_cvtepi32_ps       : 4 * f32  (values in [0, 2^24); all non-negative
        //                           so signed conversion equals unsigned)
        //   _mm256_cvtps_pd       : 4 * f64  (exact; f32 has 24-bit mantissa)
        //   * 2^-24               : floats in [0, 1)
        let top24  = _mm256_srli_epi64(result, 40);
        let i32v   = pack_lo32(top24);
        let f32v   = _mm_cvtepi32_ps(i32v);
        let f64v   = _mm256_cvtps_pd(f32v);
        let floats = _mm256_mul_pd(f64v, _mm256_set1_pd(5.960_464_477_539_063e-8_f64)); // 2^-24

        // Compare floats < probability; _CMP_LT_OS = 1 (less-than, ordered, signaling).
        // movemask extracts bit i from the sign bit of lane i -> 4-bit integer.
        let prob_v       = _mm256_set1_pd(probability);
        let cmp          = _mm256_cmp_pd::<1>(floats, prob_v); // 1 == _CMP_LT_OS
        let bedrock_mask = _mm256_movemask_pd(cmp) as u8;      // bit i = 1 <-> position i is bedrock

        // Reconcile with what the block expects.
        if should_be_bedrock { bedrock_mask } else { !bedrock_mask & 0x0F }
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
        positions: &[(i32, i32)], // exactly 4 entries
        dlo:    i64,
        dhi:    i64,
        blocks: &[Block],
    ) -> u8 {
        debug_assert_eq!(positions.len(), 4);

        // Build base position vectors.  _mm_set_epi32(e3,e2,e1,e0) puts e0 in lane 0.
        let ox_v = _mm_set_epi32(
            positions[3].0, positions[2].0, positions[1].0, positions[0].0,
        );
        let oz_v = _mm_set_epi32(
            positions[3].1, positions[2].1, positions[1].1, positions[0].1,
        );

        let mut active: u8 = 0x0F; // bits 0-3 all set = all lanes in play

        for b in blocks {
            // Absolute coordinates for this block's offset
            let x_v = _mm_add_epi32(ox_v, _mm_set1_epi32(b.x));
            let z_v = _mm_add_epi32(oz_v, _mm_set1_epi32(b.z));

            let passed = is_bedrock_x4(dlo, dhi, x_v, b.y, z_v, b.probability, b.should_be_bedrock);
            active &= passed;
            if active == 0 { return 0; } // no lane can match; exit immediately
        }
        active
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
// - guard_zero  Omitted (same reasoning as AVX2; see issue-3 comment there).
//
// On Ice Lake / Zen 4 and newer this is a theoretical 2* throughput improvement
// over the AVX2 path for the inner kernel.
//
// Detection: avx512f + avx512dq.  CHUNK_SIZE (262_144) is already a multiple of 8.

#[cfg(target_arch = "x86_64")]
mod simd_avx512 {
    use core::arch::x86_64::*;
    use super::Block;

    /// Rotate each 64-bit lane left by 17 bits (AVX-512F).
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn rotl17_epi64(x: __m512i) -> __m512i {
        _mm512_or_si512(_mm512_slli_epi64(x, 17), _mm512_srli_epi64(x, 47))
    }

    /// Compute `math_hash(x[i], y, z[i])` for i in 0..8 simultaneously.
    ///
    /// `x_vec` / `z_vec` are `__m256i` holding 8 * i32.
    /// `y` is broadcast as a 64-bit constant across all 8 lanes.
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

    /// Returns an 8-bit mask: bit `i` is set if position `i` passes the
    /// `is_bedrock` test for the given block parameters.
    ///
    /// Assumes `probability in (0, 1)`: guaranteed blocks are filtered before
    /// the search begins and never reach this path.
    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    #[inline]
    unsafe fn is_bedrock_x8(
        dlo:               i64,
        dhi:               i64,
        x_vec:             __m256i,  // 8 * i32: absolute X coords
        y:                 i32,
        z_vec:             __m256i,  // 8 * i32: absolute Z coords
        probability:       f64,
        should_be_bedrock: bool,
    ) -> u8 {
        let hash = math_hash_x8(x_vec, y, z_vec);
        let s0   = _mm512_xor_si512(hash, _mm512_set1_epi64(dlo));
        let s1   = _mm512_set1_epi64(dhi);

        // guard_zero omitted: dhi is derived from MD5 ^ xoroshiro output;
        // it is never zero in practice.
        debug_assert_ne!(dhi, 0, "deriver hi seed must be non-zero");

        // xoroshiro128++ single step: result = (s0 + s1).rotate_left(17) + s0
        let sum    = _mm512_add_epi64(s0, s1);
        let result = _mm512_add_epi64(rotl17_epi64(sum), s0);

        // nextFloat() -> top 24 bits (result >> 40), converted to f64.
        //   _mm512_cvtepi64_epi32: 8 * i64 -> 8 * i32 (truncates; top24 fits in 24 bits)
        //   _mm256_cvtepi32_ps   : 8 * f32
        //   _mm512_cvtps_pd      : 8 * f64  (exact; f32 has 24-bit mantissa)
        let top24  = _mm512_srli_epi64(result, 40);
        let i32v   = _mm512_cvtepi64_epi32(top24);
        let f32v   = _mm256_cvtepi32_ps(i32v);
        let f64v   = _mm512_cvtps_pd(f32v);
        let floats = _mm512_mul_pd(f64v, _mm512_set1_pd(5.960_464_477_539_063e-8_f64)); // 2^-24

        // _mm512_cmp_pd_mask returns a u8 k-register directly (1 = _CMP_LT_OS).
        // No movemask step needed.
        let prob_v: __m512d  = _mm512_set1_pd(probability);
        let bedrock_mask: u8 = _mm512_cmp_pd_mask::<1>(floats, prob_v);

        if should_be_bedrock { bedrock_mask } else { !bedrock_mask }
    }

    /// Returns an 8-bit mask of the positions (within a group of 8) that match
    /// the entire formation.  Exits early as soon as no lanes remain active.
    ///
    /// # Safety
    /// Requires AVX-512F + AVX-512DQ.  Caller must have verified feature support.
    #[target_feature(enable = "avx512f,avx512dq,avx2")]
    pub unsafe fn check_formation_x8(
        positions: &[(i32, i32)], // exactly 8 entries
        dlo:    i64,
        dhi:    i64,
        blocks: &[Block],
    ) -> u8 {
        debug_assert_eq!(positions.len(), 8);

        // _mm256_set_epi32(e7,e6,...,e0) places e0 in lane 0.
        let ox_v = _mm256_set_epi32(
            positions[7].0, positions[6].0, positions[5].0, positions[4].0,
            positions[3].0, positions[2].0, positions[1].0, positions[0].0,
        );
        let oz_v = _mm256_set_epi32(
            positions[7].1, positions[6].1, positions[5].1, positions[4].1,
            positions[3].1, positions[2].1, positions[1].1, positions[0].1,
        );

        let mut active: u8 = 0xFF; // bits 0-7 all set = all lanes in play

        for b in blocks {
            let x_v = _mm256_add_epi32(ox_v, _mm256_set1_epi32(b.x));
            let z_v = _mm256_add_epi32(oz_v, _mm256_set1_epi32(b.z));

            let passed = is_bedrock_x8(dlo, dhi, x_v, b.y, z_v, b.probability, b.should_be_bedrock);
            active &= passed;
            if active == 0 { return 0; }
        }
        active
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

fn search_chunk(chunk: &[(i32, i32)], dlo: i64, dhi: i64, blocks: &[Block], simd: SimdLevel) -> Option<usize> {
    // All paths iterate sequentially over groups to guarantee spiral order.
    // SIMD kernels provide fast per-group candidate screening; the scalar
    // check_formation confirms the exact position within a matching group.

    #[cfg(target_arch = "x86_64")]
    if simd == SimdLevel::Avx512 {
        let n_groups = chunk.len() / 8;
        for g in 0..n_groups {
            let start = g * 8;
            // SAFETY: AVX-512F+DQ verified by detect_simd; slice is exactly 8 elements.
            let mask = unsafe {
                simd_avx512::check_formation_x8(&chunk[start..start + 8], dlo, dhi, blocks)
            };
            if mask != 0 {
                for j in 0..8 {
                    let (cx, cz) = chunk[start + j];
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
        let n_groups = chunk.len() / 4;
        for g in 0..n_groups {
            let start = g * 4;
            // SAFETY: AVX2 verified by detect_simd; slice is exactly 4 elements.
            let mask = unsafe {
                simd_avx2::check_formation_x4(&chunk[start..start + 4], dlo, dhi, blocks)
            };
            if mask != 0 {
                for j in 0..4 {
                    let (cx, cz) = chunk[start + j];
                    if check_formation(cx, cz, dlo, dhi, blocks) {
                        return Some(start + j);
                    }
                }
            }
        }
        return None;
    }

    // Scalar fallback: sequential to guarantee spiral order.
    (0..chunk.len()).find(|&i| {
        let (cx, cz) = chunk[i];
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

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: bedrockformation <seed> <x:z> <floor|roof> [x,y,z:bedrock ...]");
        eprintln!("Example: bedrockformation 124352345 0:0 floor 0,-63,0:1 1,-62,0:1 0,-63,1:0");
        std::process::exit(1);
    }

    let seed: i64 = args[1].parse().expect("seed must be a long integer");
    let xz: Vec<&str> = args[2].split(':').collect();
    let start_x: i32  = xz[0].parse().expect("invalid center x");
    let start_z: i32  = xz[1].parse().expect("invalid center z");
    let bt = if args[3] == "roof" { BedrockType::Roof } else { BedrockType::Floor };

    let mut blocks: Vec<Block> = args[4..].iter().map(|arg| {
        let halves: Vec<&str> = arg.splitn(2, ':').collect();
        let xyz: Vec<i32> = halves[0].split(',').map(|s| s.parse().unwrap()).collect();
        let should_be_bedrock = halves[1].parse::<i32>().unwrap() == 1;
        let probability = compute_probability(xyz[1], bt);
        Block { x: xyz[0], y: xyz[1], z: xyz[2], should_be_bedrock, probability }
    }).collect();

    if blocks.is_empty() { return; }

    // Detect impossible constraints up front so we fail fast instead of looping forever
    for b in &blocks {
        let always_bedrock = b.probability >= 1.0;
        let never_bedrock  = b.probability <= 0.0;
        if always_bedrock && !b.should_be_bedrock {
            eprintln!(
                "Error: block ({},{},{}) is always bedrock but declared as air. \
                 No solution exists.",
                 b.x, b.y, b.z
            );
            std::process::exit(1);
        }
        if never_bedrock && b.should_be_bedrock {
            eprintln!(
                "Error: block ({},{},{}) is never bedrock but declared as bedrock. \
                 No solution exists.",
                 b.x, b.y, b.z
            );
            std::process::exit(1);
        }
    }

    // Sort by descending mismatch probability (most-likely-to-reject first) so
    // check_formation short-circuits as early as possible.
    // sort_by_cached_key computes each key exactly once (no recompute per comparison,
    // no intermediate Vec). Casting to bits is safe: all keys are finite non-negative
    // f64, so IEEE bit order matches numeric order; Reverse makes it descending.
    blocks.sort_by_cached_key(|b| {
        let key = if b.should_be_bedrock { 1.0 - clamp01(b.probability) } else { clamp01(b.probability) };
        std::cmp::Reverse(key.to_bits())
    });

    // Issue 3: Strip trivially-informationless blocks after sorting.
    // Blocks where the outcome is guaranteed (always-bedrock declared as bedrock,
    // or never-bedrock declared as air) always pass their check and contribute
    // nothing to candidate rejection. Removing them avoids iterating over them
    // on every single candidate.
    let blocks: Vec<Block> = blocks.into_iter().filter(|b| {
        let p = clamp01(b.probability);
        if b.should_be_bedrock { p < 1.0 } else { p > 0.0 }
    }).collect();

    if blocks.is_empty() {
        // Every block was trivially guaranteed, so every coordinate matches.
        println!("Found Bedrock Formation at X:{} Z:{}", start_x, start_z);
        return;
    }

    for b in &blocks {
        println!("BedrockBlock{{x={}, y={}, z={}, shouldBeBedrock={}, p={:.3}}}",
            b.x, b.y, b.z, b.should_be_bedrock, clamp01(b.probability));
    }

    let (dlo, dhi) = compute_deriver_seeds(seed, bt);

    // Detect the widest available SIMD level once here rather than on every
    // call to search_chunk.  std's is_x86_feature_detected! caches the result
    // in an atomic, but it still costs a function call + memory load + branch
    // that the compiler can't eliminate.  Hoisting gives a trivially-predictable
    // branch (always the same value) and makes the dispatch point explicit.
    let simd = detect_simd();

    // Spiral search
    // Fill a chunk buffer, then TODO: search it in parallel with rayon.
    //
    // rayon::find_first guarantees the earliest (spiral-order) match, automatically
    // cancels other workers once a match is found, and never wastes threads on
    // empty ranges. No manual pool, no Vec copies, no mpsc plumbing needed.
    //
    // AoS layout: one Vec<(i32,i32)> keeps each position's x and z adjacent
    // so rayon workers traverse a single contiguous allocation.
    //
    // search_chunk dispatches to the AVX2 SIMD path (groups of 4) when available,
    // falling back to the scalar path otherwise.
    let mut x   = start_x;
    let mut z   = start_z;
    let mut dir = Dir::Right;
    let mut steps_to_take: i32 = 1;
    let mut steps_taken: i32 = 0;
    let mut sides_until_incremental: i32 = 0;

    let mut chunk = vec![(0i32, 0i32); CHUNK_SIZE];

    loop {
        for slot in chunk.iter_mut() {
            *slot = (x, z);
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

        if let Some(idx) = search_chunk(&chunk, dlo, dhi, &blocks, simd) {
            let (cx, cz) = chunk[idx];
            println!("Found Bedrock Formation at X:{} Z:{}", cx, cz);
            break;
        }
    }
}
