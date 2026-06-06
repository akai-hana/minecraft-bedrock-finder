/// bedrockformation
/// Rust port of the Minecraft Bedrock Formation Finder.
///
/// Usage: bedrockformation <seed> <x:z> <floor|roof> [x,y,z:bedrock ...]
/// Example: bedrockformation 124352345 0:0 floor 0,-63,0:1 1,-62,0:1 0,-63,1:0

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use iced::{
    Application, Command, Element, Length, Settings, Theme, Alignment,
    executor, theme,
    widget::{button, checkbox, container, horizontal_rule, mouse_area, radio, row, scrollable, text, text_input, Column, Row, Space},
    window,
};

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
// - guard_zero  Omitted (same reasoning as AVX2).
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

fn main() -> iced::Result {
    App::run(Settings {
        window: window::Settings {
            size: iced::Size::new(820.0, 720.0),
            min_size: Some(iced::Size::new(680.0, 560.0)),
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
        probability: b.probability,
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
    let mut rotations: Vec<Vec<Block>> = Vec::with_capacity(4);
    let mut seen: Vec<Vec<(i32, i32, i32, bool)>> = Vec::with_capacity(4);
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

    let mut x   = start_x;
    let mut z   = start_z;
    let mut dir = Dir::Right;
    let mut steps_to_take: i32 = 1;
    let mut steps_taken: i32 = 0;
    let mut sides_until_incremental: i32 = 0;

    let mut chunk = vec![(0i32, 0i32); CHUNK_SIZE];

    loop {
        if cancel.load(Ordering::Relaxed) { return Ok(None); }

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
    // Grid dimensions (1–16 each)
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
                                probability: compute_probability(y, bt),
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
                            // Try each rotation in turn; return the first hit (or
                            // the last error/cancellation if none match).
                            let mut last = Ok(None);
                            for rot in rotations {
                                last = run_search(seed, start_x, start_z, bt, rot, cancel.clone());
                                match &last {
                                    Ok(Some(_)) => return last, // found
                                    Ok(None)    => return last, // cancelled
                                    Err(_)      => {}           // impossible pattern, try next rotation
                                }
                            }
                            last
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

        let seed_row = row![
            text("World Seed").width(Length::Fixed(130.0)),
            text_input("e.g. 124352345", &self.seed)
                .on_input(Message::SeedChanged)
                .width(Length::Fill)
                .padding(8),
        ].spacing(12).align_items(Alignment::Center);

        let center_row = row![
            text("Search Center").width(Length::Fixed(130.0)),
            text("X"),
            text_input("0", &self.center_x).on_input(Message::CenterXChanged).width(Length::Fixed(90.0)).padding(8),
            text("Z"),
            text_input("0", &self.center_z).on_input(Message::CenterZChanged).width(Length::Fixed(90.0)).padding(8),
        ].spacing(10).align_items(Alignment::Center);

        let type_row = row![
            text("Bedrock Layer").width(Length::Fixed(130.0)),
            radio("Floor (Y -64 to -59)", BedrockType::Floor, Some(self.bedrock_type), Message::TypeChanged),
            Space::with_width(Length::Fixed(20.0)),
            radio("Roof  (Y 123 to 128)", BedrockType::Roof,  Some(self.bedrock_type), Message::TypeChanged),
        ].spacing(10).align_items(Alignment::Center);

        // Grid size + offset controls
        let grid_controls = row![
            text("Grid Size").width(Length::Fixed(80.0)),
            text("Cols"),
            text_input("8", &self.grid_cols_str)
                .on_input(Message::GridColsChanged)
                .width(Length::Fixed(46.0))
                .padding(7),
            text("Rows"),
            text_input("8", &self.grid_rows_str)
                .on_input(Message::GridRowsChanged)
                .width(Length::Fixed(46.0))
                .padding(7),
            Space::with_width(Length::Fixed(20.0)),
            text("Offset").width(Length::Fixed(48.0)),
            text("X"),
            text_input("0", &self.grid_offset_x)
                .on_input(Message::GridOffsetXChanged)
                .width(Length::Fixed(58.0))
                .padding(7),
            text("Z"),
            text_input("0", &self.grid_offset_z)
                .on_input(Message::GridOffsetZChanged)
                .width(Length::Fixed(58.0))
                .padding(7),
        ].spacing(8).align_items(Alignment::Center);

        // Y-layer tab strip
        // Tabs marked with * have at least one non-Unknown cell.
        let ys = y_values(self.bedrock_type);
        let mut y_row: Row<'_, Message> = Row::new()
            .spacing(6)
            .align_items(Alignment::Center)
            .push(text("Y Layer").width(Length::Fixed(70.0)));
        for (i, &y) in ys.iter().enumerate() {
            let has_data = self.grid_cells[i].iter()
                .any(|r| r.iter().any(|&c| c != CellState::Unknown));
            let label = if has_data { format!("{}*", y) } else { y.to_string() };
            let btn = if i == self.grid_y_idx {
                // Active tab: no on_press so it is not re-clickable
                button(text(label).size(13))
                    .style(theme::Button::Primary)
                    .padding([5, 10])
            } else {
                button(text(label).size(13))
                    .style(theme::Button::Secondary)
                    .on_press(Message::GridYChanged(i))
                    .padding([5, 10])
            };
            y_row = y_row.push(btn);
        }

        // Cell grid
        let mut grid_col: Column<'_, Message> = Column::new().spacing(2);
        for row_idx in 0..self.grid_rows {
            let mut grid_row: Row<'_, Message> = Row::new().spacing(2);
            for col_idx in 0..self.grid_cols {
                let state = self.grid_cells[self.grid_y_idx][row_idx][col_idx];
                let (label, style) = match state {
                    CellState::Unknown    => ("?", theme::Button::Secondary),
                    CellState::NonBedrock => ("O", theme::Button::Primary),
                    CellState::Bedrock    => ("X", theme::Button::Destructive),
                };
                let cell = mouse_area(
                    button(
                            container(text(label).size(15))
                                .width(Length::Fill)
                                .height(Length::Fill)
                                .center_x()
                                .center_y()
                        )
                        .on_press(Message::GridCellClicked(row_idx, col_idx))
                        .style(style)
                        .width(Length::Fixed(30.0))
                        .height(Length::Fixed(30.0))
                        .padding(0)
                ).on_right_press(Message::GridCellRightClicked(row_idx, col_idx));
                grid_row = grid_row.push(cell);
            }
            grid_col = grid_col.push(grid_row);
        }

        let rotate_row = row![
            text("Rotate grid:").size(12).width(Length::Fixed(80.0)),
            button(text("+90º (Clockwise)").size(13))
                .on_press(Message::RotateCW)
                .style(theme::Button::Secondary)
                .padding([4, 10]),
            button(text("−90º (Counter-clockwise)").size(13))
                .on_press(Message::RotateCCW)
                .style(theme::Button::Secondary)
                .padding([4, 10]),
        ].spacing(8).align_items(Alignment::Center);

        let legend = row![
            text("Click to cycle:").size(12),
            Space::with_width(Length::Fixed(8.0)),
            text("? Unknown").size(12),
            Space::with_width(Length::Fixed(12.0)),
            text("O Non-bedrock").size(12),
            Space::with_width(Length::Fixed(12.0)),
            text("X Bedrock").size(12),
        ].align_items(Alignment::Center);

        let all_rotations_row = row![
            checkbox(
                "Search all 4 rotations (if north direction is unknown)",
                self.search_all_rotations,
            ).on_toggle(Message::ToggleAllRotations).text_size(13),
        ].align_items(Alignment::Center);

        let search_btn = if is_searching {
            button("Searching...").padding([10, 28])
        } else {
            button("Search").on_press(Message::Search).padding([10, 28])
        };
        let cancel_btn = if is_searching {
            button("Cancel").on_press(Message::Cancel).padding([10, 20])
        } else {
            button("Cancel").padding([10, 20])
        };

        let status_msg = match &self.status {
            SearchStatus::Idle        => text("Ready when you are."),
            SearchStatus::Searching   => text("Looking for that juicy leaked stash..."),
            SearchStatus::Cancelled   => text("Search cancelled. :("),
            SearchStatus::Found(x, z) => text(format!("Found formation at X: {}   Z: {}", x, z)).size(18),
            SearchStatus::Error(e)    => text(format!("Error: {}", e)),
        };

        let content = Column::new()
            .spacing(2)
            .padding(28)
            .max_width(760)
            .push(text("Bedrock Formation Finder").size(26))
            .push(Space::with_height(Length::Fixed(4.0)))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(8.0)))
            .push(seed_row)
            .push(center_row)
            .push(type_row)
            .push(Space::with_height(Length::Fixed(8.0)))
            .push(horizontal_rule(1))
            .push(Space::with_height(Length::Fixed(8.0)))
            .push(grid_controls)
            .push(Space::with_height(Length::Fixed(8.0)))
            .push(y_row)
            .push(Space::with_height(Length::Fixed(4.0)))
            .push(scrollable(grid_col))
            .push(Space::with_height(Length::Fixed(4.0)))
            .push(rotate_row)
            .push(Space::with_height(Length::Fixed(4.0)))
            .push(legend)
            .push(Space::with_height(Length::Fixed(8.0)))
            .push(all_rotations_row)
            .push(Space::with_height(Length::Fixed(8.0)))
            // .push(horizontal_rule(1))
            // .push(Space::with_height(Length::Fixed(12.0)))
            .push(
                container(row![search_btn, cancel_btn].spacing(16).align_items(Alignment::Center))
                    .width(Length::Fill)
                    .center_x()
            )
            .push(Space::with_height(Length::Fixed(12.0)))
            .push(container(status_msg).width(Length::Fill).padding([10, 14]));

        container(content).width(Length::Fill).height(Length::Fill).center_x().into()
    }
}
