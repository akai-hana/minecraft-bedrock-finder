// search.wgsl - bedrock formation GPU kernel
//
// Each thread independently computes its spiral position from its global
// thread index, so no candidate position buffer needs to be uploaded.
// 64-bit arithmetic is emulated with U64 (lo, hi) pairs throughout.

// U64 helpers

struct U64 { lo: u32, hi: u32 }

fn u64_xor(a: U64, b: U64) -> U64 { return U64(a.lo ^ b.lo, a.hi ^ b.hi); }

fn u64_add(a: U64, b: U64) -> U64 {
    let lo    = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

fn u64_rotl17(a: U64) -> U64 {
    return U64((a.lo << 17u) | (a.hi >> 15u),
               (a.hi << 17u) | (a.lo >> 15u));
}

// 32x32 -> 64-bit widening multiply (no overflow in any partial product).
fn mul32x32(a: u32, b: u32) -> U64 {
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p10 = a1 * b0;
    let p01 = a0 * b1; let p11 = a1 * b1;
    let r0  = p00 & 0xFFFFu;
    var acc = (p00 >> 16u) + (p10 & 0xFFFFu) + (p01 & 0xFFFFu);
    let r1  = acc & 0xFFFFu;
    acc     = (acc >> 16u) + (p10 >> 16u) + (p01 >> 16u) + (p11 & 0xFFFFu);
    let r2  = acc & 0xFFFFu;
    let r3  = (acc >> 16u) + (p11 >> 16u);
    return U64(r0 | (r1 << 16u), r2 | (r3 << 16u));
}

// Low 64 bits of a x b.
fn u64_mul(a: U64, b: U64) -> U64 {
    let ll    = mul32x32(a.lo, b.lo);
    let cross = a.lo * b.hi + a.hi * b.lo;   // only low 32 bits matter
    return U64(ll.lo, ll.hi + cross);
}

// a - b, assuming a >= b (no underflow check - callers guarantee this).
fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    return U64(a.lo - b.lo, a.hi - b.hi - borrow);
}

// a < b (unsigned).
fn u64_lt(a: U64, b: U64) -> bool {
    if (a.hi != b.hi) { return a.hi < b.hi; }
    return a.lo < b.lo;
}

// 2 * a.
fn u64_double(a: U64) -> U64 { return u64_add(a, a); }

// Zero-extend a u32 to U64.
fn u32_to_u64(v: u32) -> U64 { return U64(v, 0u); }

// Arithmetic right-shift by 16 (sign extension from bit 63).
fn i64_sra16(a: U64) -> U64 {
    let sf = select(0u, 0xFFFFu, (a.hi & 0x80000000u) != 0u);
    return U64((a.lo >> 16u) | (a.hi << 16u), (a.hi >> 16u) | (sf << 16u));
}

// Sign-extend i32 to U64.
fn i32_to_u64(v: i32) -> U64 {
    return U64(u32(v), select(0u, 0xFFFFFFFFu, v < 0));
}

// Spiral position (closed form)
//
// Mirrors the Rust `coords_from_lj(l, j)` exactly:
//   Shell l, intra-shell offset j = k - (4l^2-4l+1)
//   j < 2l-1              -> Leg 0 Up   (+z): (l,  -(l-1)+j)
//   2l-1 <= j < 4l-1       -> Leg 1 Left (-x): (l-o,  l)       o = j-(2l-1)
//   4l-1 <= j < 6l-1       -> Leg 2 Down (-z): (-l,   l-o)     o = j-(4l-1)
//   j >= 6l-1              -> Leg 3 Right(+x): (-l+o, -l)      o = j-(6l-1)
//
// `k` is a full U64 (lo/hi u32 pair, mirroring the CPU's i64). All
// shell-boundary arithmetic (4l^2 +/- 4l, comparisons against k) is done with
// exact U64 add/sub/compare - there is no i32(k) cast and therefore no
// ~2.1-billion ceiling. The only inexact step is the *initial* shell
// estimate via f32::sqrt, which the loop below corrects to the exact
// value, so the final result is exact for any k.
//
// `l` stays a u32 and the final `ox`/`oz` offsets are cast to i32, giving
// the same bound as the CPU path (l <= i32::MAX, unreachable in practice
// since `4*l*l` overflows i64 long before that).

fn spiral_pos(k: U64, sx: i32, sz: i32) -> vec2<i32> {
    if (k.lo == 0u && k.hi == 0u) { return vec2<i32>(sx, sz); }

    // Initial shell estimate. f32 has 24 bits of mantissa, so for very
    // large k this can be off by more than 1 - the loop below corrects it
    // to the exact value using U64 arithmetic.
    let kf = f32(k.hi) * 4294967296.0 + f32(k.lo);
    var l: u32 = max(1u, u32((1.0 + sqrt(kf)) * 0.5));

    // Correct l so that 4l^2-4l+1 <= k <= 4l^2+4l, placing k in shell l.
    // f32 imprecision is tiny relative to l even near u64::MAX, so this
    // converges in at most a few iterations; 64 is a generous safety bound.
    var four_l: U64;
    var lower:  U64; // 4l^2 - 4l + 1
    for (var iter = 0u; iter < 64u; iter = iter + 1u) {
        let l64     = u32_to_u64(l);
        let lsq     = u64_mul(l64, l64);
        let four_lsq = u64_double(u64_double(lsq));   // 4l^2
        four_l       = u64_double(u64_double(l64));   // 4l
        let upper    = u64_add(four_lsq, four_l);                       // 4l^2 + 4l
        lower        = u64_add(u64_sub(four_lsq, four_l), U64(1u, 0u)); // 4l^2 - 4l + 1

        if (u64_lt(upper, k)) {
            l += 1u;
        } else if (u64_lt(k, lower)) {
            l -= 1u;
        } else {
            break;
        }
    }

    // j = k - (4l^2 - 4l + 1). Fits in u32: j <= 8l-2 and l <= i32::MAX
    // implies j < u32::MAX.
    let j = u64_sub(k, lower).lo;

    let b0 = u64_sub(u64_double(u32_to_u64(l)), U64(1u, 0u)).lo;                  // 2l - 1
    let b1 = u64_sub(four_l, U64(1u, 0u)).lo;                                     // 4l - 1
    let b2 = u64_sub(u64_add(four_l, u64_double(u32_to_u64(l))), U64(1u, 0u)).lo; // 6l - 1

    let li = i32(l);
    var ox: i32; var oz: i32;
    if (j < b0) {
        ox = li;             oz = -(li - 1) + i32(j);
    } else if (j < b1) {
        let o = i32(j - b0); ox = li - o;   oz = li;
    } else if (j < b2) {
        let o = i32(j - b1); ox = -li;      oz = li - o;
    } else {
        let o = i32(j - b2); ox = -li + o;  oz = -li;
    }
    return vec2<i32>(sx + ox, sz + oz);
}

// math_hash
//
// The CPU computes z_term as:
//   (oz as i64).wrapping_mul(K).wrapping_add((bz as i64).wrapping_mul(K))
// i.e. it sign-extends oz and bz to 64 bits *individually*, multiplies each by
// K = 116_129_781 in full 64-bit width, then adds.  It never sums oz+bz in i32
// first.  If we summed (oz+bz) as i32 and then sign-extended, the result would
// be wrong whenever oz+bz overflows i32 - a silent mismatch for world
// coordinates near +/-2^31.
//
// oz and bz are taken as separate parameters so each is sign-extended to
// U64 before multiplication, and the call site passes them individually.

fn math_hash(x: i32, y: i32, oz: i32, bz: i32) -> U64 {
    let K       = U64(116129781u, 0u);
    let x_term  = i32_to_u64(x * 3129871);
    // Sign-extend oz and bz to 64 bits first, multiply each by K, then add.
    // This mirrors: (oz as i64)*K + (bz as i64)*K  (never wrapping at 32 bits).
    let z_term  = u64_add(u64_mul(i32_to_u64(oz), K),
                           u64_mul(i32_to_u64(bz), K));
    var l       = u64_xor(u64_xor(x_term, z_term), i32_to_u64(y));
    let inner   = u64_add(u64_mul(l, U64(42317861u, 0u)), U64(11u, 0u));
    l           = u64_mul(l, inner);
    return i64_sra16(l);
}

// xoroshiro128++ step

fn xoroshiro_step(s0: U64, s1: U64) -> U64 {
    return u64_add(u64_rotl17(u64_add(s0, s1)), s0);
}

fn u64_shr40(a: U64) -> u32 { return a.hi >> 8u; }

// Bindings

struct SearchUniforms {
    dlo_lo:              u32,
    dlo_hi:              u32,
    dhi_lo:              u32,
    dhi_hi:              u32,
    // Number of blocks in each rotation's block list (all rotations are the
    // same length; shorter ones are padded with always-pass sentinel blocks).
    blocks_per_rotation: u32,
    candidate_count:     u32,
    batch_base_k_lo:     u32,   // absolute spiral index of this batch's first candidate (low 32 bits)
    start_x:             i32,   // spiral centre
    start_z:             i32,
    // Width (in threads) of one dispatch "row" = workgroups_x * 256. Combined
    // with @builtin(global_invocation_id) this reconstructs a linear
    // candidate index even when the dispatch grid is 2-D (see gpu.rs).
    dispatch_width:      u32,
    // Number of rotations stored back-to-back in the block buffer.
    // The kernel reports a hit when ANY rotation's block-set is satisfied.
    rotation_count:      u32,
    // High 32 bits of the absolute spiral index of this batch's first
    // candidate (was unused `_pad2`). Together with `batch_base_k_lo` this
    // forms a full 64-bit spiral index k - see spiral_pos.
    batch_base_k_hi:     u32,
}

struct BlockData {
    bx:                i32,
    by:                i32,
    bz:                i32,
    prob_threshold:    u32,
    should_be_bedrock: u32,
    pad0:              u32,
    pad1:              u32,
    pad2:              u32,
}

@group(0) @binding(0) var<uniform>             uniforms: SearchUniforms;
@group(0) @binding(1) var<storage, read>       blocks:   array<BlockData>;
@group(0) @binding(2) var<storage, read_write> result:   atomic<u32>;

// Kernel

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The dispatch grid may be 2-D (see gpu.rs for why), so reconstruct a
    // single linear candidate index from (gid.x, gid.y).
    let tid = gid.y * uniforms.dispatch_width + gid.x;
    if (tid >= uniforms.candidate_count) { return; }

    // Compute this thread's spiral position - no buffer read needed.
    // k = batch_base_k + tid, computed as a full U64 so spiral_pos has no
    // ~2.1-billion ceiling (see spiral_pos for details).
    let batch_base_k = U64(uniforms.batch_base_k_lo, uniforms.batch_base_k_hi);
    let k   = u64_add(batch_base_k, u32_to_u64(tid));
    let pos = spiral_pos(k, uniforms.start_x, uniforms.start_z);
    let ox  = pos.x;
    let oz  = pos.y;

    let dlo = U64(uniforms.dlo_lo, uniforms.dlo_hi);
    let dhi = U64(uniforms.dhi_lo, uniforms.dhi_hi);

    var any_rotation_passes = false;
    for (var r = 0u; r < uniforms.rotation_count; r++) {
        if (any_rotation_passes) { break; }
        var rot_passes = true;
        let base = r * uniforms.blocks_per_rotation;
        for (var i = 0u; i < uniforms.blocks_per_rotation; i++) {
            if (!rot_passes) { break; }
            let b          = blocks[base + i];
            // Pass ox+b.bx as a single i32 (safe: x_term uses 32-bit modular
            // multiply which distributes over addition, so overflow is harmless).
            // Pass oz and b.bz *separately* so math_hash can sign-extend each to
            // 64 bits before multiplying - matching the CPU's wrapping_mul order.
            let hash       = math_hash(ox + b.bx, b.by, oz, b.bz);
            let s0         = u64_xor(hash, dlo);
            let res        = xoroshiro_step(s0, dhi);
            let top24      = u64_shr40(res);
            let is_bedrock = top24 < b.prob_threshold;
            rot_passes     = is_bedrock == (b.should_be_bedrock != 0u);
        }
        if (rot_passes) { any_rotation_passes = true; }
    }

    // atomicMin keeps the earliest (spiral-order) match in the batch.
    // The result buffer is pre-initialised to 0xFFFF_FFFF so any hit wins.
    if (any_rotation_passes) { atomicMin(&result, tid); }
}
