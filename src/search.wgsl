// Bedrock formation GPU kernel
//
// Each thread independently computes its spiral position from its global
// thread index, so no candidate position buffer needs to be uploaded.
// 64-bit arithmetic is emulated with U64 (lo, hi) pairs throughout.

// Workgroup size
// Overridable via PipelineCompilationOptions (wgpu >= 0.20).
// To benchmark different sizes, change GPU_WORKGROUP_SIZE in gpu.rs; this
// constant is set at pipeline-creation time to match it.  The GPU dispatch
// calculations in search_batch() use the same constant on the Rust side.
// Candidate values to benchmark on your target hardware: 64, 128, 256, 512.
// The U64 emulation raises per-thread register usage relative to a native-i64
// kernel, which can cap occupancy at the driver level; smaller sizes may
// improve occupancy on register-pressure-sensitive GPUs (typically AMD), while
// larger sizes may improve IPC on warp-wide execution hardware (NVIDIA).
//
// This is a plain `const`, not a pipeline-overridable `override`: the naga
// WGSL frontend bundled with wgpu 0.19 (the version this project is pinned
// to) doesn't parse `override` declarations at all. That means the value
// can't be supplied at pipeline-creation time from gpu.rs; it must be
// edited here directly, and `GPU_WORKGROUP_SIZE` in gpu.rs must be kept in
// sync with whatever you set it to.
const WORKGROUP_SIZE_X: u32 = 256u;

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

// Low 64 bits of a * b where b fits in u32 (b.hi is always 0).
// Saves one u32 multiply in the cross-product vs the general u64_mul:
// the `a.lo * b.hi` term vanishes, leaving only `a.hi * b` for the high word.
fn u64_mul_u32(a: U64, b: u32) -> U64 {
    let ll    = mul32x32(a.lo, b);
    let cross = a.hi * b;   // a.lo * b.hi term vanishes because b.hi == 0
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

// Spiral position helpers
//
// The original monolithic `spiral_pos` is split into three functions so that
// the expensive shell-finding step (f32 sqrt + correction loop) can be amortised
// across an entire workgroup via shared memory (Section 1 of the optimisation
// plan). The three pieces are:
//
//   find_shell   - the sqrt estimate and U64 correction loop; returns the
//                  shell index l, its lower/upper k-range boundaries, and 4*l.
//   leg_from_lj  - the four-leg position dispatch; pure integer arithmetic,
//                  takes (l, j, four_l) and returns the unshifted (ox, oz) offset.
//   spiral_pos_slow - the full reference path: k==0 shortcut, then find_shell +
//                  leg_from_lj + centre shift. Bit-for-bit identical to the
//                  previous monolithic `spiral_pos`. Used as the slow-path
//                  fallback for workgroups that straddle a shell boundary.
//
// The shell spiral coordinate system (mirrors Rust `coords_from_lj`):
//   Shell l, intra-shell offset j = k - (4l^2 - 4l + 1)
//   j < 2l-1              -> Leg 0 Up    (+z): (l,    -(l-1)+j)
//   2l-1 <= j < 4l-1      -> Leg 1 Left  (-x): (l-o,  l)         o = j-(2l-1)
//   4l-1 <= j < 6l-1      -> Leg 2 Down  (-z): (-l,   l-o)       o = j-(4l-1)
//   j >= 6l-1             -> Leg 3 Right (+x): (-l+o, -l)        o = j-(6l-1)
//
// `k` is a full U64 (lo/hi u32 pair). All shell-boundary arithmetic is done
// with exact U64 add/sub/compare; no i32(k) cast, no ~2.1-billion ceiling.
// The only inexact step is the initial f32 sqrt estimate, corrected by the
// loop. `l` stays a u32; final (ox, oz) are cast to i32 (l <= i32::MAX is
// unreachable in practice before 4*l*l overflows i64).

// Return type for find_shell.
struct ShellInfo {
    l:      u32,
    lower:  U64,   // 4l^2 - 4l + 1  (inclusive lower bound of shell l)
    upper:  U64,   // 4l^2 + 4l      (inclusive upper bound of shell l)
    four_l: U64,   // 4 * l          (cached for leg_from_lj)
}

// find_shell: locate the shell that contains k and return its boundary info.
//
// Precondition: k != 0.  The k == 0 origin does not belong to any shell;
// callers must handle it as a special case before invoking this function.
//
// The f32 sqrt estimate is cheap but can be off by a small amount for very
// large k (f32 has 24 mantissa bits); the correction loop adjusts l by ±1
// each iteration until 4l^2-4l+1 <= k <= 4l^2+4l. l^2 is tracked
// incrementally; (l+1)^2 = l^2 + 2l + 1; to avoid a full u64_mul each
// iteration.  The loop breaks on the first iteration in the overwhelmingly
// common case where the f32 estimate is already exact; 64 iterations is a
// generous safety bound.
fn find_shell(k: U64) -> ShellInfo {
    let kf = f32(k.hi) * 4294967296.0 + f32(k.lo);
    var l: u32  = max(1u, u32((1.0 + sqrt(kf)) * 0.5));
    var lsq: U64 = u64_mul(u32_to_u64(l), u32_to_u64(l));

    var four_l: U64;
    var lower:  U64;
    var upper:  U64;
    for (var iter = 0u; iter < 64u; iter = iter + 1u) {
        let l64      = u32_to_u64(l);
        let four_lsq = u64_double(u64_double(lsq));   // 4l^2 (no multiply)
        four_l       = u64_double(u64_double(l64));   // 4l
        upper        = u64_add(four_lsq, four_l);                       // 4l^2 + 4l
        lower        = u64_add(u64_sub(four_lsq, four_l), U64(1u, 0u)); // 4l^2 - 4l + 1

        if (u64_lt(upper, k)) {
            lsq = u64_add(lsq, u64_add(u64_double(l64), U64(1u, 0u))); // l^2 += 2l+1
            l  += 1u;
        } else if (u64_lt(k, lower)) {
            lsq = u64_sub(lsq, u64_sub(u64_double(l64), U64(1u, 0u))); // l^2 -= 2l-1
            l  -= 1u;
        } else {
            break;
        }
    }
    return ShellInfo(l, lower, upper, four_l);
}

// leg_from_lj: given shell index l, intra-shell offset j = k - lower, and
// 4*l (pre-computed by find_shell), return the (ox, oz) position offset
// relative to the spiral centre. Does NOT add the centre coordinates
// callers do that so the shift can be folded with other arithmetic.
//
// j must fit in u32: j <= 8l-2 and l <= i32::MAX guarantees j < u32::MAX.
fn leg_from_lj(l: u32, j: u32, four_l: U64) -> vec2<i32> {
    let l2 = u32_to_u64(l);
    let b0 = u64_sub(u64_double(l2), U64(1u, 0u)).lo;                  // 2l - 1
    let b1 = u64_sub(four_l,         U64(1u, 0u)).lo;                  // 4l - 1
    let b2 = u64_sub(u64_add(four_l, u64_double(l2)), U64(1u, 0u)).lo; // 6l - 1

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
    return vec2<i32>(ox, oz);
}

// spiral_pos_slow: full reference implementation, behaviourally identical to
// the previous monolithic `spiral_pos`. Used on the slow path for workgroups
// that straddle a shell boundary and for the k==0 origin.
//
// Keeping this function unconditionally present and reachable means the fast
// path (below) can be disabled for rollback or A/B testing by replacing the
// fast/slow dispatch in `main` with a single `spiral_pos_slow` call, without
// reverting the refactor.
fn spiral_pos_slow(k: U64, sx: i32, sz: i32) -> vec2<i32> {
    if (k.lo == 0u && k.hi == 0u) { return vec2<i32>(sx, sz); }
    let s   = find_shell(k);
    let j   = u64_sub(k, s.lower).lo;
    let off = leg_from_lj(s.l, j, s.four_l);
    return vec2<i32>(sx + off.x, sz + off.y);
}

// Workgroup-shared anchor
//
// For any reasonably large spiral radius the shell length (8*l) is much larger
// than a 256-thread workgroup, so all threads in a workgroup almost always
// fall inside the same shell l. Only thread 0 (local_invocation_index == 0)
// needs to run the expensive find_shell; every other thread derives its
// intra-shell offset j with a single U64 subtraction and reads the leg
// boundaries from these shared variables.
//
// The four shared variables hold the anchor shell state written by thread 0 and
// read by all threads after workgroupBarrier(). They are only valid after the
// barrier; reading them before the barrier is a data race.
var<workgroup> wg_l:      u32;
var<workgroup> wg_lower:  U64;
var<workgroup> wg_upper:  U64;
var<workgroup> wg_four_l: U64;

// math_hash_terms
//
// Takes the already-combined 64-bit `term_x`/`term_z` hash terms rather than
// raw coordinates. The caller is responsible for computing:
//
//   term_x = i32_to_u64( (ox * 3_129_871) [wrapping i32] + b.bx_k )
//   term_z = u64_add( oz_term, U64(b.bz_k_lo, b.bz_k_hi) )
//
// where `b.bx_k`/`b.bz_k_lo/hi` are the block's own coordinate pre-multiplied
// by the same constants on the CPU, and `oz_term = i32_to_u64(oz) * K` is the
// thread's own z-multiply, computed once per thread (see `main`). Because
// wrapping 32-bit multiplication distributes over addition
// ((ox+bx)*K == ox*K + bx*K mod 2^32), and 64-bit multiplication distributes
// over addition exactly, splitting the multiply this way and pushing the
// `ox`/`oz`-dependent half out of the per-block loop is mathematically
// identical to the original "multiply the sum every block" approach, just
// far cheaper: an O(1) multiply per thread instead of an O(blocks) one. This
// exactly mirrors the CPU's `check_formation_with_terms` / `bx_hash_term` /
// `bz_hash_term` split in core.rs.
fn math_hash_terms(term_x: U64, term_z: U64, y: i32) -> U64 {
    var l       = u64_xor(u64_xor(term_x, term_z), i32_to_u64(y));
    let inner   = u64_add(u64_mul_u32(l, 42317861u), U64(11u, 0u));
    l           = u64_mul(l, inner);
    return i64_sra16(l);
}

// xoroshiro128++ step

fn xoroshiro_step(s0: U64, s1: U64) -> U64 {
    return u64_add(u64_rotl17(u64_add(s0, s1)), s0);
}

fn u64_shr40(a: U64) -> u32 { return a.hi >> 8u; }

// Bindings
//
// NOTE: when push constants are enabled in gpu.rs, the Rust side replaces the
// `@group(0) @binding(0) var<uniform>` declaration below with
// `var<push_constant>` at pipeline-creation time (runtime string patch).
// The blocks and result bindings are unaffected and stay as storage buffers.

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
    // Width (in threads) of one dispatch "row" = workgroups_x * WORKGROUP_SIZE_X.
    // Combined with @builtin(global_invocation_id) this reconstructs a linear
    // candidate index even when the dispatch grid is 2-D (see gpu.rs).
    dispatch_width:      u32,
    // Number of rotations stored back-to-back in the block buffer.
    // The kernel reports a hit when ANY rotation's block-set is satisfied.
    rotation_count:      u32,
    // High 32 bits of the absolute spiral index of this batch's first
    // candidate (was unused `_pad2`). Together with `batch_base_k_lo` this
    // forms a full 64-bit spiral index k - see spiral_pos_slow.
    batch_base_k_hi:     u32,
}

// Must match `GpuBlock` in gpu.rs (32 bytes).
struct BlockData {
    // Precomputed (bx as u32).wrapping_mul(3_129_871) - see math_hash_terms.
    bx_k:              u32,
    by:                i32,
    // Precomputed (bz as i64).wrapping_mul(116_129_781), low/high 32 bits.
    bz_k_lo:           u32,
    bz_k_hi:           u32,
    prob_threshold:    u32,
    should_be_bedrock: u32,
    pad0:              u32,
    pad1:              u32,
}

@group(0) @binding(0) var<uniform>             uniforms: SearchUniforms;
@group(0) @binding(1) var<storage, read>       blocks:   array<BlockData>;
@group(0) @binding(2) var<storage, read_write> result:   atomic<u32>;

// Kernel

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_index) li:  u32,
) {
    // The dispatch grid may be 2-D (see gpu.rs for why), so reconstruct a
    // single linear candidate index from (gid.x, gid.y).
    let tid = gid.y * uniforms.dispatch_width + gid.x;

    // k = batch_base_k + tid as a full U64. Computed before the early-return
    // guard so that thread 0 can write the workgroup anchor unconditionally
    // the barrier below must be reached by ALL invocations in the workgroup,
    // including trailing threads in the last workgroup whose tid >= candidate_count.
    let batch_base_k = U64(uniforms.batch_base_k_lo, uniforms.batch_base_k_hi);
    let k = u64_add(batch_base_k, u32_to_u64(tid));

    // Workgroup-shared spiral anchor
    //
    // Thread 0 (li == 0) calls find_shell for its own k and writes the four
    // shared anchor variables. Every other thread reads them after the barrier
    // and uses them to compute its own intra-shell offset j with a single
    // U64 subtraction, skipping the sqrt and the correction loop entirely.
    //
    // Why the anchor key invariant holds (k >= wg_lower for all threads):
    //   Thread 0 holds the smallest k in the workgroup (tid values are
    //   contiguous within a workgroup and increase with li, so k0 <= k_i for
    //   all i > 0). find_shell guarantees k0 >= wg_lower for the shell it
    //   places k0 in. Therefore k_i >= k0 >= wg_lower for all i > 0.
    //   The lower-bound check can be skipped; only the upper bound is uncertain.
    //
    // k == 0 sentinel (k0 == 0, the very first candidate of the search):
    //   find_shell is undefined for k == 0 (no shell contains the origin).
    //   Thread 0 instead writes a sentinel with wg_upper == wg_lower == 0.
    //   For any thread with k > 0: u64_lt(wg_upper=0, k>0) is true, so the
    //   fast-path condition `!u64_lt(wg_upper, k)` is false: all threads
    //   fall through to spiral_pos_slow, which handles k == 0 internally.
    //   Thread 0 itself (k == 0) is caught by the explicit k == 0 guard in
    //   the position dispatch below before any u64_sub is attempted.
    if (li == 0u) {
        if (k.lo == 0u && k.hi == 0u) {
            // Sentinel: forces every thread in this workgroup to slow path.
            wg_l      = 0u;
            wg_lower  = U64(0u, 0u);
            wg_upper  = U64(0u, 0u);
            wg_four_l = U64(0u, 0u);
        } else {
            let s     = find_shell(k);
            wg_l      = s.l;
            wg_lower  = s.lower;
            wg_upper  = s.upper;
            wg_four_l = s.four_l;
        }
    }

    // Synchronise: all threads must wait here before reading the wg_* variables.
    //
    // The barrier is placed BEFORE the candidate_count early-return, not after,
    // because WGSL requires every invocation in a workgroup to execute the same
    // barrier calls in the same order. In the last dispatch workgroup some
    // threads may have tid >= candidate_count; they must still participate in
    // the barrier and only return afterwards. If thread 0's tid >= candidate_count
    // then ALL threads in the workgroup are out of bounds (thread 0 has the
    // smallest tid), so the barrier is correctly reached by all, then all return.
    workgroupBarrier();

    // Out-of-bounds guard; placed after the barrier (see note above).
    if (tid >= uniforms.candidate_count) { return; }

    // Spiral position dispatch
    //
    // Fast path: k is within the anchor shell [wg_lower, wg_upper].
    //   The lower-bound invariant holds automatically (see anchor comment above),
    //   so only the upper bound needs an explicit check.
    //   j = k - wg_lower fits in u32: j <= 8l-2 for any shell, << u32::MAX.
    //
    // Slow path: k is in a later shell than the anchor (shell boundary crossed
    //   within this workgroup), or the k == 0 sentinel was written. In both
    //   cases spiral_pos_slow runs the full find_shell + leg_from_lj sequence.
    //
    // k == 0 guard: must precede the fast-path check because the sentinel
    //   writes wg_upper == 0, and u64_lt(0, 0) == false makes the condition
    //   `!u64_lt(wg_upper, k)` true for k == 0; which would incorrectly
    //   enter the fast path and underflow u64_sub(0, wg_lower).
    var pos: vec2<i32>;
    if (k.lo == 0u && k.hi == 0u) {
        // k == 0 is the spiral origin. Occurs at most once, in the very first
        // batch, for the very first candidate. Inline the shortcut here rather
        // than calling spiral_pos_slow to make the control flow explicit.
        pos = vec2<i32>(uniforms.start_x, uniforms.start_z);
    } else if (!u64_lt(wg_upper, k)) {
        // Fast path: k is in the anchor shell. Amortises find_shell across all
        // ~255/256 threads that share a shell with thread 0. For large l (the
        // common case far into a search) the shell spans billions of candidates,
        // so virtually every thread in every workgroup takes this branch.
        let j   = u64_sub(k, wg_lower).lo;
        let off = leg_from_lj(wg_l, j, wg_four_l);
        pos     = vec2<i32>(uniforms.start_x + off.x, uniforms.start_z + off.y);
    } else {
        // Slow path: shell boundary crosses this workgroup. Typically affects
        // only one workgroup per shell transition. At l == 1 every workgroup
        // uses the slow path (shell length == 8 << 256 threads), but shells
        // grow as 8l so this vanishes rapidly with search depth.
        pos = spiral_pos_slow(k, uniforms.start_x, uniforms.start_z);
    }
    let ox = pos.x;
    let oz = pos.y;

    let dlo = U64(uniforms.dlo_lo, uniforms.dlo_hi);
    let dhi = U64(uniforms.dhi_lo, uniforms.dhi_hi);

    // Hoisted per-thread hash terms. These depend only on this thread's
    // (ox, oz) position, so computing them once here - instead of once per
    // block, as the naive version would - turns what was up to
    // rotation_count * blocks_per_rotation 64-bit multiplies per thread into
    // a single 32-bit and a single 64-bit multiply, with only a cheap U64 add
    // remaining inside the block loop. See math_hash_terms for the algebra.
    let ox_k    = bitcast<u32>(ox) * 3129871u;
    let oz_term = u64_mul(i32_to_u64(oz), U64(116129781u, 0u));

    // rotation_count == 1 fast path
    // `rotation_count` is a uniform; this branch is divergence-free: every
    // thread in the workgroup reads the same value and takes the same side.
    // In the single-rotation case (overwhelmingly common; four rotations are
    // only active when the formation has full rotational symmetry) the general
    // loop's `any_rotation_passes` bool and the `if (any_rotation_passes) { break; }`
    // guard are purely redundant overhead. This path replaces them with a direct
    // early-return on the first failing block, which also allows the compiler to
    // see that there is no per-block write to `any_rotation_passes`/`rot_passes`
    // and may unlock additional instruction folding.
    if (uniforms.rotation_count == 1u) {
        for (var i = 0u; i < uniforms.blocks_per_rotation; i++) {
            let b          = blocks[i];
            let term_x     = i32_to_u64(bitcast<i32>(ox_k + b.bx_k));
            let term_z     = u64_add(oz_term, U64(b.bz_k_lo, b.bz_k_hi));
            let hash       = math_hash_terms(term_x, term_z, b.by);
            let s0         = u64_xor(hash, dlo);
            let res        = xoroshiro_step(s0, dhi);
            let top24      = u64_shr40(res);
            let is_bedrock = top24 < b.prob_threshold;
            if (!(is_bedrock == (b.should_be_bedrock != 0u))) { return; }
        }
        atomicMin(&result, tid);
        return;
    }

    // General multi-rotation path
    // atomicMin keeps the earliest (spiral-order) match in the batch.
    // The result buffer is pre-initialised to 0xFFFF_FFFF so any hit wins.
    var any_rotation_passes = false;
    for (var r = 0u; r < uniforms.rotation_count; r++) {
        if (any_rotation_passes) { break; }
        var rot_passes = true;
        let base = r * uniforms.blocks_per_rotation;
        for (var i = 0u; i < uniforms.blocks_per_rotation; i++) {
            if (!rot_passes) { break; }
            let b          = blocks[base + i];
            // term_x/term_z combine this thread's hoisted terms with the
            // block's precomputed bx_k/bz_k via cheap adds only - see
            // math_hash_terms for why this is exact, not approximate.
            let term_x     = i32_to_u64(bitcast<i32>(ox_k + b.bx_k));
            let term_z     = u64_add(oz_term, U64(b.bz_k_lo, b.bz_k_hi));
            let hash       = math_hash_terms(term_x, term_z, b.by);
            let s0         = u64_xor(hash, dlo);
            let res        = xoroshiro_step(s0, dhi);
            let top24      = u64_shr40(res);
            let is_bedrock = top24 < b.prob_threshold;
            rot_passes     = is_bedrock == (b.should_be_bedrock != 0u);
        }
        if (rot_passes) { any_rotation_passes = true; }
    }

    if (any_rotation_passes) { atomicMin(&result, tid); }
}
