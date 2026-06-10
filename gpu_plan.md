# GPU Computation Plan — Bedrock Formation Finder

## Overview

This document is a step-by-step implementation plan to add optional GPU-accelerated
searching to `bedrockformation`. The search hot path (computing `math_hash` →
xoroshiro128++ → threshold for every candidate position) is embarrassingly parallel
and maps directly to GPU compute. The existing CPU path (AVX-512 / AVX2 / scalar via
Rayon) remains **completely unchanged** and is the default; GPU is an explicit opt-in
toggle in the UI.

---

## 1. Why This Works on a GPU

The inner kernel does the following per candidate `(ox, oz)`:

1. `math_hash(ox + bx, y, oz + bz)` — a few multiplies and XORs
2. One xoroshiro128++ step — rotate, add, XOR
3. Integer threshold comparison (`top24 < prob_threshold`)
4. AND the result across all blocks in the formation

These are independent across all candidates, with no branching that diverges between
lanes and no shared mutable state. A modern GPU can run tens of thousands of these in
parallel per wave, giving order-of-magnitude throughput gains over even AVX-512 for
large search radii.

---

## 2. Technology Choice — `wgpu`

Use the [`wgpu`](https://github.com/gfx-rs/wgpu) crate (WebGPU over Vulkan / Metal /
DX12 / OpenGL). Reasons:

- **`iced` already depends on `wgpu`** (it is the default renderer), so adding
  `wgpu` as a direct dependency adds zero new transitive weight.
- Cross-platform: Vulkan on Linux/Windows, Metal on macOS, DX12 on Windows.
- Shaders are written in **WGSL**, which is human-readable and easy to generate
  or review.
- No CUDA/ROCm SDK requirement; works on integrated GPUs as well as discrete ones.

---

## 3. New Dependencies (`Cargo.toml`)

```toml
[dependencies]
# existing deps omitted for brevity
wgpu     = "22"          # match whatever iced pulls in; run `cargo tree` to confirm
pollster = "0.3"         # block-on helper for the async wgpu init at startup
bytemuck = { version = "1", features = ["derive"] }  # safe cast for POD structs
```

---

## 4. Architecture Changes

### 4.1 `SimdLevel` is untouched — add a separate `ComputeBackend` enum

**Do not modify `SimdLevel` in any way.** It remains exactly as written and continues
to drive the CPU path. Add a new, independent enum alongside it:

```rust
// Keep SimdLevel exactly as-is:
//   enum SimdLevel { Avx512, Avx2, Scalar }
//
// Add a new top-level enum that the search caller selects from:
enum ComputeBackend {
    /// wgpu compute shader (GPU opt-in).
    Gpu(Arc<GpuContext>),
    /// Existing CPU path — SimdLevel is detected as before inside run_search.
    Cpu,
}
```

When GPU is requested, `run_search` receives a `Some(Arc<GpuContext>)` and uses the
GPU dispatch loop. Otherwise it falls through to the existing `detect_simd()` /
`run_chunk_batch` path untouched. At no point is `SimdLevel` or any existing CPU
kernel modified.

### 4.2 New `GpuContext` Struct

Create a module `mod gpu` (file `src/gpu.rs`) containing:

```rust
pub struct GpuContext {
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout cached at creation time.
    bgl:      wgpu::BindGroupLayout,
    /// Preallocated candidate position buffer (max chunk size, reused each dispatch).
    candidate_buf: wgpu::Buffer,
    /// Preallocated result buffer (4 bytes, read-write).
    result_buf:    wgpu::Buffer,
    /// Staging buffer for reading back the result (MAP_READ).
    staging_buf:   wgpu::Buffer,
}
```

`GpuContext::new() -> Option<GpuContext>` — returns `None` when no GPU adapter is
found, triggering automatic CPU fallback.

### 4.3 App State Changes

Add two fields to the `App` struct:

```rust
gpu_ctx:  Option<Arc<GpuContext>>,  // None = GPU not available or not yet initialised
use_gpu:  bool,                     // controlled by the UI toggle (default: false)
```

Also add `gpu_ctx: None, use_gpu: false` to `App::default()`.

**Important:** `GpuContext::new()` is async (wgpu adapter/device requests are
async). It must **not** be called inside `Default::default()`. Instead, override
`App::new()` to construct the default struct first, then initialise the GPU field:

```rust
fn new(_flags: ()) -> (Self, Command<Message>) {
    let mut app = App::default();
    // pollster::block_on is safe here because new() runs on the main thread
    // before the iced event loop starts — not inside the async executor.
    app.gpu_ctx = pollster::block_on(GpuContext::new()).map(Arc::new);
    (app, Command::none())
}
```

---

## 5. 64-Bit Arithmetic in WGSL — Portability Strategy

> **This section is critical. Read it before writing a single line of WGSL.**

The entire compute kernel requires 64-bit integer arithmetic:

- `math_hash` — signed 64-bit multiplies and XORs.
- `xoroshiro128++` — unsigned 64-bit add, rotate, XOR.
- `SearchUniforms` — `dlo` and `dhi` are both 64-bit values.
- The `main` kernel body — every intermediate value between the hash and `top24` is 64-bit.

WGSL native `i64`/`u64` types exist only behind the `shader-int64` extension
(`wgpu::Features::SHADER_INT64`). This feature is **not universally available**:
it is absent on many integrated GPUs, on macOS Metal (as of wgpu 22), and on older
Vulkan drivers. These are exactly the adapters most users without a discrete GPU
will have.

### Recommended approach: always use the u32-pair emulation

Implement all 64-bit arithmetic using `(lo: u32, hi: u32)` struct pairs throughout
the shader. This is portable to every adapter that supports compute shaders at all,
and the performance difference is small compared to the memory bandwidth of the
candidate buffer.

```wgsl
// Portable 64-bit integer represented as (lo, hi) u32 words.
// lo = bits 0-31, hi = bits 32-63. Two's-complement for signed values.
struct U64 { lo: u32, hi: u32 }
```

Required helper functions (implement all of these in WGSL before writing the kernel):

```wgsl
// XOR
fn u64_xor(a: U64, b: U64) -> U64 {
    return U64(a.lo ^ b.lo, a.hi ^ b.hi);
}

// Wrapping addition with carry
fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

// Rotate left by N bits (N must be a compile-time constant; implement one per N used)
fn u64_rotl17(a: U64) -> U64 {
    // rotate_left(17): hi_bits = top 15 of hi | bottom 17 of lo shifted
    return U64(
        (a.hi << 17u) | (a.lo >> 15u),
        (a.lo << 17u) | (a.hi >> 15u),
    );
}
fn u64_rotl49(a: U64) -> U64 {
    // rotate_left(49) = rotate_left(32+17) => swap words then rotl17
    return u64_rotl17(U64(a.hi, a.lo));
}
fn u64_rotl28(a: U64) -> U64 {
    return U64(
        (a.hi << 28u) | (a.lo >> 4u),
        (a.lo << 28u) | (a.hi >> 4u),
    );
}

// Shift right by 40 bits (used to extract top24)
fn u64_shr40(a: U64) -> u32 {
    // Result fits in u32: only the top 24 bits of the original are kept.
    return a.hi >> 8u;
}

// 64-bit wrapping multiply — needed for math_hash.
// Uses the standard (lo*lo, lo*hi + hi*lo) schoolbook approach.
fn u64_mul(a: U64, b: U64) -> U64 {
    // Split into 16-bit halves to avoid u32 overflow in intermediate products.
    let a0 = a.lo & 0xFFFFu;
    let a1 = a.lo >> 16u;
    let a2 = a.hi & 0xFFFFu;
    let a3 = a.hi >> 16u;
    let b0 = b.lo & 0xFFFFu;
    let b1 = b.lo >> 16u;
    let b2 = b.hi & 0xFFFFu;
    let b3 = b.hi >> 16u;
    // Only terms that contribute to the low 64 bits:
    let t0  = a0 * b0;
    let t1  = a0 * b1 + a1 * b0;
    let t2  = a0 * b2 + a1 * b1 + a2 * b0;
    let t3  = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
    let lo0 = t0 & 0xFFFFu;
    let lo1 = ((t0 >> 16u) + (t1 & 0xFFFFu));
    let lo2 = (lo1 >> 16u) + (t1 >> 16u) + (t2 & 0xFFFFu);
    let lo3 = (lo2 >> 16u) + (t2 >> 16u) + (t3 & 0xFFFFu);
    let lo  = lo0 | ((lo1 & 0xFFFFu) << 16u);
    let hi  = (lo2 & 0xFFFFu) | ((lo3 & 0xFFFFu) << 16u);
    return U64(lo, hi);
}

// Arithmetic right shift by 16 (for math_hash final step).
// Replicates Rust `(l as i64) >> 16` (sign-extending from bit 63).
fn i64_sra16(a: U64) -> U64 {
    let sign_fill = select(0u, 0xFFFFu, (a.hi & 0x80000000u) != 0u);
    let new_hi = (a.hi >> 16u) | (sign_fill << 16u);
    let new_lo = (a.lo >> 16u) | (a.hi << 16u);
    return U64(new_lo, new_hi);
}

// Sign-extend i32 to U64 (two's complement).
fn i32_to_u64(v: i32) -> U64 {
    let lo = u32(v);
    let hi = select(0u, 0xFFFFFFFFu, v < 0);
    return U64(lo, hi);
}
```

If you later detect `SHADER_INT64` at runtime and want a fast path, you can compile
a second shader variant with native types. But do not make this the default — always
ship the portable u32-pair shader.

---

## 6. WGSL Compute Shader

Create `src/search.wgsl`. Because all 64-bit values are represented as u32 pairs,
`SearchUniforms` stores `dlo` and `dhi` as split words:

### Binding table

| Binding | Type | Contents |
|---------|------|----------|
| 0 | uniform | `SearchUniforms` (dlo, dhi as u32 pairs, block count, candidate count) |
| 1 | read-only storage | `BlockData[]` (bx, by, bz, prob_threshold, should_be_bedrock) |
| 2 | read-only storage | `CandidatePos[]` (ox, oz) — filled by CPU in spiral order each chunk |
| 3 | read-write storage | `atomic<u32>` result index (initialised to 0xFFFFFFFF = not found) |

### Full shader

```wgsl
// --- 64-bit helper types and functions (paste u64 helpers from §5 here) ---
struct U64 { lo: u32, hi: u32 }
// ... (all helpers from §5) ...

// --- Bindings ---

struct SearchUniforms {
    dlo_lo:          u32,   // bits  0-31 of dlo (i64)
    dlo_hi:          u32,   // bits 32-63 of dlo
    dhi_lo:          u32,   // bits  0-31 of dhi (i64)
    dhi_hi:          u32,   // bits 32-63 of dhi
    block_count:     u32,
    candidate_count: u32,
    _pad0:           u32,
    _pad1:           u32,   // pad to 32 bytes (uniform buffer alignment)
}

struct BlockData {
    bx:                i32,
    by:                i32,
    bz:                i32,
    prob_threshold:    u32,   // (probability * 2^24) as u32
    should_be_bedrock: u32,   // 0 = air, 1 = bedrock
    _pad0:             u32,
    _pad1:             u32,
    _pad2:             u32,   // pad to 32 bytes
}

struct CandidatePos { ox: i32, oz: i32 }

@group(0) @binding(0) var<uniform>             uniforms:   SearchUniforms;
@group(0) @binding(1) var<storage, read>       blocks:     array<BlockData>;
@group(0) @binding(2) var<storage, read>       candidates: array<CandidatePos>;
@group(0) @binding(3) var<storage, read_write> result:     atomic<u32>;

// --- math_hash ---
//
// Replicates the Rust scalar:
//   let l = ((x as i64 * 3_129_871) ^ (z as i64 * 116_129_781) ^ y as i64);
//   let l = l.wrapping_mul(l).wrapping_mul(42_317_861).wrapping_add(l.wrapping_mul(11));
//   l >> 16
//
// x-path: Java idiom `(long)(x * 3129871)` — multiply in i32 wrapping space,
//         then sign-extend.  Replicated here: i32 mul → sign-extend to U64.
// z-path: Java idiom `(long)z * 116129781L` — true i64 multiply.
// y-path: sign-extend i32 y to U64.
fn math_hash(x: i32, y: i32, z: i32) -> U64 {
    // x term: wrapping i32 mul then sign-extend (matches Java `(long)(x*K)`)
    let x_mul_i32 = x * 3129871;            // i32 wrapping multiply
    let x_term    = i32_to_u64(x_mul_i32);  // sign-extend to 64 bits

    // z term: true i64 mul — use u64_mul on sign-extended operands
    let z64    = i32_to_u64(z);
    let kz     = U64(116129781u, 0u);
    let z_term = u64_mul(z64, kz);

    // y term
    let y_term = i32_to_u64(y);

    // l = x_term ^ z_term ^ y_term
    var l = u64_xor(u64_xor(x_term, z_term), y_term);

    // l = l * (l * 42_317_861 + 11)
    let k1    = U64(42317861u, 0u);
    let k2    = U64(11u, 0u);
    let inner = u64_add(u64_mul(l, k1), k2);
    l = u64_mul(l, inner);

    // arithmetic right shift 16 (signed)
    return i64_sra16(l);
}

// --- xoroshiro128++ single step ---
//
// Replicates Rust:
//   let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);
fn xoroshiro_step(s0: U64, s1: U64) -> U64 {
    let sum = u64_add(s0, s1);
    return u64_add(u64_rotl17(sum), s0);
}

// --- Main kernel ---

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.candidate_count { return; }

    let ox = candidates[idx].ox;
    let oz = candidates[idx].oz;

    let dlo = U64(uniforms.dlo_lo, uniforms.dlo_hi);
    let dhi = U64(uniforms.dhi_lo, uniforms.dhi_hi);

    var passes = true;
    for (var i = 0u; i < uniforms.block_count; i++) {
        if !passes { break; }
        let b    = blocks[i];
        let hash = math_hash(ox + b.bx, b.by, oz + b.bz);
        let s0   = u64_xor(hash, dlo);
        // s1 = dhi (constant across all candidates for a given search)
        let res   = xoroshiro_step(s0, dhi);
        let top24 = u64_shr40(res);   // returns u32, value in [0, 2^24)
        let is_bedrock = top24 < b.prob_threshold;
        passes = is_bedrock == (b.should_be_bedrock != 0u);
    }

    if passes {
        // Record the lowest-index match (earliest in spiral order).
        atomicMin(&result, idx);
    }
}
```

---

## 7. GPU Buffer Layout (Rust side)

### Uniform buffer — 32 bytes, updated once per chunk

`dlo` and `dhi` are stored as u32 pairs to match the portable shader above.

```rust
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct SearchUniforms {
    dlo_lo:          u32,   // dlo as u64: bits  0-31
    dlo_hi:          u32,   // dlo as u64: bits 32-63
    dhi_lo:          u32,   // dhi as u64: bits  0-31
    dhi_hi:          u32,   // dhi as u64: bits 32-63
    block_count:     u32,
    candidate_count: u32,
    _pad:            [u32; 2],   // 32 bytes total
}

impl SearchUniforms {
    fn new(dlo: i64, dhi: i64, block_count: u32, candidate_count: u32) -> Self {
        let dlo = dlo as u64;
        let dhi = dhi as u64;
        Self {
            dlo_lo: dlo as u32,
            dlo_hi: (dlo >> 32) as u32,
            dhi_lo: dhi as u32,
            dhi_hi: (dhi >> 32) as u32,
            block_count,
            candidate_count,
            _pad: [0; 2],
        }
    }
}
```

### Block storage buffer — written once per search

```rust
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct GpuBlock {
    bx:                i32,
    by:                i32,
    bz:                i32,
    prob_threshold:    u32,   // (probability * 2^24) as u32, same as Blocks::prob_threshold
    should_be_bedrock: u32,   // 0 = air, 1 = bedrock
    _pad:              [u32; 3],   // 32 bytes total, matches WGSL BlockData
}
```

### Candidate position buffer — rewritten every chunk

**Size: `GROUPS_PER_CHUNK * 8 * 8` bytes = `512 * 8 * 8 = 32,768 bytes`.**

Each entry is two `i32` values (8 bytes). There are `GROUPS_PER_CHUNK * 8 = 4,096`
positions per chunk. Allocate this buffer once at `GpuContext::new()` time and reuse
it across all chunk dispatches via `queue.write_buffer`.

### Result buffer — 4 bytes

Written to `0xFFFF_FFFFu32` before each dispatch using `queue.write_buffer`. Read
back via a `MAP_READ` staging buffer after the dispatch completes. The value is the
spiral-order index within the chunk (0..4095), or `0xFFFF_FFFF` if no match.

---

## 8. CPU-Side Candidate Generation

Before each GPU dispatch, the CPU must pre-generate the `4,096` candidate `(ox, oz)`
positions for the current chunk using the existing spiral state machine. This is the
same work the CPU batch loop does internally — it is just made explicit and packed
into the candidate buffer instead of being consumed inline.

```rust
/// Fill `out_xs` and `out_zs` with the spiral positions for one chunk starting at
/// `chunk_base_group` (a group index, not a position index).
/// Both slices must have length exactly `GROUPS_PER_CHUNK * 8`.
fn fill_chunk_candidates(
    chunk_base_group: i64,
    start_x: i32,
    start_z: i32,
    out_xs: &mut [i32],  // length = GROUPS_PER_CHUNK * 8 = 4096
    out_zs: &mut [i32],
) {
    let base_k = chunk_base_group * 8;
    let (mut x, mut z, mut l, mut j, mut dx, mut dz, mut next_leg_j) =
        spiral_coords_with_state(base_k, start_x, start_z);
    let mut xs = [0i32; 8];
    let mut zs = [0i32; 8];
    for g in 0..(GROUPS_PER_CHUNK as usize) {
        fill_group_from_state(
            &mut x, &mut z, &mut l, &mut j,
            &mut dx, &mut dz, &mut next_leg_j,
            &mut xs, &mut zs,
            start_x, start_z,
        );
        let base = g * 8;
        out_xs[base..base + 8].copy_from_slice(&xs);
        out_zs[base..base + 8].copy_from_slice(&zs);
    }
}
```

`GpuContext::search_chunk` calls this before writing the candidate buffer. The
interleaved layout sent to the GPU is then:

```rust
// Interleave xs/zs into the candidate buffer as pairs of i32.
let mut interleaved = vec![0i32; GROUPS_PER_CHUNK as usize * 8 * 2];
for i in 0..(GROUPS_PER_CHUNK as usize * 8) {
    interleaved[i * 2]     = chunk_xs[i];
    interleaved[i * 2 + 1] = chunk_zs[i];
}
queue.write_buffer(&self.candidate_buf, 0, bytemuck::cast_slice(&interleaved));
```

---

## 9. GPU Search Function

```rust
impl GpuContext {
    /// Search one chunk on the GPU.
    ///
    /// `chunk_base_group` is the group index of the first group in this chunk
    /// (same coordinate space as `batch_start_group * CHUNKS_PER_BATCH + ci * GROUPS_PER_CHUNK`
    /// in the CPU batch loop).
    ///
    /// Returns `Some(spiral_index)` if a match was found (the absolute spiral
    /// position index of the earliest match in this chunk), or `None` otherwise.
    pub fn search_chunk(
        &self,
        chunk_base_group: i64,
        start_x:          i32,
        start_z:          i32,
        dlo:              i64,
        dhi:              i64,
        blocks:           &[GpuBlock],
    ) -> Option<i64> {
        let candidate_count = (GROUPS_PER_CHUNK * 8) as u32;  // always 4096

        // 1. Pre-generate candidate positions on the CPU.
        let mut chunk_xs = vec![0i32; candidate_count as usize];
        let mut chunk_zs = vec![0i32; candidate_count as usize];
        fill_chunk_candidates(chunk_base_group, start_x, start_z,
                              &mut chunk_xs, &mut chunk_zs);

        // 2. Write uniform buffer.
        let uniforms = SearchUniforms::new(dlo, dhi, blocks.len() as u32, candidate_count);
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // 3. Write block buffer (written once per run_search call in practice;
        //    kept here for clarity — see §10 for the optimised placement).
        self.queue.write_buffer(&self.block_buf, 0, bytemuck::cast_slice(blocks));

        // 4. Write candidate buffer (interleaved x/z pairs).
        let mut interleaved = vec![0i32; candidate_count as usize * 2];
        for i in 0..candidate_count as usize {
            interleaved[i * 2]     = chunk_xs[i];
            interleaved[i * 2 + 1] = chunk_zs[i];
        }
        self.queue.write_buffer(&self.candidate_buf, 0,
                                bytemuck::cast_slice(&interleaved));

        // 5. Reset result buffer to 0xFFFF_FFFF (= no match).
        let no_match = 0xFFFF_FFFFu32;
        self.queue.write_buffer(&self.result_buf, 0,
                                bytemuck::bytes_of(&no_match));

        // 6. Build bind group and encode dispatch.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.block_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.candidate_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.result_buf.as_entire_binding() },
            ],
            label: None,
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Dispatch: ceil(4096 / 256) = 16 workgroups.
            cpass.dispatch_workgroups((candidate_count + 255) / 256, 1, 1);
        }
        // Copy result to staging buffer for CPU readback.
        encoder.copy_buffer_to_buffer(&self.result_buf, 0, &self.staging_buf, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));

        // 7. Map staging buffer and read result.
        let slice = self.staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let raw: u32 = {
            let view = slice.get_mapped_range();
            bytemuck::pod_read_unaligned(&view[..4])
        };
        self.staging_buf.unmap();

        // 8. Translate intra-chunk index back to absolute spiral index.
        if raw == 0xFFFF_FFFF {
            None
        } else {
            Some(chunk_base_group * 8 + raw as i64)
        }
    }
}
```

---

## 10. Integration into `run_search`

`run_search` receives an `Option<Arc<GpuContext>>`. When `Some`, it converts blocks
to `Vec<GpuBlock>` once and then dispatches each chunk to the GPU instead of
`run_chunk_batch`. The existing CPU path (`run_chunk_batch` / SIMD / scalar) is
called when the option is `None` and is not modified in any way.

```rust
fn run_search(
    seed:        i64,
    start_x:     i32,
    start_z:     i32,
    bt:          BedrockType,
    mut blocks:  Vec<Block>,
    cancel:      Arc<AtomicBool>,
    stop_early:  Arc<AtomicBool>,
    progress_cb: Option<&dyn Fn(i64)>,
    gpu_ctx:     Option<Arc<GpuContext>>,   // <-- new parameter
) -> Result<Option<(i32, i32)>, String> {
    // ... (existing validation and sort unchanged) ...

    let (dlo, dhi) = compute_deriver_seeds(seed, bt);
    let simd       = detect_simd();

    let blocks = Blocks::from_vec(blocks);   // existing SoA conversion, unchanged

    // Convert to GPU block format once, before the search loop.
    let gpu_blocks: Option<Vec<GpuBlock>> = gpu_ctx.as_ref().map(|_| {
        (0..blocks.len()).map(|i| GpuBlock {
            bx:                blocks.x[i],
            by:                blocks.y[i],
            bz:                blocks.z[i],
            prob_threshold:    blocks.prob_threshold[i] as u32,
            should_be_bedrock: blocks.should_be_bedrock[i] as u32,
            _pad:              [0; 3],
        }).collect()
    });

    let mut batch_start_group: i64 = 0;

    loop {
        if cancel.load(Ordering::Relaxed) || stop_early.load(Ordering::Relaxed) {
            return Ok(None);
        }

        let batch_base = batch_start_group;

        // GPU path: iterate chunks directly, bypassing run_chunk_batch.
        let found_spiral_idx: Option<i64> = if let (Some(ctx), Some(gblocks)) =
            (gpu_ctx.as_ref(), gpu_blocks.as_deref())
        {
            let mut found: Option<i64> = None;
            'chunks: for ci in 0..CHUNKS_PER_BATCH {
                let chunk_base_group = batch_base + ci * GROUPS_PER_CHUNK;
                if let Some(spiral_idx) =
                    ctx.search_chunk(chunk_base_group, start_x, start_z,
                                     dlo, dhi, gblocks)
                {
                    found = Some(spiral_idx);
                    break 'chunks;
                }
            }
            found
        } else {
            // CPU path: existing run_chunk_batch / SIMD / scalar, completely unchanged.
            // (existing match simd { Avx512 | Avx2 | Scalar } block goes here)
            // Returns Some(chunk_index_within_batch) on a hit.
            let found_chunk = match simd {
                // ... existing arms unchanged ...
            };
            // Convert chunk index to absolute spiral position for the scalar
            // confirmation walk below.
            found_chunk.map(|ci| {
                (batch_base + ci * GROUPS_PER_CHUNK) * 8
            })
        };

        if let Some(hit_spiral_idx) = found_spiral_idx {
            // Scalar confirmation walk (same for both GPU and CPU paths).
            // Walk position-by-position from the start of the hit chunk to find
            // the exact first match in spiral order.
            let chunk_base_group = hit_spiral_idx / 8 / GROUPS_PER_CHUNK * GROUPS_PER_CHUNK
                + batch_base;
            // ... (existing scalar walk unchanged) ...
            return Ok(Some((cx, cz)));
        }

        batch_start_group += GROUPS_PER_BATCH;
        if let Some(cb) = progress_cb {
            cb(batch_start_group * 8);
        }
    }
}
```

> **Note on the scalar confirmation walk:** The GPU path returns the absolute spiral
> index of the earliest match. The existing confirmation walk already recomputes
> the chunk's spiral state and walks group-by-group to find the exact position. No
> changes are needed there.

---

## 11. Multi-Rotation and GPU — Constraint

The existing "Search all 4 rotations" code uses `par_iter()` to run all rotations as
concurrent Rayon tasks. If GPU is enabled, each task would call
`GpuContext::search_chunk` concurrently, causing races on the wgpu queue.

**Resolution: GPU and "Search all 4 rotations" are mutually exclusive.**

When the user enables GPU, the all-rotations checkbox is automatically disabled and
greyed out (and vice versa). This is enforced in both the UI and the search dispatch.

Benefits:
- No Mutex needed around the queue.
- No per-rotation device allocation.
- Behaviour is easy to reason about and test.
- In practice, a user with a fast GPU rarely needs all-rotations since the GPU path
  is fast enough that running the correct rotation explicitly is fine.

---

## 12. UI Changes

### 12.1 New Message Variants

```rust
ToggleGpu(bool),
```

### 12.2 New `App` fields (recap)

```rust
gpu_ctx:  Option<Arc<GpuContext>>,
use_gpu:  bool,
```

### 12.3 New UI Rows

Replace the existing `all_rotations_row` and add a `gpu_row` below it:

```rust
// "Search all 4 rotations" — greyed out when GPU is active.
let all_rotations_row = row![
    checkbox(
        "Search all 4 rotations (if north direction is unknown)",
        self.search_all_rotations,
    )
    .on_toggle_maybe((!self.use_gpu).then_some(Message::ToggleAllRotations))
    .text_size(sc(13.0) as u16),
].align_items(Alignment::Center);

// "Use GPU" — greyed out when no adapter is found or all-rotations is active.
let gpu_available = self.gpu_ctx.is_some() && !self.search_all_rotations;
let gpu_label = if self.gpu_ctx.is_some() {
    if self.search_all_rotations {
        "Use GPU acceleration (unavailable with all-rotations search)"
    } else {
        "Use GPU acceleration (experimental)"
    }
} else {
    "GPU acceleration (no compatible adapter found)"
};
let gpu_row = row![
    checkbox(gpu_label, self.use_gpu)
        .on_toggle_maybe(gpu_available.then_some(Message::ToggleGpu))
        .text_size(sc(13.0) as u16),
].align_items(Alignment::Center);
```

Both rows are pushed into the column layout, `all_rotations_row` first, `gpu_row`
directly below it.

### 12.4 Message Handlers

```rust
Message::ToggleGpu(v) => {
    self.use_gpu = v;
    // Enabling GPU disables all-rotations search (mutually exclusive).
    if v { self.search_all_rotations = false; }
    Command::none()
}

Message::ToggleAllRotations(v) => {
    self.search_all_rotations = v;
    // Enabling all-rotations disables GPU (mutually exclusive).
    if v { self.use_gpu = false; }
    Command::none()
}
```

### 12.5 Pass GPU Context to Search Thread

In the `Message::Search` handler, before the `spawn_blocking` call:

```rust
let gpu_ctx = if self.use_gpu { self.gpu_ctx.clone() } else { None };
// Inside spawn_blocking closure, pass gpu_ctx as the last argument to run_search.
```

---

## 13. Spiral-Order Correctness

The GPU kernel uses `atomicMin` to record the **lowest candidate index** that passes
all block checks. The candidate array is filled by `fill_chunk_candidates` in the
same spiral order as the CPU path, so the lowest index is always the position closest
to the search centre. The scalar confirmation walk then pinpoints the exact coordinate.
This preserves the existing result contract exactly.

---

## 14. Cancellation

The GPU path iterates chunks inside the batch loop with the same
`cancel.load(Ordering::Relaxed)` check at the top of the loop. Cancellation latency
is at most one chunk dispatch, which takes well under 10 ms even for large block sets.
No changes to the cancellation logic are needed.

---

## 15. File / Module Layout

```
src/
  main.rs        ← minor additions only: gpu_ctx/use_gpu fields, ToggleGpu message,
                   gpu_row UI, gpu_ctx parameter in run_search, App::new() override.
                   No existing function body or SIMD code is changed.
  gpu.rs         ← new: GpuContext, GpuBlock, SearchUniforms, search_chunk,
                   fill_chunk_candidates, buffer management.
  search.wgsl    ← new: portable u32-pair 64-bit helpers + compute kernel.
```

---

## 16. Implementation Order (Suggested for AI)

1. Add `wgpu`, `pollster`, `bytemuck` to `Cargo.toml`.
2. Create `src/gpu.rs` with `GpuContext::new()` — adapter/device/queue init only,
   no pipeline or buffers yet. Add the `gpu_ctx` and `use_gpu` fields to `App`
   (both inert), override `App::new()` to call `GpuContext::new()`. Verify it
   compiles and the app launches normally.
3. Write `src/search.wgsl` with only the u32-pair helper functions and a `main`
   kernel that always writes `0xFFFF_FFFF` without doing any real work (no-op
   shader). Hook up the full pipeline and bind groups to verify the GPU path
   dispatches and reads back without crashing.
4. Implement `math_hash` in WGSL using the u32-pair helpers. Write a small Rust
   unit test (in `gpu.rs` under `#[cfg(test)]`) that runs a handful of known
   `(x, y, z)` inputs through both the Rust scalar `check_formation` and a
   CPU-side re-implementation of the WGSL `math_hash` logic, confirming identical
   results before any GPU hardware is involved.
5. Implement `xoroshiro_step` and the full kernel block loop with `atomicMin`.
   Extend the unit test to verify end-to-end `is_bedrock` equivalence for a known
   `(x, y, z, dlo, dhi)` triple.
6. Implement `GpuContext::search_chunk` with `fill_chunk_candidates`, interleaved
   buffer write, dispatch, and readback.
7. Integrate into `run_search` behind the `gpu_ctx` option. Add `ToggleGpu` to
   `Message` and the mutual-exclusion logic to both message handlers.
8. Add `gpu_row` to the UI layout.
9. Test end-to-end against a known seed/formation pair. GPU result must match the
   CPU result exactly.

---

## 17. Known Limitations and Edge Cases

| Issue | Mitigation |
|-------|-----------|
| WGSL lacks native `i64`/`u64` on many adapters | Default to portable u32-pair emulation throughout (§5). Do not require `SHADER_INT64`. |
| GPU readback adds ~1–2 ms per chunk | Negligible vs. compute time for 4,096 candidates; do not pipeline across chunks (complicates cancellation). |
| Integrated GPUs may be slower than AVX-512 | The toggle is opt-in; users can profile and decide. The status bar reports elapsed time for comparison. |
| `wgpu` async API inside `spawn_blocking` | Use `pollster::block_on` for GPU futures inside the blocking thread; safe because the thread is not the async executor thread. |
| Multi-rotation + GPU | Mutually exclusive in the UI (§11). GPU is disabled when all-rotations is active and vice versa. |
| `GpuContext::new()` blocks at startup | Called via `pollster::block_on` in `App::new()`, before the iced event loop starts. Startup delay is typically < 200 ms and only occurs when a GPU adapter is present. |
