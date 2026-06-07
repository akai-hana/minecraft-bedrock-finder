# GPU Computation Plan — Bedrock Formation Finder

## Overview

This document is a step-by-step implementation plan to add optional GPU-accelerated
searching to `bedrockformation`. The search hot path (computing `math_hash` →
xoroshiro128++ → threshold for every candidate position) is embarrassingly parallel
and maps directly to GPU compute. The existing CPU path (AVX-512 / AVX2 / scalar via
Rayon) remains unchanged and is the default; GPU is an explicit opt-in toggle.

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

### 4.1 New `ComputeBackend` Enum

Replace (or extend) `SimdLevel` to account for GPU:

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
enum ComputeBackend {
    Gpu,                // wgpu compute shader
    Avx512,             // existing 8-wide SIMD
    Avx2,               // existing 4-wide SIMD
    Scalar,             // existing scalar fallback
}
```

Detection order at startup: GPU → AVX-512 → AVX2 → Scalar. If the user disables
GPU via the toggle, skip the GPU branch and proceed to AVX-512.

### 4.2 New `GpuContext` Struct

Create a module `mod gpu` containing:

```rust
pub struct GpuContext {
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    // Bind group layout cached at creation time
    bgl:      wgpu::BindGroupLayout,
}
```

`GpuContext::new() -> Option<GpuContext>` — returns `None` when no GPU adapter is
found (automatic CPU fallback).

### 4.3 App State Changes

Add two fields to `App`:

```rust
gpu_ctx:       Option<Arc<GpuContext>>,  // None = GPU not available
use_gpu:       bool,                     // controlled by the UI toggle (default: false)
```

Initialise `gpu_ctx` in `App::new()` by calling `GpuContext::new()` via
`pollster::block_on`.

---

## 5. WGSL Compute Shader

Create `src/search.wgsl`. The shader receives:

| Binding | Type | Contents |
|---------|------|----------|
| 0 | uniform | `SearchUniforms` (dlo, dhi, block count) |
| 1 | read-only storage | `BlockData[]` (bx, by, bz, prob_threshold, should_be_bedrock) |
| 2 | read-only storage | `CandidatePos[]` (ox, oz) for the chunk |
| 3 | read-write storage | `u32` result index (init to 0xFFFFFFFF = not found) |

```wgsl
struct SearchUniforms {
    dlo:         i64,
    dhi:         i64,
    block_count: u32,
    candidate_count: u32,
}

struct BlockData {
    bx: i32, by: i32, bz: i32,
    prob_threshold:    u32,   // (probability * 2^24) as u32
    should_be_bedrock: u32,   // 0 = no, 1 = yes
}

struct CandidatePos { ox: i32, oz: i32 }

@group(0) @binding(0) var<uniform>        uniforms:    SearchUniforms;
@group(0) @binding(1) var<storage, read>  blocks:      array<BlockData>;
@group(0) @binding(2) var<storage, read>  candidates:  array<CandidatePos>;
@group(0) @binding(3) var<storage, read_write> result: atomic<u32>;

fn math_hash(x: i32, y: i32, z: i32) -> i64 {
    // Replicate the Rust scalar implementation exactly.
    // WGSL does not have native i64; use a pair of u32 (lo, hi) with
    // manual carry arithmetic, or use the i64 extension if the adapter
    // supports it (check via Features::SHADER_INT64).
    // See implementation note below.
}

fn xoroshiro_step(s0: u64, s1: u64) -> u64 {
    // result = (s0 + s1).rotate_left(17) + s0
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.candidate_count { return; }

    let ox = candidates[idx].ox;
    let oz = candidates[idx].oz;

    var passes = true;
    for (var i = 0u; i < uniforms.block_count && passes; i++) {
        let b   = blocks[i];
        let hash = math_hash(ox + b.bx, b.by, oz + b.bz);
        let s0   = u64(hash ^ uniforms.dlo);
        let s1   = u64(uniforms.dhi);
        let res  = xoroshiro_step(s0, s1);
        let top24 = u32(res >> 40u);
        let is_bedrock = top24 < b.prob_threshold;
        passes = is_bedrock == bool(b.should_be_bedrock);
    }

    if passes {
        // Record the lowest-index match (earliest in spiral order).
        atomicMin(&result, idx);
    }
}
```

> **Implementation note on i64 in WGSL:**  
> The `math_hash` function requires signed 64-bit integer arithmetic
> (`x.wrapping_mul(3_129_871) as i64`, etc.). WGSL gains native `i64`/`u64`
> via the `shader-int64` extension; check `adapter.features()` for
> `wgpu::Features::SHADER_INT64`. If unavailable, implement a `i64_mul`
> helper using two `u32` words and manual carry — this is mechanical but
> verbose; an AI assistant can generate it from the Rust scalar code.

---

## 6. GPU Buffer Layout

### Uniform buffer (`SearchUniforms`) — 32 bytes, updated once per chunk

```rust
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct SearchUniforms {
    dlo:             i64,
    dhi:             i64,
    block_count:     u32,
    candidate_count: u32,
    _pad:            [u32; 2],   // align to 16 bytes
}
```

### Block storage buffer — written once per search (blocks don't change mid-search)

```rust
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct GpuBlock {
    bx: i32, by: i32, bz: i32,
    prob_threshold:    u32,
    should_be_bedrock: u32,
    _pad:              [u32; 3],
}
```

### Candidate position buffer — rewritten every chunk

Size: `CHUNK_SIZE * 8` bytes (two `i32` per candidate).

### Result buffer — 4 bytes

Initialised to `0xFFFF_FFFFu32` (= not found) before each dispatch via a write of
the staging buffer. Read back via a `MAP_READ` staging buffer after dispatch.

---

## 7. GPU Search Function

Add to `gpu` module:

```rust
impl GpuContext {
    /// Search one chunk on the GPU.
    /// Returns Some(index) if a match is found, None otherwise.
    pub fn search_chunk(
        &self,
        chunk_x:  &[i32],
        chunk_z:  &[i32],
        dlo:      i64,
        dhi:      i64,
        blocks:   &[GpuBlock],   // pre-converted once per search
    ) -> Option<usize> {
        // 1. Write uniform buffer (dlo, dhi, block_count, candidate_count)
        // 2. Write candidate buffer (interleaved x/z pairs)
        // 3. Write result buffer (0xFFFF_FFFF)
        // 4. Create bind group
        // 5. Encode: set_pipeline, set_bind_group, dispatch_workgroups((N+255)/256, 1, 1)
        // 6. Copy result buffer -> staging buffer
        // 7. queue.submit(encoder.finish())
        // 8. Map staging buffer, read u32 result
        // 9. Return Some(result as usize) if result != 0xFFFF_FFFF
    }
}
```

Buffer reuse strategy: allocate the candidate buffer once at the maximum size
(`CHUNK_SIZE * 8` bytes) and reuse it across chunk iterations with `queue.write_buffer`.
This avoids GPU allocation overhead in the hot loop.

---

## 8. Integration into `run_search`

In `run_search`, after converting `Blocks` to `Blocks` (SoA), also convert to
`Vec<GpuBlock>` if a GPU context is available:

```rust
let gpu_blocks: Option<Vec<GpuBlock>> = gpu_ctx.as_ref().map(|_| {
    (0..blocks.len()).map(|i| GpuBlock {
        bx: blocks.x[i], by: blocks.y[i], bz: blocks.z[i],
        prob_threshold:    blocks.prob_threshold[i] as u32,
        should_be_bedrock: blocks.should_be_bedrock[i] as u32,
        _pad: [0; 3],
    }).collect()
});
```

Inside the chunk loop, dispatch:

```rust
let found_idx = match (gpu_ctx.as_ref(), gpu_blocks.as_deref()) {
    (Some(ctx), Some(gblocks)) => {
        ctx.search_chunk(&chunk_x, &chunk_z, dlo, dhi, gblocks)
    }
    _ => search_chunk(&chunk_x, &chunk_z, dlo, dhi, &blocks, simd),
};
```

---

## 9. UI Changes

### 9.1 New Message Variant

```rust
ToggleGpu(bool),
```

### 9.2 New UI Row

Add below the existing "Search all 4 rotations" checkbox:

```rust
let gpu_row = {
    let available = self.gpu_ctx.is_some();
    let label = if available {
        "Use GPU acceleration (experimental)"
    } else {
        "GPU acceleration (no compatible adapter found)"
    };
    row![
        checkbox(label, self.use_gpu)
            .on_toggle_maybe(available.then_some(Message::ToggleGpu))
            .text_size(sc(13.0) as u16),
    ].align_items(Alignment::Center)
};
```

The checkbox is greyed-out and non-interactive when no GPU adapter was detected.

### 9.3 Message Handler

```rust
Message::ToggleGpu(v) => {
    self.use_gpu = v;
    Command::none()
}
```

### 9.4 Pass GPU Context to Search Thread

In the `Message::Search` handler, clone the `Arc<GpuContext>` and pass it into
the `spawn_blocking` closure:

```rust
let gpu_ctx = if self.use_gpu { self.gpu_ctx.clone() } else { None };
// ... inside spawn_blocking:
// pass gpu_ctx to run_search
```

---

## 10. Spiral-Order Correctness

The GPU kernel uses `atomicMin` to record the **lowest candidate index** that
passes all block checks. Because the candidate array is filled in spiral order
(same as the CPU path), the lowest index corresponds to the closest position to
the search centre. This preserves the existing result contract exactly.

---

## 11. Cancellation

The GPU path operates chunk-by-chunk, just like the CPU path. The
`cancel.load(Ordering::Relaxed)` check at the top of the chunk loop fires
before each GPU dispatch, so cancellation latency is at most one chunk
(`CHUNK_SIZE = 32_768` positions). No changes to cancellation logic are needed.

---

## 12. File / Module Layout

```
src/
  main.rs        ← existing; minor additions (gpu_ctx field, ToggleGpu message, gpu_row UI)
  gpu.rs         ← new; GpuContext, GpuBlock, search_chunk, buffer management
  search.wgsl    ← new; compute shader
```

---

## 13. Implementation Order (Suggested for AI)

1. Add `wgpu`, `pollster`, `bytemuck` to `Cargo.toml`.
2. Create `src/gpu.rs` with `GpuContext::new()` — adapter/device/queue init only,
   no pipeline yet. Verify it compiles.
3. Write `src/search.wgsl` — start with a shader that always writes `0xFFFF_FFFF`
   (no-op). Hook it up with a pipeline and empty bind groups to verify the GPU path
   runs without crashing.
4. Implement `math_hash` and `xoroshiro_step` in WGSL. Write a small CPU-side test
   that checks a handful of known `(x, y, z, dlo, dhi)` inputs against the Rust
   scalar `is_bedrock` to confirm numerical equivalence.
5. Implement the full shader kernel with the block loop and `atomicMin`.
6. Implement `GpuContext::search_chunk` with buffer writes, dispatch, and readback.
7. Integrate into `run_search` behind the `gpu_ctx` option.
8. Add `ToggleGpu` to the `App` struct and UI.
9. Test end-to-end against a known seed/formation pair; results must match the CPU path.

---

## 14. Known Limitations and Edge Cases

| Issue | Mitigation |
|-------|-----------|
| WGSL lacks `i64` on older adapters | Check `SHADER_INT64` feature; fall back to u32-pair emulation if absent |
| GPU readback adds ~1–2 ms of round-trip latency | Negligible vs. compute time for `CHUNK_SIZE = 32_768`; do not pipeline across chunks (complicates cancellation) |
| Integrated GPUs may be slower than AVX-512 | Profile; expose the choice explicitly to the user via the toggle |
| `wgpu` async API inside `spawn_blocking` | Use `pollster::block_on` for GPU futures inside the blocking thread; safe because the thread is not the async executor thread |
| Multi-rotation parallel search | Each rotation's `run_search` call independently acquires the shared `GpuContext`; since dispatches are serial within each call and `wgpu::Queue::submit` is `&self`, concurrent calls from different Rayon threads need the queue behind a `Mutex` or each rotation gets its own device |
