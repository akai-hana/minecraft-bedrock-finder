/// gpu.rs - wgpu compute backend for the bedrock formation search.
///
/// Key design: candidate positions are computed **inside the shader** from each
/// thread's global invocation ID using the closed-form spiral formula.  There is
/// no candidate buffer, no sequential CPU position loop, and no multi-MB upload
/// per batch - the CPU cost per dispatch is a single 48-byte uniform write (or
/// push-constant set) and one 4-byte result readback.

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, pod_read_unaligned};
use wgpu::util::DeviceExt;

use crate::core::{GROUPS_PER_CHUNK};

// POD types

/// Must match `SearchUniforms` in search.wgsl (48 bytes, std140 padded).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SearchUniforms {
    pub dlo_lo:              u32,   // offset  0
    pub dlo_hi:              u32,   // offset  4
    pub dhi_lo:              u32,   // offset  8
    pub dhi_hi:              u32,   // offset 12
    /// Blocks per rotation in the block buffer (all rotations are the same
    /// length; the buffer stores `rotation_count` rotation-sets back-to-back).
    pub blocks_per_rotation: u32,   // offset 16
    pub candidate_count:     u32,   // offset 20
    pub batch_base_k_lo:     u32,   // offset 24 -- low 32 bits of the absolute spiral index of candidate 0
    pub start_x:             i32,   // offset 28
    pub start_z:             i32,   // offset 32
    /// Number of threads per dispatched "row" (= workgroups_x * GPU_WORKGROUP_SIZE).
    /// Used to turn the 2-D dispatch grid back into a single linear candidate index:
    /// tid = gid.y * dispatch_width + gid.x. See `search_batch` for why a 2-D
    /// grid is needed at all (the 1-D workgroup-count limit is 65 535).
    pub dispatch_width:      u32,   // offset 36
    /// How many rotation-sets are packed in the block buffer.  The kernel
    /// reports a hit when ANY rotation-set is fully satisfied.
    pub rotation_count:      u32,   // offset 40
    /// High 32 bits of the absolute spiral index of candidate 0. Together with
    /// `batch_base_k_lo` this forms a full 64-bit spiral index `k`, matching the
    /// shader's `U64`-based `spiral_pos` with no `u32`/`i32` ceiling.
    pub batch_base_k_hi:     u32,   // offset 44
}   // total: 48 bytes

/// Must match `BlockData` in search.wgsl (32 bytes).
///
/// Instead of storing the raw `bx` and `bz` coordinates, this struct carries
/// precomputed hash terms that the shader previously recomputed for every thread:
///
///   `bx_k    = (bx as u32).wrapping_mul(3_129_871)`
///   `bz_k    = (bz as i64).wrapping_mul(116_129_781)` stored as (lo, hi) u32
///
/// Inside the shader the per-position contribution (`ox * K_x` and `oz * K_z`)
/// is hoisted before the block loops, and the full x/z hash terms are obtained
/// with a single cheap add per block instead of a full 64-bit multiply.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuBlock {
    /// Precomputed `(bx as u32).wrapping_mul(3_129_871)`.
    /// Inside the shader: `x_term = i32_to_u64(i32(ox_k_u32 + b.bx_k))`.
    pub bx_k:              u32,   // offset  0
    /// Raw Y coordinate.
    pub by:                i32,   // offset  4
    /// Low 32 bits of `(bz as i64).wrapping_mul(116_129_781)`.
    pub bz_k_lo:           u32,   // offset  8
    /// High 32 bits of `(bz as i64).wrapping_mul(116_129_781)`.
    pub bz_k_hi:           u32,   // offset 12
    pub prob_threshold:    u32,   // offset 16
    pub should_be_bedrock: u32,   // offset 20
    pub _pad:              [u32; 2], // offset 24-28 (pad to 32 bytes)
}  // total: 32 bytes

// Buffer size constants

/// Maximum blocks per rotation (4 Y-layers x 16 cols x 16 rows).
const MAX_BLOCKS_PER_ROTATION: u64 = 1024;

/// Block buffer holds up to 4 rotations back-to-back.
const MAX_TOTAL_BLOCKS: u64 = 4 * MAX_BLOCKS_PER_ROTATION;

// Workgroup size
//
// This constant is the single source of truth for the GPU compute workgroup
// size on the Rust side (used by the dispatch arithmetic in `search_batch`).
// It is NOT passed into the shader at pipeline-creation time. That would
// normally use a WGSL `override` constant plus
// `PipelineCompilationOptions::constants`, but neither is available here:
// the naga WGSL frontend bundled with this project's pinned wgpu (0.19)
// doesn't parse `override` declarations at all, and the
// `compilation_options`/`cache` pipeline-descriptor fields don't exist until
// wgpu 22. So `search.wgsl` instead declares `WORKGROUP_SIZE_X` as a plain
// `const`, hardcoded to 256. That means this constant must currently stay
// 256 too; changing it here without also editing the shader's `const` will
// desync the two sides. (If you later upgrade wgpu, both the shader's
// `const` -> `override` and this comment can be revisited.)
//
// Values to benchmark: 64, 128, 256, 512.  The U64 emulation in the shader
// raises per-thread register pressure compared to a native-i64 kernel, which
// can silently cap occupancy at the driver level.  Smaller workgroup sizes free
// registers and can improve occupancy on register-pressure-sensitive GPUs (AMD
// RDNA, older GCN); larger sizes may improve IPC on warp-wide execution
// hardware (NVIDIA Ampere/Ada, Intel Arc).  Measure on your target hardware
// with the microbenchmark described in the Section 1 validation notes.
const GPU_WORKGROUP_SIZE: u32 = 256;

// GpuContext

pub struct GpuContext {
    device:             wgpu::Device,
    queue:              wgpu::Queue,
    pipeline:           wgpu::ComputePipeline,
    /// Pre-built bind group reused every batch.
    ///
    /// The bind group references the buffer *objects*, not their contents.
    /// Writing new data into `uniform_buf` (or setting push constants) each
    /// batch is perfectly valid; the bind group only needs to be rebuilt if
    /// the buffer handles themselves change, which they never do here.
    bind_group:         wgpu::BindGroup,
    /// Uniform buffer used to upload `SearchUniforms` when push constants are
    /// unavailable on the current backend. `None` when `use_push_constants` is
    /// true; in that case uniforms are set via `ComputePass::set_push_constants`
    /// and no uniform buffer is allocated.
    uniform_buf:        Option<wgpu::Buffer>,
    block_buf:          wgpu::Buffer,
    result_buf:         wgpu::Buffer,
    staging_buf:        wgpu::Buffer,
    /// 4-byte buffer permanently pre-filled with `0xFFFF_FFFF`, used as the
    /// copy source for the per-batch result-sentinel reset.  Allocated once in
    /// `new` with `COPY_SRC` usage; its contents never change.  Encoding the
    /// reset as `copy_buffer_to_buffer` into the command stream means the GPU
    /// DMA engine handles it without any CPU→GPU `write_buffer` call per batch.
    sentinel_buf:       wgpu::Buffer,
    /// True when `wgpu::Features::PUSH_CONSTANTS` was successfully requested
    /// from the adapter.  Affects the pipeline layout, the shader binding
    /// declaration (patched at creation time), and the per-batch uniform upload
    /// path in `search_batch`.
    ///
    /// Push constants skip a small buffer write plus its binding indirection
    /// per batch.  On Vulkan and DX12 this is typically a direct register write
    /// with no GPU memory traffic; on Metal it maps to argument buffers with
    /// similar characteristics.  The uniform-buffer path is kept as an automatic
    /// fallback for backends that do not expose the feature (WebGPU, some OpenGL
    /// compatibility layers).
    use_push_constants: bool,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("use_push_constants", &self.use_push_constants)
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await?;

        // Probe for push-constant support before device creation.  This feature
        // is available on Vulkan, DX12, and Metal but NOT on WebGPU or on some
        // older OpenGL/ES adapters; the fallback path (uniform buffer) is used
        // transparently when the feature is absent.
        let uniform_size = std::mem::size_of::<SearchUniforms>() as u32;
        let has_push_constants = adapter
            .features()
            .contains(wgpu::Features::PUSH_CONSTANTS);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             None,
                    required_features: if has_push_constants {
                        wgpu::Features::PUSH_CONSTANTS
                    } else {
                        wgpu::Features::empty()
                    },
                    required_limits: if has_push_constants {
                        wgpu::Limits {
                            // Request exactly the space we need; some drivers
                            // have a very low default ceiling (128 bytes is
                            // guaranteed by the Vulkan spec).
                            max_push_constant_size: uniform_size,
                            ..wgpu::Limits::default()
                        }
                    } else {
                        wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        // Patch the shader source when push constants are available: replace the
        // uniform-buffer binding declaration with a push-constant declaration.
        // The two forms are otherwise syntactically identical in WGSL; the rest
        // of the kernel code (`uniforms.foo`) is unchanged.
        //
        // The `@group(0) @binding(0)` prefix is included in the search string so
        // this replacement is uniquely anchored and won't accidentally match any
        // other `var<uniform>` declaration added in the future.
        const UNIFORM_BINDING_DECL: &str =
            "@group(0) @binding(0) var<uniform>             uniforms: SearchUniforms;";
        const PUSH_CONST_DECL: &str =
            "var<push_constant> uniforms: SearchUniforms;";

        let shader_src: std::borrow::Cow<'static, str> = if has_push_constants {
            let patched = include_str!("search.wgsl")
                .replace(UNIFORM_BINDING_DECL, PUSH_CONST_DECL);
            debug_assert!(
                patched.contains(PUSH_CONST_DECL),
                "push-constant shader patch failed: uniform binding declaration not found; \
                 check that UNIFORM_BINDING_DECL in gpu.rs exactly matches the line in search.wgsl"
            );
            patched.into()
        } else {
            include_str!("search.wgsl").into()
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("bedrock_search"),
            source: wgpu::ShaderSource::Wgsl(shader_src),
        });

        // Bind group layout.
        //
        // When push constants are active, binding 0 (the uniform buffer) is
        // absent from the layout; it is no longer a buffer binding at all.
        // Bindings 1 (blocks, read-only storage) and 2 (result, read-write
        // storage/atomic) are the same in both cases.
        let storage_block_entry = wgpu::BindGroupLayoutEntry {
            binding:    1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };
        let storage_result_entry = wgpu::BindGroupLayoutEntry {
            binding:    2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };

        let bgl = if has_push_constants {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   None,
                entries: &[storage_block_entry, storage_result_entry],
            })
        } else {
            let uniform_entry = wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            };
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   None,
                entries: &[uniform_entry, storage_block_entry, storage_result_entry],
            })
        };

        // Pipeline layout: push-constant range is declared here when the feature
        // is active so the driver can allocate the right register space.
        let push_constant_ranges: Vec<wgpu::PushConstantRange> = if has_push_constants {
            vec![wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range:  0..uniform_size,
            }]
        } else {
            vec![]
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("bedrock_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &push_constant_ranges,
        });

        // No workgroup-size override here: `search.wgsl` declares
        // `WORKGROUP_SIZE_X` as a plain `const` (see the comment by
        // `GPU_WORKGROUP_SIZE` above for why), so its value is fixed at
        // shader-compile time and there's nothing to inject at
        // pipeline-creation time.
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("bedrock_pipeline"),
            layout:      Some(&pipeline_layout),
            module:      &shader,
            entry_point: "main",
        });

        let block_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("blocks"),
            size:               MAX_TOTAL_BLOCKS * std::mem::size_of::<GpuBlock>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("result"),
            size:               4,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("staging"),
            size:               4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Uniform buffer: only allocated when push constants are unavailable.
        let uniform_buf: Option<wgpu::Buffer> = if has_push_constants {
            None
        } else {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("uniforms"),
                size:               std::mem::size_of::<SearchUniforms>() as u64,
                usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        };

        // Sentinel buffer: 4 bytes permanently set to 0xFFFF_FFFF.
        //
        // Each batch resets `result_buf` to this value before the compute pass
        // so that the shader's `atomicMin` starts from the "no hit" state.
        // Encoding the reset as `copy_buffer_to_buffer` (GPU-side DMA) into the
        // command stream avoids one `queue.write_buffer` (CPU→GPU transfer) per
        // batch compared to the old approach.  The contents of this buffer are
        // written once here and never touched again.
        let sentinel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("sentinel"),
            contents: &0xFFFF_FFFFu32.to_le_bytes(),
            usage:    wgpu::BufferUsages::COPY_SRC,
        });

        // Build the bind group once.  The buffer *objects* are fixed for the
        // lifetime of this GpuContext; only their contents change between batches.
        // Rebuilding the bind group on every `search_batch` call would incur
        // driver-side descriptor-set allocation and layout validation for no reason.
        let bind_group = if has_push_constants {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("bedrock_bg"),
                layout:  &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: block_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: result_buf.as_entire_binding() },
                ],
            })
        } else {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("bedrock_bg"),
                layout:  &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: uniform_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: block_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: result_buf.as_entire_binding() },
                ],
            })
        };
        // `bgl` is no longer needed after the bind group and pipeline layout are
        // built.  Let it drop here; the bind group holds an internal reference
        // that keeps the layout alive as long as `bind_group` is live.

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group,
            uniform_buf,
            block_buf,
            result_buf,
            staging_buf,
            sentinel_buf,
            use_push_constants: has_push_constants,
        })
    }

    /// Write the block data to the GPU buffer.
    ///
    /// **Call this exactly once per `run_search` invocation**, before entering
    /// the super-batch loop.  Block data is constant for the lifetime of a
    /// single search; re-uploading it on every `search_batch` call wastes GPU
    /// bus bandwidth proportional to the number of super-batches searched.
    ///
    /// `blocks` must contain ALL rotations concatenated back-to-back
    /// (rotation 0 first, rotation 1 next, ...), matching the layout expected by
    /// the shader and `search_batch`.
    pub fn write_blocks(&self, blocks: &[GpuBlock]) {
        self.queue.write_buffer(&self.block_buf, 0, cast_slice(blocks));
    }

    /// Search `num_chunks` consecutive spiral chunks in a single GPU dispatch.
    ///
    /// CPU cost per call:
    ///   - 48 bytes written via one `queue.write_buffer` (uniform-buffer path)
    ///     or one `ComputePass::set_push_constants` call (push-constant path)
    ///   - 4 bytes read back via a staging buffer
    ///
    /// The result-sentinel reset (writing 0xFFFF_FFFF to the result buffer) is
    /// encoded into the command stream via `encoder.fill_buffer` rather than a
    /// second `queue.write_buffer`, so only a single host->GPU write is needed
    /// per batch for the uniform/push-constant payload.
    ///
    /// Block data must have been uploaded by a preceding `write_blocks` call;
    /// it is not re-written here (blocks are constant within a `run_search`
    /// call; uploading them once outside the loop saves significant bus
    /// bandwidth for long searches).
    ///
    /// All candidate (x, z) positions are computed on the GPU from the global
    /// thread ID using the closed-form spiral formula; there is no candidate
    /// buffer and no multi-MB upload.
    ///
    /// `blocks` and `rotation_count` are still required here to derive the
    /// `blocks_per_rotation` value written into the uniform buffer; no buffer
    /// upload is performed for them.
    ///
    /// Returns `Some(chunk_index_within_batch)` on a hit, `None` otherwise.
    pub fn search_batch(
        &self,
        batch_base_group: i64,    // first group index of this batch
        num_chunks:       usize,  // number of chunks to process
        start_x:          i32,
        start_z:          i32,
        dlo:              i64,
        dhi:              i64,
        blocks:           &[GpuBlock],  // ALL rotations, concatenated
        rotation_count:   u32,          // number of rotation-sets in `blocks`
    ) -> Option<i64> {
        debug_assert!(rotation_count >= 1, "rotation_count must be at least 1");
        debug_assert_eq!(
            blocks.len() as u32 % rotation_count, 0,
            "blocks.len() must be a multiple of rotation_count",
        );
        let blocks_per_rotation = blocks.len() as u32 / rotation_count;
        let positions_per_chunk = GROUPS_PER_CHUNK as u32 * 8;
        let candidate_count     = num_chunks as u32 * positions_per_chunk;

        // batch_base_k is the absolute spiral index (k) of candidate 0, passed
        // to the shader as two u32 words (lo/hi). The shader's `spiral_pos`
        // does all shell-boundary arithmetic in `U64`, so there is no effective
        // ceiling beyond `ox`/`oz` fitting in `i32` (shell `l <= i32::MAX`,
        // unreachable in practice before `4*l*l` overflows `i64`).
        //
        // `batch_base_group` is always non-negative, so `* 8` fits in `u64`;
        // `checked_mul` is a defensive sanity check, not a real limit.
        let batch_base_k: u64 = u64::try_from(batch_base_group)
            .expect("GPU search: batch_base_group is negative")
            .checked_mul(8)
            .expect("GPU search: batch_base_group * 8 overflowed u64 (spiral index out of range)");

        let batch_base_k_lo = (batch_base_k & 0xFFFF_FFFF) as u32;
        let batch_base_k_hi = (batch_base_k >> 32) as u32;

        // Dispatch grid
        // WebGPU/wgpu caps `max_compute_workgroups_per_dimension` at 65 535.
        // With the default super-batch size (2048 chunks x 1024 groups/chunk x
        // 8 positions/group = 16 777 216 candidates), a 1-D dispatch would need
        // 65 536 workgroups of size 256 - exactly one *more* than the limit,
        // which makes `dispatch_workgroups` fail validation (this was the
        // crash reported when GPU search was enabled).
        //
        // Fix: spread the workgroups over a 2-D grid (x, y), both dimensions
        // kept <= the limit, and reconstruct a linear candidate index in the
        // shader as `tid = gid.y * dispatch_width + gid.x`. For the common
        // case (total_workgroups <= 65 535) this degenerates to the original
        // 1-D dispatch (dispatch_y == 1).
        const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

        let total_workgroups = (candidate_count + GPU_WORKGROUP_SIZE - 1) / GPU_WORKGROUP_SIZE;
        let dispatch_x = if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
            total_workgroups.max(1)
        } else {
            // Roughly square the grid so we don't dispatch far more threads
            // than candidates (e.g. 65536 workgroups -> 256 x 256).
            ((total_workgroups as f64).sqrt().ceil() as u32)
                .clamp(1, MAX_WORKGROUPS_PER_DIM)
        };
        let dispatch_y = ((total_workgroups + dispatch_x - 1) / dispatch_x)
            .clamp(1, MAX_WORKGROUPS_PER_DIM);
        let dispatch_width = dispatch_x * GPU_WORKGROUP_SIZE;

        // Build the uniform struct (used by both the uniform-buffer path and
        // the push-constant path; the bytes are the same either way).
        let uniforms = SearchUniforms {
            dlo_lo:              (dlo as u64 & 0xFFFF_FFFF) as u32,
            dlo_hi:              ((dlo as u64) >> 32) as u32,
            dhi_lo:              (dhi as u64 & 0xFFFF_FFFF) as u32,
            dhi_hi:              ((dhi as u64) >> 32) as u32,
            blocks_per_rotation,
            candidate_count,
            batch_base_k_lo,
            start_x,
            start_z,
            dispatch_width,
            rotation_count,
            batch_base_k_hi,
        };

        // 1. Upload per-batch parameters.
        //
        // Push-constant path: `set_push_constants` is called inside the compute
        // pass below (it must be called after `set_pipeline`), so nothing to do
        // here on the CPU side before encoding.
        //
        // Uniform-buffer path: write the 48-byte uniform blob in a single
        // `write_buffer` call.  The result-sentinel reset is handled by a
        // GPU-side `copy_buffer_to_buffer` encoded into the command stream
        // (step 2), so only one host→GPU write is needed per batch.
        if !self.use_push_constants {
            self.queue.write_buffer(
                self.uniform_buf.as_ref().expect("uniform_buf must be Some when not using push constants"),
                0,
                bytes_of(&uniforms),
            );
        }

        // 2. Encode, dispatch, copy result to staging.
        //    The bind group is already built and cached in `self.bind_group` -
        //    no allocation or validation happens here.
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("search_batch") },
        );

        // Reset result sentinel via GPU-side DMA copy from `sentinel_buf`.
        // `sentinel_buf` is a 4-byte buffer permanently pre-filled with
        // 0xFFFF_FFFF (the "no hit" value).  Encoding this as
        // `copy_buffer_to_buffer` instead of a CPU-side `queue.write_buffer`
        // means the reset travels through the same command stream as the
        // dispatch and the result copy, so the GPU command buffer is fully
        // self-contained: reset → dispatch → copy result to staging.
        // This eliminates one CPU→GPU `write_buffer` call per batch.
        enc.copy_buffer_to_buffer(&self.sentinel_buf, 0, &self.result_buf, 0, 4);

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("search_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);

            // Push-constant path: set uniforms directly into shader registers.
            // This must be called after `set_pipeline` (the pipeline layout
            // defines which push-constant ranges are valid).
            if self.use_push_constants {
                pass.set_push_constants(0, bytes_of(&uniforms));
            }

            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        enc.copy_buffer_to_buffer(&self.result_buf, 0, &self.staging_buf, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        // 3. Single blocking readback - one poll per batch, not per chunk.
        //
        // A yield-loop over `Maintain::Poll` is used instead of `Maintain::Wait`
        // to avoid pinning the calling thread for the full GPU execution window.
        // On Vulkan and some DX12 backends `Maintain::Wait` is a busy-spin;
        // yielding on each iteration instead surrenders the time-slice to other
        // Rayon workers while the GPU is still executing.
        let slice = self.staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        loop {
            if matches!(self.device.poll(wgpu::Maintain::Poll), wgpu::MaintainResult::SubmissionQueueEmpty) {
                break;
            }
            std::thread::yield_now();
        }

        let raw: u32 = {
            let view = slice.get_mapped_range();
            pod_read_unaligned(&view[..4])
        };
        self.staging_buf.unmap();

        if raw == 0xFFFF_FFFF {
            None
        } else {
            // raw is the earliest (atomicMin) candidate index within the batch.
            // Convert to chunk index within this batch.
            Some((raw / positions_per_chunk) as i64)
        }
    }
}
