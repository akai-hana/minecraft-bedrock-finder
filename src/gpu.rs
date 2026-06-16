/// gpu.rs - wgpu compute backend for the bedrock formation search.
///
/// Key design: candidate positions are computed **inside the shader** from each
/// thread's global invocation ID using the closed-form spiral formula.  There is
/// no candidate buffer, no sequential CPU position loop, and no multi-MB upload
/// per batch - the CPU cost per dispatch is a single 48-byte uniform write and
/// one 4-byte result readback.

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, pod_read_unaligned};

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
    /// Number of threads per dispatched "row" (= workgroups_x * 256). Used to
    /// turn the 2-D dispatch grid back into a single linear candidate index:
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
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuBlock {
    pub bx:                i32,
    pub by:                i32,
    pub bz:                i32,
    pub prob_threshold:    u32,
    pub should_be_bedrock: u32,
    pub _pad:              [u32; 3],
}

// Buffer size constants

/// Maximum blocks per rotation (4 Y-layers x 16 cols x 16 rows).
const MAX_BLOCKS_PER_ROTATION: u64 = 1024;

/// Block buffer holds up to 4 rotations back-to-back.
const MAX_TOTAL_BLOCKS: u64 = 4 * MAX_BLOCKS_PER_ROTATION;

// GpuContext

pub struct GpuContext {
    device:      wgpu::Device,
    queue:       wgpu::Queue,
    pipeline:    wgpu::ComputePipeline,
    bgl:         wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    block_buf:   wgpu::Buffer,
    result_buf:  wgpu::Buffer,
    staging_buf: wgpu::Buffer,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext").finish_non_exhaustive()
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             None,
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("bedrock_search"),
            source: wgpu::ShaderSource::Wgsl(include_str!("search.wgsl").into()),
        });

        // 3 bindings: uniform | ro-storage (blocks) | rw-storage (result)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },

                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },

                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },

                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("bedrock_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("bedrock_pipeline"),
            layout:      Some(&pipeline_layout),
            module:      &shader,
            entry_point: "main",
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("uniforms"),
            size:               std::mem::size_of::<SearchUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        Some(Self { device, queue, pipeline, bgl, uniform_buf, block_buf, result_buf, staging_buf })
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
    ///   - 48 bytes written to the uniform buffer
    ///   - 4 bytes read back via a staging buffer
    ///
/// Block data must have been uploaded by a preceding `write_blocks` call
    /// / it is not re-written here (blocks are constant within a `run_search`
    /// call; uploading them once outside the loop saves significant bus
    /// bandwidth for long searches).
    ///
    /// All candidate (x, z) positions are computed on the GPU from the global
    /// thread ID using the closed-form spiral formula - there is no candidate
    /// buffer and no multi-MB upload.
    ///
    /// `blocks` and `rotation_count` are still required here to derive the
    /// `blocks_per_rotation` value written into the uniform buffer; no
    /// buffer upload is performed.
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
        const WORKGROUP_SIZE:        u32 = 256;
        const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

        let total_workgroups = (candidate_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
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
        let dispatch_width = dispatch_x * WORKGROUP_SIZE;

        // 1. Write uniforms (48 bytes; negligible cost).
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
        self.queue.write_buffer(&self.uniform_buf, 0, bytes_of(&uniforms));

        // 2. Reset result sentinel.
        let sentinel: u32 = 0xFFFF_FFFF;
        self.queue.write_buffer(&self.result_buf, 0, bytes_of(&sentinel));

        // 3. Bind group.
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.block_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.result_buf.as_entire_binding() },
            ],
        });

        // 4. Encode, dispatch, copy result to staging.
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("search_batch") },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("search_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        enc.copy_buffer_to_buffer(&self.result_buf, 0, &self.staging_buf, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        // 5. Single blocking readback - one poll per batch, not per chunk.
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
