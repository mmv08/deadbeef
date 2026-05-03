use super::{
    adapter::select_adapter,
    dispatch::DispatchLayout,
    input::{unpack_words, GpuInput, GpuResult},
    Backend, Result,
};
use bytemuck::Zeroable as _;
use std::{borrow::Cow, mem, sync::mpsc};

pub(super) const SHADER: &str = include_str!("../gpu.wgsl");

pub(super) struct Miner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    input_buffer: wgpu::Buffer,
    result_buffer: wgpu::Buffer,
    read_buffer: wgpu::Buffer,
}

pub(super) struct MinerGuard {
    miner: Option<Miner>,
    backend: Backend,
}

impl MinerGuard {
    pub(super) fn new(miner: Miner, backend: Backend) -> Self {
        Self {
            miner: Some(miner),
            backend,
        }
    }
}

impl std::ops::Deref for MinerGuard {
    type Target = Miner;

    fn deref(&self) -> &Self::Target {
        self.miner
            .as_ref()
            .expect("miner should exist until its guard drops")
    }
}

impl Drop for MinerGuard {
    fn drop(&mut self) {
        if self.backend == Backend::Gl {
            // Mesa's GL-over-D3D12 path on WSL has been observed crashing during
            // device teardown. The CLI exits immediately after mining, so
            // leaking only this GL device avoids losing the mined result during
            // shutdown cleanup. If the driver or wgpu teardown path changes,
            // verify GL mining and remove this workaround.
            if let Some(miner) = self.miner.take() {
                mem::forget(miner);
            }
        }
    }
}

impl Miner {
    // Open a session with the GPU and prepare the resources the miner reuses
    // across batches.
    //
    // wgpu vocabulary used below:
    // - an *adapter* is a specific physical GPU + driver combination wgpu can
    //   open (e.g. "Apple M3 Max via Metal");
    // - a *device* is our open handle to that adapter, scoped to this process;
    // - a *queue* is where we submit commands (each batch is one submission);
    // - a *shader module* is the compiled WGSL program;
    // - a *compute pipeline* binds a shader entry point to fixed pipeline
    //   state (workgroup size, layout);
    // - *buffers* are GPU-visible memory regions; *bind groups* tell the
    //   shader which buffers to read and write.
    pub(super) fn new(
        backend: Backend,
        adapter_index: Option<usize>,
        allow_software_adapter: bool,
    ) -> Result<Self> {
        let adapter = select_adapter(backend, adapter_index, allow_software_adapter)?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("deadbeef GPU miner"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deadbeef Keccak miner"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("deadbeef GPU miner pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deadbeef GPU input"),
            size: mem::size_of::<GpuInput>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deadbeef GPU result"),
            size: mem::size_of::<GpuResult>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deadbeef GPU result readback"),
            size: mem::size_of::<GpuResult>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deadbeef GPU miner bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            input_buffer,
            result_buffer,
            read_buffer,
        })
    }

    // Run one batch of candidates on the GPU. Returns `Some(nonce)` if the
    // shader claimed a match in this batch, or `None` if the batch finished
    // with no match.
    //
    // wgpu vocabulary used below:
    // - a *command encoder* records GPU commands without running them;
    // - a *compute pass* is a phase inside that recording where the GPU runs
    //   our shader for one dispatch;
    // - a *bind group* hands the shader the buffers we want it to read/write;
    // - a *buffer slice* + `map_async` is how we ask the GPU to make a
    //   buffer's bytes visible to the CPU once submitted work has finished.
    //
    // Each batch is synchronous from the caller's perspective: write the
    // input, clear the result slot, dispatch the shader, copy the result into
    // a CPU-mappable buffer, and wait for readback. The heavy work stays on
    // the GPU; the synchronous readback is one small `GpuResult` (36 bytes)
    // per batch.
    pub(super) fn run_batch(
        &self,
        input: &GpuInput,
        dispatch: &DispatchLayout,
    ) -> Result<Option<[u8; 32]>> {
        self.queue
            .write_buffer(&self.input_buffer, 0, bytemuck::bytes_of(input));
        self.queue.write_buffer(
            &self.result_buffer,
            0,
            bytemuck::bytes_of(&GpuResult::zeroed()),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deadbeef GPU miner command encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("deadbeef GPU miner pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(dispatch.workgroup_columns, dispatch.workgroup_rows, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.result_buffer,
            0,
            &self.read_buffer,
            0,
            mem::size_of::<GpuResult>() as wgpu::BufferAddress,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.read_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(Box::new(err)),
            Err(err) => return Err(Box::new(err)),
        }

        let data = slice.get_mapped_range();
        let result = *bytemuck::from_bytes::<GpuResult>(&data);
        drop(data);
        self.read_buffer.unmap();

        if result.found == 0 {
            return Ok(None);
        }
        Ok(Some(unpack_words(result.nonce)))
    }
}
