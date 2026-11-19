// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;

// What one dispatch does, for the timing table: the kernel name and the bytes that one run must read and write at least.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Work {
    // The kernel kind, for example "conv3x3". Every dispatch of the same shader has the same name, so the table adds them up.
    pub(crate) kernel: &'static str,
    // Every tensor read once and written once. A kernel that reads a value twice still counts it once.
    pub(crate) bytes: u64,
}

impl Work {
    // The work of a kernel that reads and writes `values` f32 values in total.
    pub(crate) fn f32_values(kernel: &'static str, values: u64) -> Self {
        // 4 bytes per f32.
        Self { kernel, bytes: values * 4 }
    }
}

// One kernel with its buffers connected, ready to run. Every kernel in the engine is one of these.
// A clone is a second handle to the same compiled shader and buffers, so one submit can list kernels from several owners.
#[derive(Clone)]
pub struct Dispatch {
    // The compiled shader.
    pipeline: wgpu::ComputePipeline,
    // The buffers connected to its @binding slots.
    bind_group: wgpu::BindGroup,
    // How many workgroups to start along x, y and z.
    workgroups: [u32; 3],
    // The kernel name and its bytes, for the timing table.
    work: Work,
}

impl Dispatch {
    // Compile `shader` and connect `buffers` to group 0: the first buffer to @binding(0), the next to @binding(1), and so on.
    // `work` names the kernel and its bytes for the timing table.
    pub(crate) fn new(gpu: &Gpu, label: &str, shader: &str, buffers: &[&wgpu::Buffer], workgroups: [u32; 3], work: Work) -> Result<Self, EngineError> {
        // Compile the WGSL text.
        let pipeline = compute_pipeline(gpu, label, shader);
        // Connect the buffers.
        Self::with_pipeline(gpu, label, &pipeline, buffers, workgroups, work)
    }

    // Connect `buffers` to a kernel that is already compiled. Attention uses this to compile once and bind once per chunk.
    pub(crate) fn with_pipeline(gpu: &Gpu, label: &str, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer], workgroups: [u32; 3], work: Work) -> Result<Self, EngineError> {
        // The most workgroups one dispatch can start along each axis.
        let max = gpu.max_workgroups_per_dispatch();
        // Fail before the GPU does, with a clear message.
        if let Some(&too_many) = workgroups.iter().find(|&&count| count > max) {
            // Say the count and the limit.
            return Err(EngineError::TooManyWorkgroups { workgroups: too_many, max });
        }
        // Connect the buffers.
        let bind_group = bind_buffers(gpu, label, pipeline, buffers);
        // A clone of a wgpu pipeline is a second handle to the same compiled shader, not a second compile.
        Ok(Self { pipeline: pipeline.clone(), bind_group, workgroups, work })
    }

    // The kernel name and its bytes.
    pub(crate) fn work(&self) -> Work {
        self.work
    }

    // Add this dispatch to an open compute pass. The profiler gives each dispatch its own pass.
    pub(crate) fn record(&self, pass: &mut wgpu::ComputePass) {
        // Use this kernel.
        pass.set_pipeline(&self.pipeline);
        // Connect its buffers to group 0.
        pass.set_bind_group(0, &self.bind_group, &[]);
        // The workgroup counts along x, y and z.
        let [x, y, z] = self.workgroups;
        // Start the full grid of workgroups.
        pass.dispatch_workgroups(x, y, z);
    }
}

// A kernel bound to the slots of the activation plan: one or more dispatches, and the buffers that only this kernel uses,
// such as its weights and its uniforms. It is made once and runs many times.
pub(crate) struct BoundKernel {
    // The dispatches, in run order. Attention has one per chunk. The other kernels have one.
    dispatches: Vec<Dispatch>,
    // The buffers of this kernel. We keep them as long as the dispatches, because the browser frees a dropped buffer.
    _buffers: Vec<GpuBuffer>,
}

impl BoundKernel {
    // Keep dispatches with their buffers.
    pub(crate) fn new(dispatches: Vec<Dispatch>, buffers: Vec<GpuBuffer>) -> Self {
        Self { dispatches, _buffers: buffers }
    }

    // The same kernel, which also keeps `more` buffers, for example the weights that its caller read.
    pub(crate) fn keeping(mut self, more: impl IntoIterator<Item = GpuBuffer>) -> Self {
        // Add them to the list.
        self._buffers.extend(more);
        // The kernel.
        self
    }

    // The dispatches, ready to record.
    pub(crate) fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
    }
}

// Record `runs` runs into one command list. One run is every dispatch in `dispatches`, in order.
// Most kernels are one dispatch per run. Attention is one dispatch per chunk. Nothing runs until it is submitted.
pub(crate) fn encode_runs(gpu: &Gpu, dispatches: &[Dispatch], runs: u32) -> wgpu::CommandBuffer {
    // Start a list of GPU commands.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    // A compute pass is a part of the list that runs compute shaders. The block ends the pass.
    {
        // Open the pass. WebGPU makes each dispatch see the writes of the dispatches before it.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        // Every run, then every dispatch of the run.
        (0..runs).for_each(|_| dispatches.iter().for_each(|dispatch| dispatch.record(&mut pass)));
    }
    // Close the list so it can be submitted.
    encoder.finish()
}

// Compile WGSL text into a compute pipeline whose entry function is `main`.
pub(crate) fn compute_pipeline(gpu: &Gpu, label: &str, source: &str) -> wgpu::ComputePipeline {
    // Compile the WGSL text into a shader module.
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        // The name shows up in compile errors.
        label: Some(label),
        // The source is WGSL text.
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    // A pipeline is the compiled shader plus the layout of the buffers it uses.
    gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        // The name shows up in GPU error messages.
        label: Some(label),
        // None: wgpu reads the @group and @binding lines and builds the layout for us.
        layout: None,
        // The compiled shader.
        module: &module,
        // The function marked @compute.
        entry_point: Some("main"),
        // No special compile options.
        compilation_options: Default::default(),
        // No pipeline cache.
        cache: None,
    })
}

// Connect buffers to the @binding slots of group 0. The position in the list is the binding number.
fn bind_buffers(gpu: &Gpu, label: &str, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    // One entry per buffer. The layout from the shader says if it is a storage or a uniform buffer.
    let entries: Vec<wgpu::BindGroupEntry> = (0u32..)
        // Pair each binding number with its buffer.
        .zip(buffers)
        // Each pair becomes one entry that covers the whole buffer.
        .map(|(binding, buffer)| wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() })
        // Put the entries into one list.
        .collect();
    // Count it, so a test can see that a run makes no new bind groups.
    gpu.count_bind_group();
    // Make the bind group with the layout that wgpu built from the shader.
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: &pipeline.get_bind_group_layout(0), entries: &entries })
}
