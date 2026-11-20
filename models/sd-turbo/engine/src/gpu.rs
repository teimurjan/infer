// Cell holds the bind group count and the longest submit, and RefCell the profile, which we change through a shared reference.
use std::cell::{Cell, RefCell};
// Rc shares the memory counter with every buffer.
use std::rc::Rc;

// The sending end of a one-shot channel. A GPU callback uses it to hand us its result.
use futures_channel::oneshot::{self, Sender};
// create_buffer_init makes a GPU buffer and fills it in one call.
use wgpu::util::DeviceExt;

// Our helpers for f32 bytes.
use crate::bytes::{f32_from_le_bytes, f32_to_le_bytes};
// Our error type.
use crate::error::EngineError;
// Buffers that count their bytes.
use crate::memory::{GpuBuffer, MemoryCategory, MemoryCounter, MemoryReport};
// What one dispatch does.
use crate::dispatch::Work;
// The timing per kernel.
use crate::profile::{Profile, Timer};

// One open connection to the GPU. Make it once and use it for every kernel.
pub struct Gpu {
    // Creates buffers, shaders and pipelines.
    pub(crate) device: wgpu::Device,
    // Sends finished command lists to the GPU and writes data into buffers.
    pub(crate) queue: wgpu::Queue,
    // The bytes of every buffer made through this Gpu that still exists, per category, and the peak.
    memory: Rc<MemoryCounter>,
    // The number of bind groups made through this Gpu so far. A test uses it to prove that a run makes none.
    bind_groups: Cell<u64>,
    // The seconds of the longest submit since the last take_longest_submit. The 20-image run prints it per image.
    longest_submit: Cell<f64>,
    // The timing per kernel, while a profile runs. None for a normal run.
    profile: RefCell<Option<Profile>>,
}

impl Gpu {
    // Find a GPU and open a device on it. On the Mac this uses Metal. In the browser it uses WebGPU.
    pub async fn new() -> Result<Self, EngineError> {
        // The entry point to wgpu. Default turns on every backend that this build has.
        let instance = wgpu::Instance::default();
        // Ask for an adapter. HighPerformance picks the faster GPU when there are two.
        let options = wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() };
        // Wait for the adapter, or stop with an error.
        let adapter = instance.request_adapter(&options).await.map_err(EngineError::NoAdapter)?;
        // Ask for timestamps if the adapter has them. The profiler uses them. Nothing else changes with them.
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        // Default asks for the standard WebGPU limits, which every WebGPU GPU supports.
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { required_features, ..Default::default() }).await.map_err(EngineError::NoDevice)?;
        // Keep the device and the queue together.
        Ok(Self { device, queue, memory: Rc::default(), bind_groups: Cell::new(0), longest_submit: Cell::new(0.0), profile: RefCell::new(None) })
    }

    // The largest number of workgroups that one dispatch can start along x.
    pub(crate) fn max_workgroups_per_dispatch(&self) -> u32 {
        // WebGPU guarantees at least 65535.
        self.device.limits().max_compute_workgroups_per_dimension
    }

    // The most bytes that our GPU buffers held at one time since the last reset_peak_bytes.
    // It counts only buffers that the engine made, not memory that the driver or the browser uses.
    pub(crate) fn peak_bytes(&self) -> u64 {
        self.memory.peak()
    }

    // Start a new peak measurement from the bytes held now.
    pub fn reset_peak_bytes(&self) {
        self.memory.reset_peak();
    }

    // The live bytes of our GPU buffers per category, and the peak, right now.
    pub fn memory_report(&self) -> MemoryReport {
        self.memory.report()
    }

    // Remember one submit time if it is the longest so far. submit.rs calls this after each submit.
    pub(crate) fn record_submit(&self, seconds: f64) {
        self.longest_submit.set(self.longest_submit.get().max(seconds));
    }

    // The longest submit since the last call, in seconds, and start again from 0.
    pub fn take_longest_submit(&self) -> f64 {
        self.longest_submit.replace(0.0)
    }

    // True if this device can write GPU timestamps at the edges of a compute pass ("timestamp-query" in WebGPU).
    pub fn has_timestamps(&self) -> bool {
        self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    // Start a profile, or end it with None. start_profile in profile/ calls this.
    pub(crate) fn set_profile(&self, profile: Option<Profile>) {
        self.profile.replace(profile);
    }

    // End the profile and give it back. None if no profile ran.
    pub fn take_profile(&self) -> Option<Profile> {
        self.profile.take()
    }

    // What times each dispatch of the running profile. None for a normal run. The borrow ends before the caller awaits.
    pub(crate) fn profile_timer(&self) -> Option<Timer> {
        self.profile.borrow().as_ref().map(Profile::timer)
    }

    // Add dispatches and their seconds to the running profile. Does nothing if no profile runs.
    pub(crate) fn add_to_profile(&self, dispatches: impl Iterator<Item = (Work, f64)>) {
        // The borrow ends at the end of this call, before any await of the caller.
        if let Some(profile) = self.profile.borrow_mut().as_mut() {
            dispatches.for_each(|(work, seconds)| profile.add(work, seconds));
        }
    }

    // Make a buffer that a query set can be copied into, and that read_bytes can copy out of. It counts as read-back memory.
    pub(crate) fn query_resolve_buffer(&self, bytes: u64) -> GpuBuffer {
        // QUERY_RESOLVE: resolve_query_set can write it. COPY_SRC: read_bytes can copy out of it.
        let descriptor = wgpu::BufferDescriptor { label: Some("timestamps"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false };
        // Make it and count it.
        self.counted(self.device.create_buffer(&descriptor), MemoryCategory::ReadBack)
    }

    // Count one more bind group. dispatch.rs calls this each time it makes one.
    pub(crate) fn count_bind_group(&self) {
        self.bind_groups.set(self.bind_groups.get() + 1);
    }

    // The number of GPU buffers made so far, of every category.
    pub fn buffers_made(&self) -> u64 {
        self.memory.buffers_made()
    }

    // The number of bind groups made so far.
    pub fn bind_groups_made(&self) -> u64 {
        self.bind_groups.get()
    }

    // The largest buffer that this device allows, in bytes. The default WebGPU limits give 256 MiB.
    pub(crate) fn max_buffer_bytes(&self) -> u64 {
        self.device.limits().max_buffer_size
    }

    // Start counting a buffer that was just made, under its category.
    fn counted(&self, buffer: wgpu::Buffer, category: MemoryCategory) -> GpuBuffer {
        GpuBuffer::new(buffer, category, &self.memory)
    }

    // Make a storage buffer (GPU memory that a shader can read) that holds `values`.
    pub(crate) fn storage_buffer_from(&self, label: &str, values: &[f32], category: MemoryCategory) -> GpuBuffer {
        // 4 bytes per value, lowest byte first.
        self.storage_buffer_from_bytes(label, &f32_to_le_bytes(values), category)
    }

    // Make a storage buffer that holds raw bytes, for example float32 weights read straight from a file.
    pub(crate) fn storage_buffer_from_bytes(&self, label: &str, bytes: &[u8], category: MemoryCategory) -> GpuBuffer {
        // Describe the buffer: its name, its bytes, and what we use it for.
        let descriptor = wgpu::util::BufferInitDescriptor {
            // The name shows up in GPU error messages.
            label: Some(label),
            // The bytes to copy to the GPU.
            contents: bytes,
            // STORAGE: a shader can use it. COPY_SRC: we can copy out of it.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        };
        // Make the buffer and copy the bytes in.
        self.counted(self.device.create_buffer_init(&descriptor), category)
    }

    // Make an empty weight buffer of `bytes` bytes. The weight loader fills it with queue.write_buffer, one piece at a time.
    pub(crate) fn empty_weight_buffer(&self, label: &str, bytes: u64) -> GpuBuffer {
        // Describe the buffer.
        let descriptor = wgpu::BufferDescriptor {
            // The tensor name, for GPU error messages.
            label: Some(label),
            // The caller rounds the size up to a multiple of 4, as write_buffer needs.
            size: bytes,
            // STORAGE: a shader can read it. COPY_DST: write_buffer can fill it. COPY_SRC: a test can read it back.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            // The data comes later through write_buffer.
            mapped_at_creation: false,
        };
        // Make the buffer. WebGPU fills new buffers with zeros.
        self.counted(self.device.create_buffer(&descriptor), MemoryCategory::Weights)
    }

    // Make an empty storage buffer with room for `count` f32 values. A shader writes into it.
    pub(crate) fn empty_storage_buffer(&self, label: &str, count: usize) -> GpuBuffer {
        // Describe the buffer.
        let descriptor = wgpu::BufferDescriptor {
            // The name shows up in GPU error messages.
            label: Some(label),
            // 4 bytes per f32.
            size: f32_buffer_size(count),
            // STORAGE: a shader can use it. COPY_SRC: we can copy out of it. COPY_DST: the CPU can write into it.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            // We do not fill it at creation, so it does not need to start mapped.
            mapped_at_creation: false,
        };
        // Make the buffer. WebGPU fills new buffers with zeros. Kernels write into it, so it counts as activations.
        self.counted(self.device.create_buffer(&descriptor), MemoryCategory::Activations)
    }

    // Make a small uniform buffer (read-only constants for a shader) that holds `values`, such as a tensor shape.
    pub(crate) fn uniform_buffer(&self, label: &str, values: &[u32]) -> GpuBuffer {
        // Pad with zeros to a multiple of 4 values (16 bytes), the size step of WGSL uniform structs.
        let padded: Vec<u32> = values.iter().copied().chain(std::iter::repeat(0)).take(values.len().next_multiple_of(4)).collect();
        // Each u32 becomes 4 bytes, lowest byte first.
        let bytes: Vec<u8> = padded.iter().flat_map(|value| value.to_le_bytes()).collect();
        // Make the buffer and copy the bytes in. UNIFORM: a shader can read it as var<uniform>.
        self.counted(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &bytes, usage: wgpu::BufferUsages::UNIFORM }), MemoryCategory::Uniforms)
    }

    // Send the waiting buffer uploads to the GPU and wait until they are done. In the browser, this also waits for the shader compiles that were asked for before.
    // write_buffer only queues an upload. Without this, the upload runs inside the next submit and adds to its time.
    pub async fn finish_uploads(&self) -> Result<(), EngineError> {
        // An empty submit sends the queued uploads to the GPU.
        self.queue.submit(std::iter::empty());
        // Wait until they are done.
        self.wait_for_submitted_work().await
    }

    // Wait until the GPU has finished every command list submitted so far (WebGPU's onSubmittedWorkDone).
    pub(crate) async fn wait_for_submitted_work(&self) -> Result<(), EngineError> {
        // Ask the queue to call us when the GPU is done, and wait for that call.
        self.wait_for_callback(|sender| {
            // The callback only sends an empty message. If nobody waits, there is nothing to do.
            self.queue.on_submitted_work_done(move || {
                let _ = sender.send(());
            });
        })
        .await
    }

    // Write f32 values to the start of a buffer. The upload runs before the next submit.
    pub(crate) fn write_f32(&self, target: &wgpu::Buffer, values: &[f32]) {
        // 4 bytes per value, lowest byte first, at offset 0.
        self.queue.write_buffer(target, 0, &f32_to_le_bytes(values));
    }

    // Copy a GPU buffer back to the CPU and read it as f32 values.
    pub(crate) async fn read_f32(&self, source: &wgpu::Buffer) -> Result<Vec<f32>, EngineError> {
        // Read the raw bytes, then build one f32 from each group of 4.
        Ok(f32_from_le_bytes(&self.read_bytes(source).await?))
    }

    // Copy the first `count` f32 values of a GPU buffer back to the CPU. A slot of the activation plan can be larger than its tensor.
    pub(crate) async fn read_f32_prefix(&self, source: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, EngineError> {
        // Read only the bytes of the tensor, then build one f32 from each group of 4.
        Ok(f32_from_le_bytes(&self.read_byte_prefix(source, f32_buffer_size(count)).await?))
    }

    // Copy a whole GPU buffer back to the CPU and return its raw bytes.
    pub(crate) async fn read_bytes(&self, source: &wgpu::Buffer) -> Result<Vec<u8>, EngineError> {
        // Every byte of the buffer.
        self.read_byte_prefix(source, source.size()).await
    }

    // Copy the first `size` bytes of a GPU buffer back to the CPU.
    // The CPU cannot read a STORAGE buffer directly, so we copy into a staging buffer that it can read.
    pub(crate) async fn read_byte_prefix(&self, source: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, EngineError> {
        // Describe the staging buffer.
        let staging_descriptor = wgpu::BufferDescriptor {
            // The name shows up in GPU error messages.
            label: Some("read back"),
            // Only the bytes that we read, so the staging buffer adds no more than that to the peak.
            size,
            // MAP_READ: the CPU can read it. COPY_DST: we can copy into it.
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            // It gets its bytes from the GPU, not from the CPU.
            mapped_at_creation: false,
        };
        // Make the staging buffer. It is counted too, because it adds to the peak while it exists.
        let staging = self.counted(self.device.create_buffer(&staging_descriptor), MemoryCategory::ReadBack);
        // A command encoder records GPU commands into a list.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Record one command: copy the first `size` bytes from source to staging.
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        // Send the list to the GPU. The GPU runs it after all earlier submits.
        self.queue.submit([encoder.finish()]);
        // Ask for CPU read access, and wait until wgpu calls back with the result.
        let map_result = self
            .wait_for_callback(|sender| {
                // wgpu calls the closure when the copy is done and the buffer is readable.
                staging.map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = sender.send(result);
                });
            })
            .await?;
        // The callback ran, but the map itself can still have failed.
        map_result.map_err(EngineError::Map)?;
        // Copy the mapped bytes into a list that we own.
        let bytes = staging.get_mapped_range(..).map_err(EngineError::MapRange)?.to_vec();
        // Give the buffer back to the GPU. The view of its bytes is already gone at this point.
        staging.unmap();
        // The bytes that the GPU holds.
        Ok(bytes)
    }

    // Register a GPU callback with `register`, then wait for the value it sends.
    async fn wait_for_callback<T>(&self, register: impl FnOnce(Sender<T>)) -> Result<T, EngineError> {
        // A channel with one sending end and one receiving end.
        let (sender, receiver) = oneshot::channel();
        // Give the sending end to the callback.
        register(sender);
        // Native: block until the GPU is done and the callback ran. Browser: does nothing, the browser runs the callback.
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(EngineError::Poll)?;
        // Wait for the value. An error here means wgpu dropped the callback without calling it.
        receiver.await.map_err(|_| EngineError::CallbackDropped)
    }
}

// The size in bytes of a buffer that holds `count` f32 values.
fn f32_buffer_size(count: usize) -> u64 {
    // usize is at most 64 bits on every target we build for, so this cannot lose bits.
    (count * std::mem::size_of::<f32>()) as u64
}
