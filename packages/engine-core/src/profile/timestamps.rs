// A kernel ready to record.
use crate::dispatch::Dispatch;
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// One timed submit, and the check against the project limit.
use crate::submit::{check_submit, timed_submit};

// The GPU error check.
use super::scope::caught;

// Bytes of one timestamp: a u64 count of GPU clock ticks.
const TIMESTAMP_BYTES: u64 = 8;
// Every dispatch writes two timestamps: one at its start and one at its end.
const TIMESTAMPS_PER_DISPATCH: u32 = 2;
// The most timestamps in one query set that WebGPU allows. Enough for 2048 dispatches in one submit.
const TIMESTAMPS: u32 = wgpu::QUERY_SET_MAX_QUERIES;
// Nanoseconds in one second.
const NANOSECONDS: f64 = 1e9;

// The query set that the GPU writes timestamps into, and the buffer that the CPU reads them from.
// One of each serves the whole profile. A new query set per submit ran out in Chromium on the Mac after about 167 submits.
pub struct Timestamps {
    // 4096 timestamp slots.
    query_set: wgpu::QuerySet,
    // 4096 x 8 bytes. It counts as read-back memory while the profile lives.
    resolved: GpuBuffer,
}

impl Timestamps {
    // Make the query set and its buffer, and fail if the GPU cannot.
    pub async fn new(gpu: &Gpu) -> Result<Self, EngineError> {
        // The GPU reports a failed allocation as an error, not as a result, so catch it.
        caught(gpu, "timestamp query set", async {
            // The slots.
            let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("kernel timestamps"), ty: wgpu::QueryType::Timestamp, count: TIMESTAMPS });
            // The buffer.
            let resolved = gpu.query_resolve_buffer(u64::from(TIMESTAMPS) * TIMESTAMP_BYTES);
            Ok(Self { query_set, resolved })
        })
        .await
    }
}

// Run `dispatches` as one submit, each in its own compute pass with a timestamp at the start and the end of the pass.
// WebGPU allows timestamps only at the edges of a pass, so one pass per dispatch gives one time per dispatch.
// The passes run in order, and each pass sees the writes of the passes before it, as the dispatches of one pass do.
// Returns the seconds of the submit and the GPU seconds of each dispatch.
pub(super) async fn time_dispatches(gpu: &Gpu, timestamps: &Timestamps, label: &str, dispatches: &[Dispatch]) -> Result<(f64, Vec<f64>), EngineError> {
    // A step without dispatches takes no time.
    if dispatches.is_empty() {
        return Ok((0.0, Vec::new()));
    }
    // Two timestamps per dispatch, and the query set has 4096.
    let count = u32::try_from(dispatches.len()).ok().and_then(|dispatches| dispatches.checked_mul(TIMESTAMPS_PER_DISPATCH)).filter(|&count| count <= TIMESTAMPS);
    let count = count.ok_or(EngineError::TooManyTimestamps { label: label.to_string(), dispatches: dispatches.len(), max: (TIMESTAMPS / TIMESTAMPS_PER_DISPATCH) as usize })?;
    // Record every dispatch in its own pass.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for (index, dispatch) in (0u32..).zip(dispatches) {
        // Timestamp 2i at the start of pass i, timestamp 2i + 1 at its end.
        let writes = wgpu::ComputePassTimestampWrites { query_set: &timestamps.query_set, beginning_of_pass_write_index: Some(index * TIMESTAMPS_PER_DISPATCH), end_of_pass_write_index: Some(index * TIMESTAMPS_PER_DISPATCH + 1) };
        // Open the pass. It closes at the end of this loop step.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: Some(writes) });
        // The one dispatch of this pass.
        dispatch.record(&mut pass);
    }
    // Copy the timestamps of this submit into the buffer, after the last pass.
    encoder.resolve_query_set(&timestamps.query_set, 0..count, &timestamps.resolved, 0);
    // Submit, wait, and stop the run if the submit took too long.
    let seconds = check_submit(gpu, label, timed_submit(gpu, encoder.finish()).await?)?;
    // Read the timestamps of this submit back. This is one more small submit, after the timed one.
    let bytes = gpu.read_byte_prefix(&timestamps.resolved, u64::from(count) * TIMESTAMP_BYTES).await?;
    // The length of one tick in seconds. WebGPU always counts in nanoseconds. Metal can use another tick.
    let tick_seconds = f64::from(gpu.queue.get_timestamp_period()) / NANOSECONDS;
    // The seconds of each dispatch.
    Ok((seconds, dispatch_seconds(&bytes, tick_seconds)))
}

// The seconds between each pair of timestamps in `bytes`: start of dispatch i, end of dispatch i.
// An end before its start (a clock that jumped) counts as 0, not as a huge number.
fn dispatch_seconds(bytes: &[u8], tick_seconds: f64) -> Vec<f64> {
    // Each timestamp is a u64, lowest byte first.
    let ticks: Vec<u64> = bytes.chunks_exact(TIMESTAMP_BYTES as usize).map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunks_exact gives 8 bytes"))).collect();
    // End minus start, in seconds.
    ticks.chunks_exact(TIMESTAMPS_PER_DISPATCH as usize).map(|pair| pair[1].saturating_sub(pair[0]) as f64 * tick_seconds).collect()
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function into the tests.
    use super::dispatch_seconds;

    // A wrong pairing (start of one dispatch with the end of another) would give each kernel the time of its neighbour.
    #[test]
    fn pairs_start_and_end_of_each_dispatch() {
        // Dispatch 0 from tick 100 to 400, dispatch 1 from 400 to 1400, dispatch 2 from 900 to 800 (a clock jump).
        let bytes: Vec<u8> = [100u64, 400, 400, 1400, 900, 800].iter().flat_map(|tick| tick.to_le_bytes()).collect();
        // One tick is 2 seconds here, so the products are exact in f64.
        assert_eq!(dispatch_seconds(&bytes, 2.0), vec![600.0, 2000.0, 0.0]);
    }
}
