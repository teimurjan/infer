// Streams real weight files into GPU buffers in chunks of an odd size, as the browser does, and reads every tensor back.

// File and fs read the weight files. Read reads the next chunk.
use std::fs::{self, File};
use std::io::Read;
// read_exact_at reads the bytes of one tensor from its offset.
use std::os::unix::fs::FileExt;
// Path builds the file paths.
use std::path::{Path, PathBuf};

// Deserialize builds one index line in the test.
use serde::Deserialize;

// The engine parts that the test uses.
use engine::{EngineError, Gpu, GpuWeights, MemoryCategory, WeightUpload};

// An odd chunk size, so most chunks end in the middle of a 4-byte word. It is a little over 64 KiB.
const CHUNK_BYTES: usize = 65_537;
// CLAUDE.md: one GPU buffer on the iPhone can be at most about 1073 MB.
const BUFFER_LIMIT_BYTES: u64 = 1_073_000_000;

// The fields of one index line that the test needs.
#[derive(Deserialize)]
struct IndexLine {
    // The tensor name.
    name: String,
    // Where it starts in the .bin file.
    offset: u64,
}

// weights/<name>.bin in the project folder, one level above engine/.
fn weight_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("weights").join(format!("{name}.bin"))
}

// Stream the first `limit` bytes of weights/<name>.bin into the GPU, in chunks of CHUNK_BYTES, and finish the upload.
async fn stream(gpu: &Gpu, name: &str, limit: u64) -> Result<GpuWeights, EngineError> {
    // The index text, next to the .bin file.
    let index = fs::read_to_string(weight_path(name).with_extension("json")).expect("weight index");
    // Parse the index and plan the buffers.
    let mut upload = WeightUpload::begin(gpu, &index)?;
    // The open file, cut at `limit` bytes.
    let mut file = File::open(weight_path(name)).expect("weight file").take(limit);
    // One chunk of room, used again for every chunk. This is the only file memory on the CPU side.
    let mut chunk = vec![0; CHUNK_BYTES];
    // Read until the end of the file.
    loop {
        // The next chunk. 0 bytes means the end.
        let length = file.read(&mut chunk).expect("read a chunk");
        if length == 0 {
            break;
        }
        // Write it into the buffers. Wait for the GPU when the upload asks for it.
        if upload.write(gpu, &chunk[..length]) {
            gpu.finish_uploads().await?;
        }
    }
    // Check that every tensor arrived.
    let weights = upload.finish(gpu)?;
    // Wait until the last writes are on the GPU.
    gpu.finish_uploads().await?;
    Ok(weights)
}

// Every tensor on the GPU must equal its bytes in the file, and its buffer padding must be zero.
async fn check_against_file(gpu: &Gpu, name: &str, weights: &GpuWeights) {
    // The index lines, for the offsets.
    let lines: Vec<IndexLine> = serde_json::from_str(&fs::read_to_string(weight_path(name).with_extension("json")).unwrap()).unwrap();
    // The file, read by offset.
    let file = File::open(weight_path(name)).unwrap();
    // One tensor at a time.
    for line in &lines {
        // The tensor on the GPU, with its stored length.
        let tensor = weights.tensor(&line.name).unwrap();
        // The bytes in its buffer.
        let on_gpu = weights.read_back(gpu, &line.name).await.unwrap();
        // The same bytes from the file.
        let mut in_file = vec![0; tensor.bytes as usize];
        file.read_exact_at(&mut in_file, line.offset).unwrap();
        // Split the buffer into the tensor and its padding.
        let (stored, padding) = on_gpu.split_at(tensor.bytes as usize);
        // A byte in the wrong place fails here.
        assert!(stored == in_file, "{name}: {} differs from the file", line.name);
        // The padding is 0 to 3 bytes and must be zero.
        assert!(padding.len() < 4 && padding.iter().all(|&byte| byte == 0), "{name}: {} has padding {padding:?}", line.name);
    }
}

// The streamed buffers must match the files, the weights category must count exactly their bytes, and dropping them must free them.
#[test]
fn streamed_weights_match_the_files() {
    pollster::block_on(async {
        // Open the GPU.
        let gpu = Gpu::new().await.unwrap();
        // taesd has float32 only. unet_packed has float16, int8 and int4 tensors.
        for name in ["taesd", "unet_packed"] {
            // The time of the stream, for the log.
            let started = web_time::Instant::now();
            // Stream the whole file.
            let weights = stream(&gpu, name, u64::MAX).await.unwrap();
            // The seconds from the first chunk until the GPU had the last write.
            let seconds = started.elapsed().as_secs_f64();
            // The memory counter, right after the stream.
            let report = gpu.memory_report();
            // One line for the log, then the counter.
            println!("{name}: {weights}, streamed in {:.1} ms\n{report}", seconds * 1000.0);
            // The weights category counts exactly these buffers. No other weights are alive in this test.
            assert_eq!(report.bytes(MemoryCategory::Weights), weights.buffer_bytes());
            // Each buffer is under the iPhone limit.
            assert!(weights.largest_buffer() <= BUFFER_LIMIT_BYTES);
            // Every tensor matches the file.
            check_against_file(&gpu, name, &weights).await;
            // Free the buffers.
            drop(weights);
            // The weights category is empty again.
            assert_eq!(gpu.memory_report().bytes(MemoryCategory::Weights), 0);
        }
    });
}

// A stream that stops before the last tensor must fail, and must free the buffers that it made.
#[test]
fn a_cut_stream_fails() {
    pollster::block_on(async {
        // Open the GPU.
        let gpu = Gpu::new().await.unwrap();
        // Half of the TAESD file.
        let result = stream(&gpu, "taesd", 4_890_368).await;
        // Print the error for the log.
        let error = result.err().expect("a cut stream must fail");
        println!("cut stream: {error}");
        // The right kind of error.
        assert!(matches!(error, EngineError::StreamEnded { received: 4_890_368, .. }));
        // The buffers of the failed upload are gone.
        assert_eq!(gpu.memory_report().bytes(MemoryCategory::Weights), 0);
    });
}
