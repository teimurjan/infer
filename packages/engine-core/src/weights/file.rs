// fs reads the index text. File holds the open .bin file.
use std::fs::{self, File};
// Read reads the next chunk of a file when it streams to the GPU.
use std::io::Read;
// read_exact_at reads bytes at an offset without moving a shared file position, so it needs only &self.
use std::os::unix::fs::FileExt;
// Path is the location of the .bin file.
use std::path::Path;

// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;

// The index format, the shared reader interface, and the streaming loader with its result.
use super::{GpuWeights, WeightIndex, WeightSource, WeightTensor, WeightUpload};

// A .bin file from convert_weights.py and its .json index, read one tensor at a time. Native only.
pub struct WeightFile {
    // The parsed index.
    index: WeightIndex,
    // The open .bin file.
    file: File,
}

impl WeightFile {
    // Open `bin`, for example weights/unet.bin, and parse the index with the same name and a .json extension.
    pub fn open(bin: &Path) -> Result<Self, EngineError> {
        // weights/unet.bin -> weights/unet.json.
        let index_path = bin.with_extension("json");
        // Read the whole index. It is small: about 100 kB for the UNet.
        let json = fs::read_to_string(&index_path).map_err(|error| EngineError::WeightRead(format!("{}: {error}", index_path.display())))?;
        // Open the .bin file. We read from it only when a tensor is asked for.
        let file = File::open(bin).map_err(|error| EngineError::WeightRead(format!("{}: {error}", bin.display())))?;
        // Parse the index and keep both.
        Ok(Self { index: WeightIndex::parse(&json)?, file })
    }
}

impl WeightSource for WeightFile {
    // Read the bytes of one tensor from its offset in the .bin file.
    fn read(&self, name: &str) -> Result<WeightTensor, EngineError> {
        // Find the tensor in the index.
        let entry = self.index.entry(name)?;
        // Room for exactly this tensor.
        let mut bytes = vec![0; entry.byte_length()?];
        // Read that many bytes, starting at the offset. A short file is an error.
        self.file.read_exact_at(&mut bytes, entry.offset).map_err(|error| EngineError::WeightRead(format!("{name}: {error}")))?;
        // The shape comes from the index.
        Ok(WeightTensor { shape: entry.shape.clone(), bytes })
    }

    // Read only the bytes of one row from the .bin file.
    fn read_row(&self, name: &str, row: usize) -> Result<Vec<u8>, EngineError> {
        // Where the row starts, and its length.
        let (offset, length) = self.index.entry(name)?.row_range(row)?;
        // Room for exactly this row.
        let mut bytes = vec![0; length];
        // Read it. A short file is an error.
        self.file.read_exact_at(&mut bytes, offset).map_err(|error| EngineError::WeightRead(format!("{name} row {row}: {error}")))?;
        // The bytes of the row.
        Ok(bytes)
    }
}

// Bytes per chunk when a file streams to the GPU. 2 MiB is near the largest chunk that Safari's fetch gave ("Weight loader on the iPhone").
const STREAM_CHUNK_BYTES: usize = 2 * 1024 * 1024;

// Stream weights/<name>.bin into one GPU buffer per tensor, with the same loader as the browser, and wait until the GPU has it all.
// Only one chunk is in CPU memory at a time.
pub async fn load_to_gpu(gpu: &Gpu, bin: &Path) -> Result<GpuWeights, EngineError> {
    // weights/unet_packed.bin -> weights/unet_packed.json.
    let index_path = bin.with_extension("json");
    // Read the whole index. It is small.
    let index = fs::read_to_string(&index_path).map_err(|error| EngineError::WeightRead(format!("{}: {error}", index_path.display())))?;
    // Parse the index and check its layout. No buffer exists yet.
    let mut upload = WeightUpload::begin(gpu, &index)?;
    // Open the file.
    let mut file = File::open(bin).map_err(|error| EngineError::WeightRead(format!("{}: {error}", bin.display())))?;
    // One chunk of room, used again for every chunk.
    let mut chunk = vec![0; STREAM_CHUNK_BYTES];
    // Read until the end of the file.
    loop {
        // The next chunk. 0 bytes means the end.
        let length = file.read(&mut chunk).map_err(|error| EngineError::WeightRead(format!("{}: {error}", bin.display())))?;
        if length == 0 {
            break;
        }
        // Write it. Every 32 MB, wait for the GPU, so the staging area of write_buffer stays small.
        if upload.write(gpu, &chunk[..length]) {
            gpu.finish_uploads().await?;
        }
    }
    // Check that every tensor arrived.
    let weights = upload.finish(gpu)?;
    // Wait until the GPU has every byte.
    gpu.finish_uploads().await?;
    // The tensors on the GPU.
    Ok(weights)
}
