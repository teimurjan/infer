// output[i] = the float16 value i of `input`, as f32. The UNet stage runs this once, in prepare, for the small float16
// tensors of the packed file (biases and norm scales), so the float32 kernels can read them.

// The float16 values, two per word, value 2j in the low 16 bits. The buffer of the weight file, rounded up to whole words.
@group(0) @binding(0) var<storage, read> input: array<u32>;
// The f32 values. This buffer has exactly one f32 per float16 value, so its length is the count.
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE must match 256.
@compute @workgroup_size(256)
// Each invocation writes one value.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The value of this invocation.
    let index = id.x;
    // The output buffer has exactly the values, so its length is the count. The last workgroup can stick out past it.
    if (index >= arrayLength(&output)) {
        return;
    }
    // unpack2x16float gives the low half first.
    output[index] = unpack2x16float(input[index / 2u])[index % 2u];
}
