// The input values. Group 0, slot 0. "read" means the shader can only read it.
@group(0) @binding(0) var<storage, read> input: array<f32>;
// The output values. Group 0, slot 1. "read_write" means the shader can write it.
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// A compute shader. The GPU starts it in groups of 64 invocations. Rust's WORKGROUP_SIZE must match 64.
@compute @workgroup_size(64)
// global_invocation_id is the position of this invocation among all invocations of the dispatch.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // We dispatch along x only, so x alone is the index of this invocation's element.
    let index = id.x;
    // The last workgroup can stick out past the end of the array. Those invocations do nothing.
    if (index >= arrayLength(&input)) {
        // Stop this invocation only. The others keep going.
        return;
    }
    // Read this invocation's one value.
    let x = input[index];
    // sigmoid(x) = 1 / (1 + e^-x). It goes from 0 (very negative x) to 1 (very positive x).
    let sigmoid = 1.0 / (1.0 + exp(-x));
    // SiLU is x times sigmoid(x). Write it to the same position in the output.
    output[index] = x * sigmoid;
}
