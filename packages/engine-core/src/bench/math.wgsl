// One start value per invocation.
@group(0) @binding(0) var<storage, read> input: array<f32>;
// One result per invocation. We must write it, or the compiler can delete the whole loop.
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Loop steps per invocation. Each step does 4 fma, and each fma is 2 operations (a multiply and an add).
// So one invocation does 256 x 4 x 2 = 2048 operations. Rust's OPS_PER_INVOCATION must match.
const ITERATIONS: u32 = 256u;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE must match.
@compute @workgroup_size(256)
// global_invocation_id is the position of this invocation among all invocations of the dispatch.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // We dispatch along x only, so x alone is the index of this invocation's value.
    let index = id.x;
    // Invocations past the end do nothing. With the sizes in math.rs, none are past the end.
    if (index >= arrayLength(&input)) {
        return;
    }
    // The multiplier comes from the buffer, so the compiler cannot work out the answer in advance.
    let a = input[index];
    // The value that the loop keeps changing.
    var x = a;
    // The long arithmetic loop.
    for (var iteration = 0u; iteration < ITERATIONS; iteration++) {
        // fma(x, a, 0.5) is x * a + 0.5 in one GPU instruction. With a below 1, x settles between 1 and 5,
        // so it never becomes infinity or a tiny "subnormal" number, which can be slower on some GPUs.
        x = fma(x, a, 0.5);
        // Each line needs the result of the line before, so the GPU cannot skip any of them.
        x = fma(x, a, 0.5);
        // Third fma of this step.
        x = fma(x, a, 0.5);
        // Fourth fma of this step.
        x = fma(x, a, 0.5);
    }
    // Write the result, so the work is kept.
    output[index] = x;
}
