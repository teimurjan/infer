// The float16 copy of math.wgsl: the same loop, but every multiply-add is in float16.
// "enable f16" turns on the f16 type. It works only on a device that was opened with the shader-f16 feature.
enable f16;

// One start value per invocation. The buffers stay float32, so both benchmarks use the same buffers.
@group(0) @binding(0) var<storage, read> input: array<f32>;
// One result per invocation. We must write it, or the compiler can delete the whole loop.
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Loop steps per invocation, as in math.wgsl. Rust's OPS_PER_INVOCATION must match.
const ITERATIONS: u32 = 256u;

// 256 invocations per workgroup, as in math.wgsl.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // We dispatch along x only, so x alone is the index of this invocation's value.
    let index = id.x;
    // Invocations past the end do nothing. With the sizes in math.rs, none are past the end.
    if (index >= arrayLength(&input)) {
        return;
    }
    // The multiplier, rounded to float16 once. It comes from the buffer, so the compiler cannot work out the answer.
    let a = f16(input[index]);
    // The value that the loop keeps changing.
    var x = a;
    // The same 4 dependent fma per step as math.wgsl. x stays between 1 and 5, far inside the float16 range.
    for (var iteration = 0u; iteration < ITERATIONS; iteration++) {
        x = fma(x, a, 0.5h);
        x = fma(x, a, 0.5h);
        x = fma(x, a, 0.5h);
        x = fma(x, a, 0.5h);
    }
    // Write the result as float32, so the work is kept.
    output[index] = f32(x);
}
