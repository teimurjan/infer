// The buffer we read. One vec4<f32> is 4 f32 values, 16 bytes, so each invocation moves 16 bytes.
@group(0) @binding(0) var<storage, read> source: array<vec4<f32>>;
// The buffer we write.
@group(0) @binding(1) var<storage, read_write> destination: array<vec4<f32>>;

// 256 invocations per workgroup, the most that WebGPU allows by default. Rust's WORKGROUP_SIZE must match.
@compute @workgroup_size(256)
// global_invocation_id is the position of this invocation among all invocations of the dispatch.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // We dispatch along x only, so x alone is the index of this invocation's vec4.
    let index = id.x;
    // Invocations past the end do nothing. With the sizes in copy.rs, none are past the end.
    if (index >= arrayLength(&source)) {
        return;
    }
    // One read of 16 bytes and one write of 16 bytes.
    destination[index] = source[index];
}
