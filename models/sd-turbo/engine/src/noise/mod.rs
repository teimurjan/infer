// mt19937.rs holds the random number generator of PyTorch on the CPU.
mod mt19937;

// The generator.
use mt19937::Mt19937;

// The values of the starting noise of one 512 x 512 image: [1, 4, 64, 64].
pub const NOISE_VALUES: usize = 4 * 64 * 64;
// PyTorch turns uniform numbers into normal numbers 16 at a time: 8 pairs, value j with value j + 8.
const BLOCK: usize = 16;
// A float32 has 24 bits of precision, so a uniform number uses the low 24 bits of a random number.
const UNIFORM_BITS: u32 = 24;

// The starting noise of `seed`: the same values as torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed)),
// which is how diffusers makes the latents on the CPU. So an image of the engine and of diffusers with the same seed start alike.
// PyTorch's normal_fill does this: first one uniform number in [0, 1) per value, then Box-Muller on each block of 16.
// The count is a multiple of 16, so PyTorch's path for a remainder is not needed.
pub fn starting_noise(seed: u32) -> Vec<f32> {
    // The generator of this seed.
    let mut generator = Mt19937::new(seed);
    // One uniform number per value, in order.
    let uniform: Vec<f32> = (0..NOISE_VALUES).map(|_| uniform_from(generator.next_u32())).collect();
    // Each block of 16 uniform numbers becomes 16 normal numbers.
    uniform.chunks_exact(BLOCK).flat_map(box_muller).collect()
}

// A uniform number in [0, 1) from the low 24 bits of a random number, as PyTorch's uniform_real does for float.
fn uniform_from(random: u32) -> f32 {
    // A whole number below 2^24, times 2^-24. Both steps are exact in f32.
    (random & ((1 << UNIFORM_BITS) - 1)) as f32 / (1u32 << UNIFORM_BITS) as f32
}

// 16 normal numbers from 16 uniform numbers, as PyTorch's normal_fill_16: value j and value j + 8 form one pair.
// The math is in f32, as in PyTorch, except the angle, which PyTorch computes in f64 and then rounds to f32.
fn box_muller(uniform: &[f32]) -> [f32; BLOCK] {
    // Half of the block: the first values of each pair.
    let half = BLOCK / 2;
    let mut normal = [0.0f32; BLOCK];
    for index in 0..half {
        // 1 - u is in (0, 1], so its logarithm is finite.
        let radius = (-2.0 * (1.0 - uniform[index]).ln()).sqrt();
        // The angle in f64, rounded to f32, as in PyTorch.
        let angle = (2.0 * std::f64::consts::PI * f64::from(uniform[index + half])) as f32;
        // The pair: cosine first, sine second.
        normal[index] = radius * angle.cos();
        normal[index + half] = radius * angle.sin();
    }
    normal
}
