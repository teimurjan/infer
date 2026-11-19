// The one Euler step of SD-Turbo, on the CPU. The latent is 16,384 values, so the GPU is not needed.
// The steps and the f32 rounding follow EulerDiscreteScheduler in diffusers, as the pipeline calls it for one step at `sigma`.

// The UNet input: the noise times sigma (the pipeline's latents), divided by sqrt(sigma^2 + 1) (scale_model_input).
pub fn scale_input(noise: &[f32], sigma: f32) -> Vec<f32> {
    // sqrt(sigma^2 + 1), in f32 as in PyTorch.
    let scale = (sigma * sigma + 1.0).sqrt();
    // Each value, in the same order of operations.
    noise.iter().map(|&value| value * sigma / scale).collect()
}

// The latent after the step from `sigma` to 0. `predicted` is the UNet output, the predicted noise.
// diffusers: sample = noise * sigma, x0 = sample - sigma * predicted, derivative = (sample - x0) / sigma, result = sample + derivative * (0 - sigma).
pub fn euler_step(noise: &[f32], predicted: &[f32], sigma: f32) -> Vec<f32> {
    // The step goes from sigma to the next sigma, which is 0.
    let step = 0.0 - sigma;
    // Each value, in the same order of operations.
    noise
        .iter()
        .zip(predicted)
        .map(|(&value, &noise_prediction)| {
            // The latent before the step.
            let sample = value * sigma;
            // The denoised latent that the model predicts.
            let original = sample - sigma * noise_prediction;
            // The slope of the step.
            let derivative = (sample - original) / sigma;
            // One step of Euler's method.
            sample + derivative * step
        })
        .collect()
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the functions into the tests.
    use super::{euler_step, scale_input};

    // Read one tensor of the golden folder. `file` is relative to golden/.
    fn read(file: &str) -> Vec<f32> {
        // The golden folder is in the project folder, one level above engine/.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../golden").join(file);
        // A missing file means a dump script did not run.
        compare::safetensors::read_tensors(&path).unwrap_or_else(|error| panic!("{file}: {error}")).remove(0).values
    }

    // Read one tensor of the float32 golden set.
    fn golden(name: &str) -> Vec<f32> {
        read(&format!("float32/{name}.safetensors"))
    }

    // The starting noise. Both golden sets read this one file, so it is not in float32/.
    fn noise() -> Vec<f32> {
        read("starting_noise.safetensors")
    }

    // The largest difference between two lists.
    fn largest_difference(first: &[f32], second: &[f32]) -> f32 {
        first.iter().zip(second).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
    }

    // The scaled noise must be the UNet input that the PyTorch pipeline made, bit for bit, or every later part starts off.
    #[test]
    fn scaled_noise_is_the_pipeline_unet_input() {
        // The starting noise and the sigma of the one step.
        let (noise, sigma) = (noise(), golden("sigma")[0]);
        // The same values as the pipeline.
        assert_eq!(largest_difference(&scale_input(&noise, sigma), &golden("unet_input")), 0.0);
    }

    // With the PyTorch UNet output, the step must give the PyTorch latent.
    #[test]
    fn step_gives_the_pipeline_latent() {
        // The inputs of the step in the pipeline.
        let (noise, sigma, predicted) = (noise(), golden("sigma")[0], golden("unet_output"));
        // The latent of the pipeline.
        let difference = largest_difference(&euler_step(&noise, &predicted, sigma), &golden("final_latent"));
        // Print it, for the log.
        println!("euler step: largest difference from final_latent {difference:e}");
        // The same operations in f32 must give the same bits.
        assert_eq!(difference, 0.0);
    }
}
