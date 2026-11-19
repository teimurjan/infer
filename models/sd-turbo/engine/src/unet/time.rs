// The sinusoidal timestep features of diffusers' Timesteps(320, flip_sin_to_cos=True, downscale_freq_shift=0).
// They depend only on one number, the timestep, so the CPU computes them and writes them into their slot.
// The steps and the f32 rounding follow get_timestep_embedding in diffusers/models/embeddings.py.

// The number of features (block_out_channels[0] in unet/config.json).
pub(super) const TIME_FEATURES: usize = 320;
// Half of the features are cosines and half are sines.
const HALF: usize = TIME_FEATURES / 2;
// The longest period of the waves (max_period in diffusers).
const MAX_PERIOD: f64 = 10000.0;

// The 320 features of `timestep`: cos(t * f_i) for i in 0..160, then sin(t * f_i), with f_i = exp(-ln(10000) * i / 160).
pub(super) fn time_features(timestep: f32) -> Vec<f32> {
    // PyTorch multiplies the Python float -ln(10000) with a float32 tensor, so the number becomes f32 first.
    let log_period = -(MAX_PERIOD.ln() as f32);
    // The frequency of each wave, in f32 as in PyTorch: (-ln(10000) * i) / 160, then exp.
    let arguments: Vec<f32> = (0..HALF).map(|index| timestep * (log_period * index as f32 / HALF as f32).exp()).collect();
    // flip_sin_to_cos puts the cosines first, then the sines.
    arguments.iter().map(|argument| argument.cos()).chain(arguments.iter().map(|argument| argument.sin())).collect()
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function into the tests.
    use super::{HALF, time_features};

    // At timestep 0, every cosine is 1 and every sine is 0. If the halves were in the wrong order, the ones would come last.
    #[test]
    fn cosines_come_first() {
        // The features of timestep 0.
        let features = time_features(0.0);
        // 160 ones, then 160 zeros.
        assert!(features[..HALF].iter().all(|&value| value == 1.0));
        assert!(features[HALF..].iter().all(|&value| value == 0.0));
    }
}
