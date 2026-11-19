// Our error type.
use crate::error::EngineError;
// The shared tensor length check.
use crate::checks::check_length;

// Red, green and blue planes in the input.
const RGB: usize = 3;
// Alpha 255 means fully opaque.
const OPAQUE: u8 = 255;

// Turn an image [3][height][width] with values in [-1, 1] into RGBA bytes (4 per pixel), one pixel after the other, as a canvas reads them.
// This is the diffusers postprocess: (x / 2 + 0.5), clamp to [0, 1], times 255, round.
pub fn image_to_rgba(values: &[f32], height: usize, width: usize) -> Result<Vec<u8>, EngineError> {
    // Pixels in one plane.
    let pixels = height * width;
    // A wrong length would mix up the planes.
    check_length("image", values, RGB * pixels)?;
    // The red, green and blue planes, one after the other.
    let (red, rest) = values.split_at(pixels);
    let (green, blue) = rest.split_at(pixels);
    // For each pixel, 4 bytes.
    Ok((0..pixels).flat_map(|pixel| [to_byte(red[pixel]), to_byte(green[pixel]), to_byte(blue[pixel]), OPAQUE]).collect())
}

// One value in [-1, 1] to a byte in 0..=255.
fn to_byte(value: f32) -> u8 {
    // [-1, 1] to [0, 1], clamped, scaled and rounded. `as u8` turns NaN into 0.
    ((value / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function into the tests.
    use super::image_to_rgba;

    // The planes must end up interleaved per pixel, with the diffusers rounding and clamping.
    #[test]
    fn interleaves_planes_and_scales() {
        // A 1 x 2 image. Red: -1 and 1. Green: 0 and 3 (clamped). Blue: -5 (clamped) and 0.5.
        let values = [-1.0, 1.0, 0.0, 3.0, -5.0, 0.5];
        // Pixel 1: red 0, green 127.5 rounds to 128, blue 0. Pixel 2: red 255, green 255, blue 191.25 rounds to 191.
        assert_eq!(image_to_rgba(&values, 1, 2).unwrap(), vec![0, 128, 0, 255, 255, 255, 191, 255]);
    }

    // A list that is not 3 planes long must fail.
    #[test]
    fn refuses_a_wrong_length() {
        // 5 values cannot be 3 planes of 2 pixels.
        assert!(image_to_rgba(&[0.0; 5], 1, 2).is_err());
    }
}
