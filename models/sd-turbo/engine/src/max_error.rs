// Our error type.
use crate::error::EngineError;

// The largest absolute difference between two lists of the same length.
// A NaN difference wins over every number, because NaN means the kernel is broken.
pub fn max_abs_error(actual: &[f32], expected: &[f32]) -> Result<f32, EngineError> {
    // Lists of different length cannot be compared value by value.
    if actual.len() != expected.len() {
        // Stop and say both lengths.
        return Err(EngineError::LengthMismatch { actual: actual.len(), expected: expected.len() });
    }
    // Walk both lists together and keep the worst difference seen so far.
    let worst = actual.iter().zip(expected).map(|(got, want)| (got - want).abs()).fold(0.0f32, |worst, difference| {
        // Once worst is NaN, it stays NaN.
        if worst.is_nan() {
            worst
        // A new NaN or a bigger difference becomes the new worst.
        } else if difference.is_nan() || difference > worst {
            difference
        // Otherwise keep the old worst.
        } else {
            worst
        }
    });
    // The largest difference, or 0 for two empty lists.
    Ok(worst)
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the error into the tests.
    use super::{EngineError, max_abs_error};

    // The result is the largest difference, not the first and not the last.
    #[test]
    fn finds_the_largest_difference() {
        // Differences are 0.5, 2.0 and 0.25.
        let worst = max_abs_error(&[1.0, 5.0, 0.0], &[1.5, 3.0, 0.25]).unwrap();
        // 2.0 is the largest.
        assert_eq!(worst, 2.0);
    }

    // A NaN in the GPU output must not hide behind a normal number.
    #[test]
    fn nan_counts_as_the_worst() {
        // The middle value is NaN, and the last difference is 9.0.
        let worst = max_abs_error(&[0.0, f32::NAN, 10.0], &[0.0, 0.0, 1.0]).unwrap();
        // NaN wins.
        assert!(worst.is_nan());
    }

    // Lists of different length are an error, not a silent partial compare.
    #[test]
    fn rejects_different_lengths() {
        // One value against two values.
        let result = max_abs_error(&[1.0], &[1.0, 2.0]);
        // The error says both lengths.
        assert!(matches!(result, Err(EngineError::LengthMismatch { actual: 1, expected: 2 })));
    }
}
