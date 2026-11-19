// The result of comparing two tensors of the same shape.
pub struct DiffStats {
    // The largest absolute difference between two matching values.
    pub max: f64,
    // The average absolute difference over all values.
    pub mean: f64,
    // The position of the worst value, one index per dimension, for example [0, 12, 3, 4].
    pub worst_index: Vec<usize>,
    // The value in file A at the worst position.
    pub worst_a: f32,
    // The value in file B at the worst position.
    pub worst_b: f32,
}

// Compare two lists of values that have the same shape.
pub fn compare(a: &[f32], b: &[f32], shape: &[usize]) -> DiffStats {
    // Walk through both lists once. Keep the running sum, the worst position and the worst difference.
    let (sum, worst_position, max) = a
        // Pair each value in A with the value at the same position in B.
        .iter()
        // Join with B.
        .zip(b)
        // Add the position number.
        .enumerate()
        // Start with sum 0, worst position 0, worst difference 0.
        .fold((0.0f64, 0usize, 0.0f64), |(sum, worst_position, max), (position, (value_a, value_b))| {
            // Use f64, so that the sum of many small differences stays precise.
            let difference = (f64::from(*value_a) - f64::from(*value_b)).abs();
            // A NaN difference counts as the worst, because NaN means something is broken.
            let is_worse = difference > max || (difference.is_nan() && !max.is_nan());
            // Keep the new worst, or keep the old one.
            if is_worse { (sum + difference, position, difference) } else { (sum + difference, worst_position, max) }
        });
    // Avoid dividing by zero for an empty tensor.
    let count = a.len().max(1);
    // Build the result. Empty tensors report 0 for everything.
    DiffStats {
        // The largest difference.
        max,
        // The average difference.
        mean: sum / count as f64,
        // Turn the flat position into one index per dimension.
        worst_index: unravel(worst_position, shape),
        // The A value at the worst position (0 if the tensor is empty).
        worst_a: a.get(worst_position).copied().unwrap_or(0.0),
        // The B value at the worst position (0 if the tensor is empty).
        worst_b: b.get(worst_position).copied().unwrap_or(0.0),
    }
}

// Turn a flat position into one index per dimension. Row-major: the last dimension changes fastest.
// Example: shape [2, 3], position 4 -> [1, 1].
fn unravel(position: usize, shape: &[usize]) -> Vec<usize> {
    // Start from the last dimension and work to the first.
    let mut index: Vec<usize> = shape
        // Go through the dimensions from last to first.
        .iter()
        // Reverse the order.
        .rev()
        // `rest` is the part of the position not used yet.
        .scan(position, |rest, &size| {
            // max(1) avoids dividing by zero for a dimension of size 0.
            let size = size.max(1);
            // The remainder is the index in this dimension.
            let index_here = *rest % size;
            // The quotient goes on to the next dimension.
            *rest /= size;
            // Emit the index for this dimension.
            Some(index_here)
        })
        // Collect the indexes, last dimension first.
        .collect();
    // Put the first dimension first again.
    index.reverse();
    // Return the indexes.
    index
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the functions into the tests.
    use super::{compare, unravel};

    // The flat position must turn into the right row and column.
    #[test]
    fn unravels_row_major_positions() {
        // In a 2 x 3 tensor, position 4 is row 1, column 1.
        assert_eq!(unravel(4, &[2, 3]), vec![1, 1]);
        // Position 5 is the last element.
        assert_eq!(unravel(5, &[2, 3]), vec![1, 2]);
    }

    // The worst element, the largest difference and the average must all be right.
    #[test]
    fn finds_worst_element_and_average() {
        // A 2 x 2 tensor with differences 0, 0.5, 0, 2.
        let stats = compare(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.5, 3.0, 2.0], &[2, 2]);
        // The largest difference is 2 at row 1, column 1.
        assert_eq!(stats.max, 2.0);
        // Row 1, column 1.
        assert_eq!(stats.worst_index, vec![1, 1]);
        // The average is (0 + 0.5 + 0 + 2) / 4.
        assert_eq!(stats.mean, 0.625);
    }

    // A NaN must be reported as the worst element.
    #[test]
    fn reports_nan_as_worst() {
        // The second value in B is NaN.
        let stats = compare(&[1.0, 2.0, 3.0], &[1.5, f32::NAN, 3.0], &[3]);
        // The largest difference is NaN.
        assert!(stats.max.is_nan());
        // The NaN is at position 1.
        assert_eq!(stats.worst_index, vec![1]);
    }
}
