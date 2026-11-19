// A kernel ready to record, and a kernel with its buffers.
use crate::dispatch::{BoundKernel, Dispatch};

// How a tensor of the plan maps to the PyTorch shape of its golden file.
#[derive(Clone, Copy)]
pub(crate) enum Layout {
    // [values, 1, 1] in the plan, (1, values) in PyTorch. For example the time embedding.
    Vector,
    // [channels, height, width] in the plan, (1, channels, height, width) in PyTorch.
    Image,
    // [tokens, channels, 1] in the plan, (1, tokens, channels) in PyTorch. The tensors inside a transformer.
    Tokens,
}

impl Layout {
    // The PyTorch shape, with the batch of 1 in front.
    pub(crate) fn torch_shape(self, [first, second, third]: [u32; 3]) -> Vec<usize> {
        // Pick the sizes that this layout uses.
        let sizes = match self {
            // One size.
            Self::Vector => vec![first],
            // Three sizes.
            Self::Image => vec![first, second, third],
            // Two sizes.
            Self::Tokens => vec![first, second],
        };
        // The batch of 1, then the sizes.
        std::iter::once(1).chain(sizes.into_iter().map(|size| size as usize)).collect()
    }
}

// A tensor that a step shows to the caller after its submit.
pub(crate) struct Check {
    // The name of the golden file, for example "down_blocks.0.resnets.0".
    pub(crate) golden: String,
    // The tensor of the plan that holds the result, for example "down_blocks.0.resnets.0.conv2".
    pub(crate) tensor: String,
    // How its plan size maps to the golden shape.
    pub(crate) layout: Layout,
}

// One submit of the UNet stage: its kernels in run order, and the tensors to check after it.
// `prepare` makes every step once. `run` submits them in order.
pub(crate) struct Step {
    // The name of the submit, for the 0.25 s error message.
    pub(crate) label: String,
    // The kernels. A later kernel sees the writes of an earlier one.
    pub(crate) kernels: Vec<BoundKernel>,
    // The tensors to check after the submit. Most steps have none or one. The last step of a level can end two parts.
    pub(crate) checks: Vec<Check>,
}

impl Step {
    // A step without a check.
    pub(crate) fn new(label: impl Into<String>, kernels: Vec<BoundKernel>) -> Self {
        Self { label: label.into(), kernels, checks: Vec::new() }
    }

    // The same step, which also shows `tensor` as the golden `golden` after its submit.
    pub(crate) fn checked(mut self, golden: impl Into<String>, tensor: impl Into<String>, layout: Layout) -> Self {
        // Add the check after the ones the step has.
        self.checks.push(Check { golden: golden.into(), tensor: tensor.into(), layout });
        // The step.
        self
    }

    // Every dispatch of every kernel, in order. A clone of a dispatch shares its pipeline and bind group.
    pub(crate) fn dispatches(&self) -> Vec<Dispatch> {
        self.kernels.iter().flat_map(|kernel| kernel.dispatches().iter().cloned()).collect()
    }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the layout into the tests.
    use super::Layout;

    // A wrong shape would make the golden compare fail on a correct result, or compare the wrong values.
    #[test]
    fn plan_sizes_map_to_pytorch_shapes() {
        // The time embedding.
        assert_eq!(Layout::Vector.torch_shape([1280, 1, 1]), vec![1, 1280]);
        // A block output.
        assert_eq!(Layout::Image.torch_shape([320, 64, 64]), vec![1, 320, 64, 64]);
        // A tensor inside a transformer.
        assert_eq!(Layout::Tokens.torch_shape([4096, 320, 1]), vec![1, 4096, 320]);
    }
}
