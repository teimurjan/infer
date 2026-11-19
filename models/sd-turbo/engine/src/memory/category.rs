// The kinds of GPU buffer that the memory counter keeps apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCategory {
    // Model weights: tensors from a weight file, which kernels only read.
    Weights,
    // Values that flow between kernels: layer inputs and outputs, and scratch space.
    Activations,
    // Small constant blocks for shaders, such as tensor shapes.
    Uniforms,
    // Staging buffers that copy a result back to the CPU.
    ReadBack,
}

impl MemoryCategory {
    // Every category, in the order the report prints them. The counter keeps one number per entry.
    pub const ALL: [Self; 4] = [Self::Weights, Self::Activations, Self::Uniforms, Self::ReadBack];

    // The position of this category in ALL. The variants are declared in the same order, so the variant number is the position.
    pub(crate) fn slot(self) -> usize {
        self as usize
    }

    // The word that the report prints.
    pub fn name(self) -> &'static str {
        // One word per category.
        match self {
            Self::Weights => "weights",
            Self::Activations => "activations",
            Self::Uniforms => "uniforms",
            Self::ReadBack => "read-back",
        }
    }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the category into the tests.
    use super::MemoryCategory;

    // If someone reorders the variants but not ALL, two categories would share one number in the counter.
    #[test]
    fn each_category_has_its_own_slot() {
        // Walk ALL with the position of each entry.
        for (position, category) in MemoryCategory::ALL.into_iter().enumerate() {
            // The slot must be the position.
            assert_eq!(category.slot(), position, "{}", category.name());
        }
    }
}
