// The bit width of a packed weight, in the format of pack_weights.py.

// How many bits each packed weight has. The kernels need 4 * bits to fill a word exactly, so only 4 and 8 work.
#[derive(Clone, Copy)]
pub enum WeightBits {
    // Integers from -8 to 7.
    Four,
    // Integers from -128 to 127.
    Eight,
}

impl WeightBits {
    // The number of bits, as the kernels read it.
    pub fn value(self) -> u32 {
        // One number per variant.
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}
