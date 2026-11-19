// The Mersenne Twister MT19937, the random number generator of PyTorch on the CPU (at::mt19937).
// The same seed gives the same 32-bit numbers in the same order as torch.Generator("cpu").manual_seed(seed).

// The size of the state, in 32-bit words.
const STATE_WORDS: usize = 624;
// The distance of the word that each twist mixes in.
const SHIFT_WORDS: usize = 397;
// The twist constant of MT19937.
const MATRIX: u32 = 0x9908_b0df;
// The multiplier of the seeding formula.
const SEED_MULTIPLIER: u32 = 1_812_433_253;

// The generator: its state, and the position of the next word to give.
pub(super) struct Mt19937 {
    // 624 words of state.
    state: [u32; STATE_WORDS],
    // The next word to give. At 624, the state is used up and must be twisted first.
    next: usize,
}

impl Mt19937 {
    // Fill the state from a 32-bit seed, as PyTorch's init_with_uint32 does.
    pub(super) fn new(seed: u32) -> Self {
        let mut state = [0u32; STATE_WORDS];
        state[0] = seed;
        // Each word comes from the one before it. The arithmetic wraps at 32 bits, as in C++.
        for index in 1..STATE_WORDS {
            state[index] = SEED_MULTIPLIER.wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30)).wrapping_add(index as u32);
        }
        // The first call twists the fresh state.
        Self { state, next: STATE_WORDS }
    }

    // The next 32-bit random number.
    pub(super) fn next_u32(&mut self) -> u32 {
        // Make 624 new words when the old ones are used up.
        if self.next == STATE_WORDS {
            self.twist();
        }
        let mut value = self.state[self.next];
        self.next += 1;
        // The tempering of MT19937, which spreads the bits.
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    // Replace every word of the state with a new one.
    fn twist(&mut self) {
        for index in 0..STATE_WORDS {
            // The top bit of this word and the low 31 bits of the next one.
            let mixed = (self.state[index] & 0x8000_0000) | (self.state[(index + 1) % STATE_WORDS] & 0x7fff_ffff);
            // Shift, and mix in the word 397 places ahead. An odd value also mixes in the constant.
            let twisted = (mixed >> 1) ^ if mixed & 1 == 1 { MATRIX } else { 0 };
            self.state[index] = self.state[(index + SHIFT_WORDS) % STATE_WORDS] ^ twisted;
        }
        self.next = 0;
    }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the generator into the tests.
    use super::Mt19937;

    // The C++ standard fixes this value: the 10000th number of std::mt19937 with its default seed 5489 is 4123659995.
    // A wrong seeding formula, twist or tempering gives another number.
    #[test]
    fn ten_thousandth_number_matches_the_cpp_standard() {
        let mut generator = Mt19937::new(5489);
        // Skip the first 9999 numbers.
        (1..10_000).for_each(|_| {
            generator.next_u32();
        });
        assert_eq!(generator.next_u32(), 4_123_659_995);
    }
}
