// The cos and sin tables of the rotary positions, made on the CPU in float32 the way Lfm3RotaryEmbedding makes them:
// inv_freq[j] = 1 / theta^(2j / head_dim) for the first `angles` pairs, angle = position * inv_freq[j].

// The two tables, each [positions][angles].
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

// The tables for positions 0 to `positions` - 1.
pub fn rope_table(positions: u32, angles: u32, head_dim: u32, theta: f32) -> RopeTable {
    // PyTorch: 1.0 / (base ** (torch.arange(0, 2 * angles, 2).float() / dim)), all in float32.
    let inverse: Vec<f32> = (0..angles).map(|pair| 1.0 / theta.powf((2 * pair) as f32 / head_dim as f32)).collect();
    // PyTorch: position_ids.float() * inv_freq, then cos and sin in float32.
    let angle = |position: u32, pair: usize| position as f32 * inverse[pair];
    let cos = (0..positions).flat_map(|position| (0..inverse.len()).map(move |pair| angle(position, pair).cos())).collect();
    let sin = (0..positions).flat_map(|position| (0..inverse.len()).map(move |pair| angle(position, pair).sin())).collect();
    RopeTable { cos, sin }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    use super::rope_table;

    // Position 0 has angle 0 for every pair: cos 1 and sin 0, so the first token is not rotated.
    #[test]
    fn position_zero_is_not_rotated() {
        let table = rope_table(2, 32, 128, 1_000_000.0);
        assert!(table.cos[..32].iter().all(|&value| value == 1.0));
        assert!(table.sin[..32].iter().all(|&value| value == 0.0));
        // Pair 0 turns by exactly 1 radian per position, because theta^0 = 1.
        assert_eq!(table.sin[32], 1.0_f32.sin());
    }
}
