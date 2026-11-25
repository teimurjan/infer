// Convert the 16 bits of a float16 number into a normal f32.
// float16 layout: 1 sign bit, 5 exponent bits, 10 mantissa (fraction) bits.
// f32 layout:     1 sign bit, 8 exponent bits, 23 mantissa bits.
pub fn f16_to_f32(bits: u16) -> f32 {
    // Move the sign bit from position 15 to position 31.
    let sign = u32::from(bits >> 15) << 31;
    // The 5 exponent bits.
    let exponent = u32::from((bits >> 10) & 0x1f);
    // The 10 mantissa bits.
    let mantissa = bits & 0x3ff;
    // Exponent 0 and exponent 31 are special cases. Every other exponent is a normal number.
    match exponent {
        // Exponent 0 means zero or a very small "subnormal" number: mantissa x 2^-24.
        0 => {
            // f32 can hold this value exactly.
            let magnitude = f32::from(mantissa) * 2f32.powi(-24);
            // Put the sign back.
            if sign == 0 { magnitude } else { -magnitude }
        }
        // Exponent 31 means infinity (mantissa 0) or NaN (mantissa not 0). f32 uses exponent 255 for the same.
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (u32::from(mantissa) << 13)),
        // float16 stores exponent + 15 and f32 stores exponent + 127, so add 112. Widen the mantissa from 10 to 23 bits.
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (u32::from(mantissa) << 13)),
    }
}

// Tests run with `cargo test`. Each one checks a float16 value whose bits we know.
#[cfg(test)]
mod tests {
    // Bring f16_to_f32 into the tests.
    use super::f16_to_f32;

    // Normal numbers: 1.0, -2.0 and the largest float16 value.
    #[test]
    fn converts_normal_numbers() {
        // 0x3C00 is 1.0.
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        // 0xC000 is -2.0.
        assert_eq!(f16_to_f32(0xc000), -2.0);
        // 0x7BFF is 65504, the largest finite float16.
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    // Zero and the smallest subnormal number.
    #[test]
    fn converts_zero_and_subnormals() {
        // All bits 0 is +0.0.
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0x0001 is 2^-24, the smallest positive float16.
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    // Infinity and NaN.
    #[test]
    fn converts_infinity_and_nan() {
        // 0x7C00 is +infinity.
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        // 0x7E00 is a NaN.
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
