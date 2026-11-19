// erf(x), which GEGLU and GELU need. Rust puts this text in front of their shaders, because WGSL has no include.

// erf(x) by Abramowitz and Stegun, formula 7.1.26. Its largest error is 1.5e-7, close to one f32 step near 1.
// WGSL has no erf function, so we build it.
fn erf(x: f32) -> f32 {
    // The formula is for x >= 0. erf is odd: erf(-x) = -erf(x).
    let a = abs(x);
    // The helper variable of the formula.
    let t = 1.0 / (1.0 + 0.3275911 * a);
    // The polynomial a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5, written in Horner form.
    let polynomial = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    // erf for x >= 0.
    let positive = 1.0 - polynomial * exp(-a * a);
    // Put the sign back.
    return select(positive, -positive, x < 0.0);
}

