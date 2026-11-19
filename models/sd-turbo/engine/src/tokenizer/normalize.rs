// Composes characters, as the NFC normalizer of the tokenizers library does.
use unicode_normalization::UnicodeNormalization;
// Finds runs of whitespace.
use regex::Regex;

// Clean one part of a prompt in the order of the SD-Turbo tokenizer (transformers' CLIPTokenizer):
// NFC (a letter plus a separate accent becomes one letter), every run of whitespace becomes one space, then lowercase.
pub(super) fn normalize(whitespace: &Regex, text: &str) -> String {
    // NFC first, so "e" plus a combining accent is one character before the next rules see it.
    let composed: String = text.nfc().collect();
    // One space for each run of spaces, tabs, new lines and other Unicode whitespace.
    let spaced = whitespace.replace_all(&composed, " ");
    // Lowercase, character by character, as Rust's to_lowercase does it. The tokenizers library uses the same function.
    spaced.to_lowercase()
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the regex into the tests.
    use super::{Regex, normalize};

    // Each rule changes the ids, so each must hold: the accent is joined, the whitespace is one space, the case is low.
    #[test]
    fn composes_then_collapses_whitespace_then_lowercases() {
        // The same pattern as the tokenizer uses.
        let whitespace = Regex::new(r"\s+").expect("a valid pattern");
        // "e" plus U+0301 becomes "é". Tab and new lines become one space each run. Capitals become small.
        assert_eq!(normalize(&whitespace, "Cafe\u{301}\t\n  A"), "café a");
    }
}
