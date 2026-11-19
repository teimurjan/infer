// The rank of each merge, by its two symbols.
use std::collections::HashMap;

// Our error type.
use crate::error::EngineError;

// Added to the last symbol of each piece, so the model knows where a word ends.
const END_OF_WORD: &str = "</w>";

// Byte pair encoding: the byte alphabet and the merge rules of merges.txt.
pub(super) struct Bpe {
    // The symbol of each byte value. Every byte has a visible character, so a byte can be a vocabulary key.
    byte_symbols: [char; 256],
    // The rank of each merge, keyed by "left right" as merges.txt writes it. A lower rank merges first.
    ranks: HashMap<String, usize>,
}

impl Bpe {
    // Read merges.txt: a "#version" line, then one merge per line, "left right", best first.
    pub(super) fn new(merges: &str) -> Result<Self, EngineError> {
        // Every line after the first one.
        let rules: Vec<&str> = merges.lines().skip(1).filter(|line| !line.is_empty()).collect();
        // A merge has exactly two symbols. Anything else means a wrong file.
        if let Some((line, rule)) = rules.iter().enumerate().find(|(_, rule)| rule.split(' ').count() != 2) {
            return Err(EngineError::TokenizerFiles(format!("merges line {} is not two symbols: {rule:?}", line + 2)));
        }
        // The rank is the position in the file.
        let ranks = rules.iter().enumerate().map(|(rank, rule)| (rule.to_string(), rank)).collect();
        Ok(Self { byte_symbols: byte_symbols(), ranks })
    }

    // The symbols of one piece after all merges, the last one with "</w>".
    pub(super) fn encode(&self, piece: &str) -> Vec<String> {
        // One symbol per UTF-8 byte.
        let mut symbols: Vec<String> = piece.bytes().map(|byte| self.byte_symbols[usize::from(byte)].to_string()).collect();
        // Mark the end of the word on the last symbol.
        if let Some(last) = symbols.last_mut() {
            last.push_str(END_OF_WORD);
        }
        // Merge the neighbor pair with the lowest rank, the leftmost one on a tie, until no pair has a rule.
        while let Some(index) = self.best_pair(&symbols) {
            let right = symbols.remove(index + 1);
            symbols[index].push_str(&right);
        }
        symbols
    }

    // The position of the left symbol of the pair to merge next, or None if no pair has a rule.
    fn best_pair(&self, symbols: &[String]) -> Option<usize> {
        symbols
            .windows(2)
            .enumerate()
            // The rank of each pair that has a rule.
            .filter_map(|(index, pair)| self.ranks.get(&format!("{} {}", pair[0], pair[1])).map(|&rank| (rank, index)))
            // The lowest rank. A tuple compares the rank first, then the position.
            .min()
            .map(|(_, index)| index)
    }
}

// The byte alphabet of GPT-2 and CLIP (bytes_to_unicode): the printable bytes keep their own character,
// and every other byte gets a character from 256 on, in byte order. So no byte is a space or a control character.
fn byte_symbols() -> [char; 256] {
    // The bytes that keep their own character: "!" to "~", "¡" to "¬", "®" to "ÿ".
    let printable = |byte: u8| matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    // The next free character for the other bytes.
    let mut next = 256u32;
    let mut table = ['\0'; 256];
    for byte in 0..=255u8 {
        // A printable byte maps to itself. The others count up from 256, which are all valid characters.
        let code = if printable(byte) { u32::from(byte) } else { next };
        next += u32::from(!printable(byte));
        table[usize::from(byte)] = char::from_u32(code).expect("256 to 323 are valid characters");
    }
    table
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the parts into the tests.
    use super::{Bpe, byte_symbols};

    // The vocabulary keys use this alphabet, so one wrong byte would make every word with it unknown.
    #[test]
    fn byte_alphabet_matches_gpt2() {
        let table = byte_symbols();
        // A printable byte keeps its character.
        assert_eq!(table[usize::from(b'a')], 'a');
        // The first unprintable byte, 0, is the first new character, 256 ("Ā").
        assert_eq!(table[0], '\u{100}');
        // The space, byte 32, is the 33rd unprintable byte: 256 + 32 = 288 ("Ġ").
        assert_eq!(table[usize::from(b' ')], '\u{120}');
    }

    // The lowest rank must merge first, and the end of word must take part in the merges.
    #[test]
    fn merges_lowest_rank_first() {
        // Rank 0 joins "b" and "c</w>", rank 1 joins "a" and "b". So "abc" gives "a" and "bc</w>", not "ab" and "c</w>".
        let bpe = Bpe::new("#version: 0.2\nb c</w>\na b\n").expect("valid merges");
        assert_eq!(bpe.encode("abc"), vec!["a".to_string(), "bc</w>".to_string()]);
    }
}
