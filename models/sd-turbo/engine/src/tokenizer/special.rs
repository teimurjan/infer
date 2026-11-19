// One part of a prompt: text to encode, or a special token that is already an id.
#[derive(Debug, PartialEq)]
pub(super) enum Part<'a> {
    // Text that goes on to the next rules.
    Text(&'a str),
    // A special token, for example "!" (the pad token) or "<|endoftext|>".
    Token(u32),
}

// Cut `text` at every special token of `specials`, from left to right. Where two specials start at the same place,
// the longer one wins, as in the tokenizers library. Empty text between two specials is left out.
pub(super) fn split_specials<'a>(text: &'a str, specials: &[(&str, u32)]) -> Vec<Part<'a>> {
    // The parts so far, and the start of the text that is not in a part yet.
    let mut parts = Vec::new();
    let mut start = 0;
    // The byte position that is checked now.
    let mut position = 0;
    while position < text.len() {
        // The longest special that starts here, if any.
        let found = specials.iter().filter(|(special, _)| text[position..].starts_with(special)).max_by_key(|(special, _)| special.len());
        match found {
            Some(&(special, id)) => {
                // The text before the special, if any, then the special.
                if start < position {
                    parts.push(Part::Text(&text[start..position]));
                }
                parts.push(Part::Token(id));
                // Go on after the special.
                position += special.len();
                start = position;
            }
            // No special here: go to the next character, not the next byte, so the slices stay valid UTF-8.
            None => position += text[position..].chars().next().map_or(1, char::len_utf8),
        }
    }
    // The text after the last special.
    if start < text.len() {
        parts.push(Part::Text(&text[start..]));
    }
    parts
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the part type into the tests.
    use super::{Part, split_specials};

    // A wrong cut would merge a special token into a word, or drop the text next to it.
    #[test]
    fn cuts_at_every_special_and_prefers_the_longest() {
        // "!" is a special here, and "!!" is a longer one, to test the longest match.
        let specials = [("!", 0), ("!!", 9)];
        assert_eq!(split_specials("a!b!!!", &specials), vec![Part::Text("a"), Part::Token(0), Part::Text("b"), Part::Token(9), Part::Token(0)]);
        // A text with no special is one part. Multi-byte characters are kept whole.
        assert_eq!(split_specials("東京", &specials), vec![Part::Text("東京")]);
    }
}
