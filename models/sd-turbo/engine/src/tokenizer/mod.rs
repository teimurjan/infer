// bpe.rs holds the byte alphabet and the merges of byte pair encoding.
mod bpe;
// normalize.rs cleans the text: NFC, one space per run of whitespace, lowercase.
mod normalize;
// special.rs cuts the text at the special tokens.
mod special;

// The id of each vocabulary entry.
use std::collections::HashMap;

// Finds the pieces of the text and the runs of whitespace.
use regex::Regex;

// Our error type.
use crate::error::EngineError;

// The parts of this module.
use bpe::Bpe;
use normalize::normalize;
use special::{Part, split_specials};

// The text encoder reads exactly this many ids.
pub const CONTEXT_IDS: usize = 77;
// The first id of every prompt: "<|startoftext|>".
const START: u32 = 49406;
// The id after the last token: "<|endoftext|>". It is also the id of a piece that is not in the vocabulary.
const END: u32 = 49407;
// The pad token "!", id 0. SD-Turbo pads with it, and it is also a special token inside the text.
const PAD: (&str, u32) = ("!", 0);
// The special tokens that the tokenizer finds after it cleans the text.
const CLEANED_SPECIALS: [(&str, u32); 2] = [("<|startoftext|>", START), ("<|endoftext|>", END)];
// The pieces of CLIP: an ending like 's, a run of letters, one digit, or a run of other characters. Whitespace is dropped.
// The special tokens are cut out before, so they are not in this pattern.
const PIECES: &str = r"'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+";

// The SD-Turbo tokenizer (transformers' CLIPTokenizer with the files of stabilityai/sd-turbo). It turns a prompt into 77 ids.
pub struct Tokenizer {
    // The id of each vocabulary entry, from vocab.json.
    vocab: HashMap<String, u32>,
    // The byte pair encoding, from merges.txt.
    bpe: Bpe,
    // The pattern of the pieces.
    pieces: Regex,
    // The pattern of a run of whitespace.
    whitespace: Regex,
}

impl Tokenizer {
    // Read the two files that export_tokenizer.py writes: weights/tokenizer_vocab.json and weights/tokenizer_merges.txt.
    pub fn new(vocab_json: &str, merges: &str) -> Result<Self, EngineError> {
        // The vocabulary is one JSON object: entry to id.
        let vocab: HashMap<String, u32> = serde_json::from_str(vocab_json).map_err(|error| EngineError::TokenizerFiles(format!("vocab: {error}")))?;
        // The special tokens must have the ids that the text encoder was trained with.
        for (entry, id) in [PAD, CLEANED_SPECIALS[0], CLEANED_SPECIALS[1]] {
            if vocab.get(entry) != Some(&id) {
                return Err(EngineError::TokenizerFiles(format!("vocab: {entry:?} must have id {id}, it has {:?}", vocab.get(entry))));
            }
        }
        // The two patterns are fixed, so a failure here is a bug in this file.
        let pieces = Regex::new(PIECES).expect("the piece pattern is valid");
        let whitespace = Regex::new(r"\s+").expect("the whitespace pattern is valid");
        Ok(Self { vocab, bpe: Bpe::new(merges)?, pieces, whitespace })
    }

    // The 77 ids of `prompt`: "<|startoftext|>", at most 75 token ids, "<|endoftext|>", then pad ids up to 77.
    // This is the call of StableDiffusionPipeline: padding="max_length", max_length=77, truncation=True.
    pub fn encode(&self, prompt: &str) -> Vec<u32> {
        // Room for the tokens between the start and the end id.
        let room = CONTEXT_IDS - 2;
        // Start, the tokens that fit, end, then pads.
        std::iter::once(START).chain(self.tokens(prompt).into_iter().take(room)).chain(std::iter::once(END)).chain(std::iter::repeat(PAD.1)).take(CONTEXT_IDS).collect()
    }

    // The token ids of the prompt, without start, end or pads, and without a length limit.
    fn tokens(&self, prompt: &str) -> Vec<u32> {
        // The pad token "!" is found in the raw text, before the cleaning, as in the tokenizers library.
        split_specials(prompt, &[PAD])
            .into_iter()
            .flat_map(|part| match part {
                // The pad token is one id.
                Part::Token(id) => vec![id],
                // Other text is cleaned, then cut at the other special tokens, then encoded.
                Part::Text(text) => self.cleaned_tokens(&normalize(&self.whitespace, text)),
            })
            .collect()
    }

    // The token ids of cleaned text: the special tokens in it, and the pieces between them.
    fn cleaned_tokens(&self, text: &str) -> Vec<u32> {
        split_specials(text, &CLEANED_SPECIALS)
            .into_iter()
            .flat_map(|part| match part {
                // A special token is one id.
                Part::Token(id) => vec![id],
                // Each piece of the text gives one id per symbol after the merges.
                Part::Text(text) => self.pieces.find_iter(text).flat_map(|piece| self.piece_ids(piece.as_str())).collect(),
            })
            .collect()
    }

    // The ids of one piece. A symbol that is not in the vocabulary gets the unknown id, which is the end id.
    fn piece_ids(&self, piece: &str) -> Vec<u32> {
        self.bpe.encode(piece).iter().map(|symbol| self.vocab.get(symbol).copied().unwrap_or(END)).collect()
    }
}
