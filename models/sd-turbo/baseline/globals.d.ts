// index.html loads transformers.js from a CDN and puts AutoTokenizer on window.
// These types describe only the parts that index.ts uses.

interface TokenizerOptions {
    padding: boolean;
    max_length: number;
    truncation: boolean;
    // false makes the tokenizer return plain number arrays, not tensors.
    return_tensor: false;
}

interface Tokenizer {
    // A tokenizer is a function: text in, token ids out.
    (text: string, options: TokenizerOptions): Promise<{ input_ids: number[] }>;
    pad_token_id: number;
}

interface AutoTokenizerStatic {
    from_pretrained(name: string): Promise<Tokenizer>;
}

declare var AutoTokenizer: AutoTokenizerStatic;
