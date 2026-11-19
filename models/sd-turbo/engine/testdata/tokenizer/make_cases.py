# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["transformers", "tokenizers"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.
# Writes cases.json: prompts and the 77 token ids that the SD-Turbo tokenizer of transformers gives for each,
# with the same call as StableDiffusionPipeline (pad to 77, cut at 77). The engine's tokenizer test compares with it.

# json writes the cases.
import json
# Path finds the folder of this script.
from pathlib import Path
# transformers gives the reference tokenizer and its version.
import transformers
from transformers import CLIPTokenizer

# The prompts. Each one tests a rule: case, punctuation, the "!" pad token, digits, apostrophes, accents,
# decomposed accents (NFC), other scripts, emoji, whitespace, special tokens in the text, and a prompt longer than 77 tokens.
PROMPTS = [
    "a photo of a cat sitting on a wooden table",
    "A red sports car on a mountain road at sunset!",
    "",
    "A Photo Of A CAT",
    "a cat, a dog; a bird: and a fish.",
    "hello!",
    "wow!!! amazing!",
    "a!b",
    "1024x768 at 60fps in 2024",
    "it's a dog's life, don't you think? we'll see, they've gone, i'm here, you're there, he'd go",
    "rock'n'roll ''quoted'' -'s",
    "café crème brûlée, naïve façade",
    "café with a decomposed accent",
    "Straße in München",
    "İstanbul at night",
    "東京の猫",
    "un gato en la mañana, кошка на столе, القطة",
    "🐱 cat with emoji 🎨🖌️",
    "  a\tcat\n\non   a  mat  ",
    "<|endoftext|> hello <|startoftext|>",
    "<|STARTOFTEXT|>shout",
    "#hashtag @user https://example.com/a?b=c&d=e",
    "...---... (brackets) [square] {curly} <angle>",
    "ancient greek αβγ and the number ½ and ² and ٣",
    "zero​width and no-break space",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    " ".join(["a beautiful landscape painting of mountains"] * 20),
]

# The files go next to this script.
FOLDER = Path(__file__).parent


def main() -> None:
    # The tokenizer of the model, as StableDiffusionPipeline loads it.
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
    # The same call as the pipeline: pad to 77, cut at 77.
    cases = [{"prompt": prompt, "ids": tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids} for prompt in PROMPTS]
    # Write the cases, with the version that made them.
    (FOLDER / "cases.json").write_text(json.dumps({"transformers": transformers.__version__, "cases": cases}, ensure_ascii=False, indent=1))
    # One line per case, to read in the log.
    for case in cases:
        print(f"{case['prompt'][:40]!r}: {case['ids'][:14]}")
    print(f"{len(cases)} cases, transformers {transformers.__version__}")


# Run main only when the file runs as a script.
if __name__ == "__main__":
    main()
