# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["huggingface_hub"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.
# Copies the two files of the SD-Turbo tokenizer into weights/, where the engine and the web pages read them:
# tokenizer_vocab.json (each token and its id) and tokenizer_merges.txt (the byte pair merges, best first).

# Path builds the file paths.
from pathlib import Path
# shutil copies a file.
import shutil
# hf_hub_download finds one model file in the Hugging Face cache, and downloads it on the first run.
from huggingface_hub import hf_hub_download

# The model that holds the tokenizer.
REPO = "stabilityai/sd-turbo"
# The output folder, next to this script.
WEIGHTS = Path(__file__).parent / "weights"


def main() -> None:
    # Make the folder if it is not there.
    WEIGHTS.mkdir(exist_ok=True)
    # Each source file and its name in weights/.
    for source, target in [("vocab.json", "tokenizer_vocab.json"), ("merges.txt", "tokenizer_merges.txt")]:
        # The cached file of the tokenizer folder of the model.
        path = Path(hf_hub_download(REPO, source, subfolder="tokenizer"))
        # Copy it, and say how large it is.
        shutil.copyfile(path, WEIGHTS / target)
        print(f"{target}: {(WEIGHTS / target).stat().st_size} bytes")


# Run main only when the file runs as a script.
if __name__ == "__main__":
    main()
