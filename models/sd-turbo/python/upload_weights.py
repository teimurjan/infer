# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["huggingface_hub>=2.4,<3"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.
# Uploads the weight files of the public demo to a Hugging Face model repository, so the deployed site can fetch them.
# Vercel cannot serve these files: one deployment holds at most 100 MB on Hobby, and the files of the demo are about 2.2 GB.
# Usage: uv run python/upload_weights.py <user>/<repo> [--dry-run]
# Log in once before the first upload with `hf auth login`, or set HF_TOKEN.

# argparse reads the command line.
import argparse
# os sets the upload settings of hf_xet, the upload library of huggingface_hub.
import os
# Path builds the file paths.
from pathlib import Path

# hf_xet uploads each file in pieces of about 67 MB, and it stops a piece that takes longer than 300 s.
# At 0.17 MB/s, the upload speed measured on 2026-11-19, one piece takes about 394 s, so every piece failed.
# 1800 s gives one piece time at speeds down to about 0.04 MB/s. hf_xet reads these values when it starts, so they come before the import.
os.environ.setdefault("HF_XET_CLIENT_READ_TIMEOUT", "1800s")
# One piece at a time: two pieces at once share the same slow connection, so each one takes twice as long.
os.environ.setdefault("HF_XET_FIXED_UPLOAD_CONCURRENCY", "1")

# HfApi creates the repository and uploads the files.
from huggingface_hub import HfApi

# The folder of the weight files, next to the python/ folder.
WEIGHTS = Path(__file__).parent.parent / "weights"

# The files that the entry page fetches, from client/pipeline.ts and client/engine.ts.
# Each weight is a .bin file with the tensors and a .json file that is its index.
DEMO_FILES = [
    "text_encoder_q8.bin",
    "text_encoder_q8.json",
    "unet_q8.bin",
    "unet_q8.json",
    "taesd.bin",
    "taesd.json",
    "tokenizer_vocab.json",
    "tokenizer_merges.txt",
]


def read_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the demo weights to a Hugging Face model repository.")
    # The repository, for example "teimurjan/infer-sd-turbo". The script creates it if it does not exist.
    parser.add_argument("repo", help="the repository id, <user>/<repo>")
    # --dry-run lists the files and their sizes, and uploads nothing.
    parser.add_argument("--dry-run", action="store_true", help="list the files and upload nothing")
    return parser.parse_args()


def find_missing_files() -> list[str]:
    # A missing file stops the script before the upload, so the repository never holds half a model.
    return [name for name in DEMO_FILES if not (WEIGHTS / name).is_file()]


def print_files() -> None:
    # One line for each file, with its size in megabytes (1 MB = 1,000,000 bytes).
    for name in DEMO_FILES:
        print(f"{name}: {(WEIGHTS / name).stat().st_size / 1e6:.1f} MB")
    # The total, so you know how long the upload can take.
    total = sum((WEIGHTS / name).stat().st_size for name in DEMO_FILES)
    print(f"total: {total / 1e6:.1f} MB in {len(DEMO_FILES)} files")


def upload(repo: str) -> None:
    api = HfApi()
    # The repository is public, because a browser fetches the files without a token. exist_ok: a second run reuses it.
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    # In huggingface_hub 2.x, upload_folder uploads in parts and can resume: if the connection drops, run the same command again.
    # allow_patterns limits the upload to the demo files, so unet.bin (3.5 GB) and the other files stay on the Mac.
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=WEIGHTS, allow_patterns=DEMO_FILES, commit_message="Upload the SD-Turbo demo weights")


def main() -> None:
    arguments = read_arguments()
    missing = find_missing_files()
    if missing:
        # The README of the model has the `uv run` commands that make these files.
        raise SystemExit(f"missing in {WEIGHTS}: {', '.join(missing)}")
    print_files()
    if arguments.dry_run:
        return
    upload(arguments.repo)
    # The base URL that client/urls.ts will use for weight files in production.
    print(f"uploaded. base URL: https://huggingface.co/{arguments.repo}/resolve/main/")


if __name__ == "__main__":
    main()
