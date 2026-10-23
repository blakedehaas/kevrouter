"""Download Kev pre-trained decision model weights and adapters from Hugging Face.

Model Targets:
  - Kev 0.8B: jaredpalmer/kev-0.8b (~158 MB)
  - Kev 4B:   jaredpalmer/kev-4b   (~852 MB)
  - Kev 9B:   jaredpalmer/kev-9b   (~573 MB)

Destination Directory:
  models/
    ├── kev-0.8b/
    ├── kev-4b/
    └── kev-9b/
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

KEV_MODELS = {
    "0.8b": {
        "repo_id": "jaredpalmer/kev-0.8b",
        "dir_name": "kev-0.8b",
        "description": "Kev 0.8B Decision Model (Base: Qwen3.5-0.8B)",
        "files": [
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "training_config.json",
        ],
    },
    "4b": {
        "repo_id": "jaredpalmer/kev-4b",
        "dir_name": "kev-4b",
        "description": "Kev 4B Decision Model (Base: Qwen3.5-4B)",
        "files": [
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "training_config.json",
        ],
    },
    "9b": {
        "repo_id": "jaredpalmer/kev-9b",
        "dir_name": "kev-9b",
        "description": "Kev 9B Decision Model (Base: Qwen3.5-9B)",
        "files": [
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "training_config.json",
        ],
    },
}


def format_bytes(size: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def download_file_http(url: str, dest_path: Path, force: bool = False):
    """Download individual file with progress display."""
    if dest_path.exists() and not force:
        if dest_path.stat().st_size > 0:
            return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "KevRouter-ModelDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    with urllib.request.urlopen(req) as resp:
        total_size = resp.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size else 0

        downloaded = 0
        chunk_size = 1024 * 512

        with open(dest_path, "wb") as f_out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)

                elapsed = max(time.time() - start_time, 0.001)
                speed = downloaded / elapsed

                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    status = (
                        f"\r  {dest_path.name:<26} {downloaded / (1024*1024):.1f}/"
                        f"{total_bytes / (1024*1024):.1f} MB [{pct:.1f}%] "
                        f"at {format_bytes(speed)}/s"
                    )
                else:
                    status = (
                        f"\r  {dest_path.name:<26} {downloaded / (1024*1024):.1f} MB "
                        f"at {format_bytes(speed)}/s"
                    )
                sys.stdout.write(status)
                sys.stdout.flush()

        sys.stdout.write("\n")


def download_kev_model(
    model_key: str,
    output_base_dir: Path,
    force: bool = False,
) -> Path:
    """Download a Kev model repo from Hugging Face."""
    info = KEV_MODELS[model_key]
    repo_id = info["repo_id"]
    dest_dir = output_base_dir / info["dir_name"]

    print(f"\n[*] Fetching {info['description']} ({repo_id})...")
    print(f"    Target directory: {dest_dir}")

    if dest_dir.exists() and any(dest_dir.glob("*.safetensors")) and not force:
        print(f"[+] Model weights already present at {dest_dir}. Use --force to re-download.")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Attempt 1: huggingface_hub snapshot_download
    try:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            force_download=force,
        )
        elapsed = time.time() - start_time
        print(f"[+] Successfully downloaded {repo_id} via huggingface_hub in {elapsed:.1f}s.")
        return Path(local_path)
    except ImportError:
        pass
    except Exception as e:
        print(f"[!] huggingface_hub failed ({e}), falling back to direct HTTP streaming...")

    # Attempt 2: Direct HTTP streaming per file
    print(f"[*] Downloading repo files directly from Hugging Face...")
    for filename in info["files"]:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        dest_file = dest_dir / filename
        try:
            download_file_http(url, dest_file, force=force)
        except urllib.error.HTTPError as he:
            if he.code == 404:
                # Optional file not found
                continue
            else:
                print(f"[!] Error downloading {filename}: {he}")
                raise

    elapsed = time.time() - start_time
    print(f"[+] Completed download of {info['dir_name']} in {elapsed:.1f}s.")
    return dest_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download Kev pre-trained decision models from Hugging Face."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="0.8b",
        choices=["0.8b", "4b", "9b", "all"],
        help="Model size to download (default: 0.8b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Local directory to store model checkpoints (default: models/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = (project_root / args.output_dir).resolve()

    print("=== Kev Decision Model Downloader ===")
    print(f"Output directory: {models_dir}")

    targets = list(KEV_MODELS.keys()) if args.size == "all" else [args.size]

    for size_key in targets:
        download_kev_model(size_key, models_dir, force=args.force)

    print("\n[+] Download process completed.")


if __name__ == "__main__":
    main()
