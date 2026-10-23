"""Download and extract the LLMRouterBench dataset from Hugging Face.

Dataset Source: NPULH/LLMRouterBench (Hugging Face)
Archive: bench-release.tar.gz (~1.28 GB)

Expected Datasets:
  - Primary pool (9 datasets): AIME, LiveMathBench, LiveCodeBench, SWE-Bench,
    GPQA, HLE, MMLU-Pro, SimpleQA, ArenaHard.
  - OOD pool: tau2-Bench.
"""

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import urllib.request
import urllib.error

# Hugging Face Dataset configuration
HF_REPO_ID = "NPULH/LLMRouterBench"
HF_FILENAME = "bench-release.tar.gz"
DIRECT_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_FILENAME}"

# Benchmark datasets referenced in the KevRouter project proposal
PRIMARY_DATASETS = [
    "aime",
    "livemathbench",
    "livecodebench",
    "swebench",
    "swe-bench",
    "gpqa",
    "hle",
    "mmlupro",
    "mmlu-pro",
    "simpleqa",
    "arenahard",
    "arena-hard",
]
OOD_DATASETS = ["tau2", "tau2bench", "tau2-bench", "tau_2"]


def format_bytes(size: float) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    """Download file with fallback to huggingface_hub or urllib streaming."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        file_size = dest_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # If at least 100MB exists
            print(f"[+] Archive already exists at {dest_path} ({format_bytes(file_size)}).")
            print("    Use --force-download to re-download.")
            return dest_path

    # Attempt 1: huggingface_hub if installed
    try:
        from huggingface_hub import hf_hub_download
        print(f"[*] Downloading {HF_FILENAME} via huggingface_hub...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir=str(dest_path.parent),
            local_dir_use_symlinks=False,
            force_download=force,
        )
        downloaded_path = Path(downloaded)
        print(f"[+] Download complete: {downloaded_path} ({format_bytes(downloaded_path.stat().st_size)})")
        return downloaded_path
    except ImportError:
        print("[i] huggingface_hub not installed; falling back to streaming HTTP download...")
    except Exception as e:
        print(f"[!] huggingface_hub failed ({e}); falling back to streaming HTTP download...")

    # Attempt 2: Standard urllib streaming with progress
    print(f"[*] Downloading from: {url}")
    print(f"[*] Destination: {dest_path}")

    headers = {"User-Agent": "KevRouter-DataIngest/1.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else 0

            downloaded_bytes = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            with open(dest_path, "wb") as f_out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded_bytes += len(chunk)

                    # Progress display
                    elapsed = max(time.time() - start_time, 0.001)
                    speed = downloaded_bytes / elapsed
                    if total_bytes > 0:
                        percent = (downloaded_bytes / total_bytes) * 100
                        status = (
                            f"\rDownloading: {downloaded_bytes / (1024*1024):.1f}/"
                            f"{total_bytes / (1024*1024):.1f} MB [{percent:.1f}%] "
                            f"at {format_bytes(speed)}/s"
                        )
                    else:
                        status = (
                            f"\rDownloading: {downloaded_bytes / (1024*1024):.1f} MB "
                            f"at {format_bytes(speed)}/s"
                        )
                    sys.stdout.write(status)
                    sys.stdout.flush()

            sys.stdout.write("\n")
            print(f"[+] Successfully downloaded {format_bytes(downloaded_bytes)} to {dest_path}")
            return dest_path

    except urllib.error.URLError as err:
        print(f"[!] Network error downloading {url}: {err}")
        if dest_path.exists():
            dest_path.unlink()
        raise


def extract_archive(archive_path: Path, extract_to: Path, force: bool = False) -> Path:
    """Extract .tar.gz archive with verification."""
    extract_to.mkdir(parents=True, exist_ok=True)

    # Check if bench folder already exists and contains files
    bench_dir = extract_to / "bench"
    if bench_dir.exists() and any(bench_dir.iterdir()) and not force:
        print(f"[+] Extracted benchmark directory already found at {bench_dir}.")
        print("    Use --force-extract to re-extract.")
        return bench_dir

    print(f"[*] Extracting {archive_path.name} to {extract_to}...")
    start_time = time.time()

    with tarfile.open(archive_path, "r:gz") as tar:
        # Check members
        members = tar.getmembers()
        print(f"    Found {len(members):,} files/directories in archive.")

        # Extract safely
        tar.extractall(path=extract_to)

    elapsed = time.time() - start_time
    print(f"[+] Extraction finished in {elapsed:.1f}s.")

    # Locate extracted bench root
    if (extract_to / "bench").exists():
        return extract_to / "bench"
    return extract_to


def inspect_benchmark_data(bench_dir: Path, output_manifest: Optional[Path] = None) -> Dict:
    """Inspect the extracted directory, map datasets, models, and counts."""
    print(f"[*] Inspecting benchmark data in {bench_dir}...")

    datasets_found: Dict[str, Dict] = {}
    all_models: Set[str] = set()
    total_json_files = 0
    sample_record = None

    # Search for all json files in bench_dir
    json_files = list(bench_dir.rglob("*.json"))
    total_json_files = len(json_files)
    print(f"    Found {total_json_files:,} total JSON files.")

    for jf in json_files:
        # Path convention usually: .../bench/{dataset_name}/{model_name}.json
        # or .../bench/{dataset_name}/{split}/{model_name}.json
        rel = jf.relative_to(bench_dir)
        parts = rel.parts

        if len(parts) >= 2:
            dataset_name = parts[0].lower()
            model_stem = jf.stem

            if dataset_name not in datasets_found:
                datasets_found[dataset_name] = {
                    "path": str(bench_dir / parts[0]),
                    "model_files": [],
                    "sample_record_count": None,
                }

            datasets_found[dataset_name]["model_files"].append(model_stem)
            all_models.add(model_stem)

            # Peek into the first JSON file to check schema and count
            if sample_record is None:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        sample_record = data[0]
                        datasets_found[dataset_name]["sample_record_count"] = len(data)
                    elif isinstance(data, dict):
                        sample_record = {k: type(v).__name__ for k, v in data.items()}
                except Exception as e:
                    pass

    # Print summary of datasets
    print("\n" + "=" * 60)
    print(f"{'DATASET':<25} | {'MODEL FILES':<12} | {'STATUS'}")
    print("-" * 60)

    matched_primary = 0
    matched_ood = 0

    for d_name, info in sorted(datasets_found.items()):
        status = []
        is_prim = any(p in d_name for p in PRIMARY_DATASETS)
        is_ood = any(o in d_name for o in OOD_DATASETS)

        if is_prim:
            status.append("PRIMARY (KevRouter)")
            matched_primary += 1
        if is_ood:
            status.append("OOD (tau2-bench)")
            matched_ood += 1
        if not status:
            status.append("Extra benchmark")

        num_models = len(info["model_files"])
        print(f"{d_name:<25} | {num_models:<12} | {', '.join(status)}")

    print("=" * 60)
    print(f"Total Datasets Discovered: {len(datasets_found)}")
    print(f"Total Unique Models: {len(all_models)}")
    print(f"Matched Primary Datasets: {matched_primary}")
    print(f"Matched OOD Datasets: {matched_ood}")

    manifest = {
        "bench_dir": str(bench_dir),
        "total_datasets": len(datasets_found),
        "total_models": len(all_models),
        "models_list": sorted(list(all_models)),
        "datasets": datasets_found,
        "sample_fields": list(sample_record.keys()) if isinstance(sample_record, dict) else None,
    }

    if output_manifest:
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(output_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[+] Manifest written to {output_manifest}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract LLMRouterBench dataset for KevRouter project."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded archive and extract files (default: data/raw)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step if archive already exists.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if archive already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extract even if extracted directory exists.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect already extracted benchmark directory.",
    )

    args = parser.parse_args()

    # Project root resolution (two levels up from src/data/)
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = (project_root / args.data_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive_path = raw_dir / HF_FILENAME
    manifest_path = raw_dir / "manifest.json"

    print("=== KevRouter LLMRouterBench Ingest ===")
    print(f"Working Directory: {project_root}")
    print(f"Target Raw Directory: {raw_dir}")

    # Inspect only mode
    if args.inspect_only:
        bench_dir = raw_dir / "bench" if (raw_dir / "bench").exists() else raw_dir
        inspect_benchmark_data(bench_dir, output_manifest=manifest_path)
        return

    # 1. Download
    if not args.skip_download:
        download_file(DIRECT_URL, archive_path, force=args.force_download)
    else:
        print(f"[*] Skipping download as requested. Using {archive_path}")

    # 2. Extract
    if not args.skip_extract:
        if not archive_path.exists():
            print(f"[!] Archive {archive_path} does not exist. Cannot extract.")
            sys.exit(1)
        extracted_dir = extract_archive(archive_path, raw_dir, force=args.force_extract)
    else:
        print("[*] Skipping extraction as requested.")
        extracted_dir = raw_dir / "bench" if (raw_dir / "bench").exists() else raw_dir

    # 3. Inspect
    if extracted_dir.exists():
        inspect_benchmark_data(extracted_dir, output_manifest=manifest_path)
    else:
        print(f"[!] Extracted directory {extracted_dir} does not exist.")


if __name__ == "__main__":
    main()
