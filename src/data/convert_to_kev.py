"""Convert processed LLMRouterBench dataset into official Kev fine-tuning format.

This outputs:
  - data/kev_finetune/train.jsonl
  - data/kev_finetune/calibration.jsonl (for temperature fitting)
  - data/kev_finetune/development.jsonl (for metric evaluation)
  - data/kev_finetune/test_id.jsonl
  - data/kev_finetune/test_ood.jsonl

Each line is formatted strictly as:
{
  "state": "<query prompt>",
  "questions": {
    "destination": {
      "type": "choice",
      "instructions": "Select the destination model that will answer this query correctly at the lowest cost.",
      "criteria": { ... 13 models ... },
      "label": "<optimal_model>",
      "target": { ... cost-sensitive soft weights ... }
    }
  },
  "_meta": {
    "dataset": "...",
    "optimal_cost": 0.002,
    "any_correct": true
  }
}
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import tiktoken

_ENCODER = None

def get_encoder():
    """Lazily load tiktoken cl100k_base encoder."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER

def truncate_to_tokens(text: str, max_tokens: int = 1024) -> str:
    """Truncate text to at most max_tokens using cl100k_base tokenization."""
    if not text:
        return ""
    # Fast path: short texts cannot exceed max_tokens
    if len(text) <= max_tokens:
        return text.strip()
    enc = get_encoder()
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text.strip()
    return enc.decode(tokens[:max_tokens]).strip()

DESTINATION_CRITERIA = {
    "deepseek-v3-0324": "DeepSeek V3 0324: ultra-low cost general reasoning ($0.002/query)",
    "gpt-5-chat": "GPT-5 Chat: low-cost conversational and reasoning agent ($0.002/query)",
    "deepseek-v3.1-terminus": "DeepSeek V3.1 Terminus: low-cost balanced model ($0.004/query)",
    "intern-s1": "Intern S1: low-to-mid cost reasoning and coding model ($0.010/query)",
    "qwen3-235b-a22b-2507": "Qwen 3 235B A22B: fast dense open-weights foundation model ($0.007/query)",
    "kimi-k2-0905": "Kimi K2 0905: strong coding and multi-step reasoning ($0.012/query)",
    "claude-sonnet-4": "Claude Sonnet 4: premier coding, logic, and instruction following ($0.016/query)",
    "gemini-2.5-flash": "Gemini 2.5 Flash: high-speed multimodal and reasoning model ($0.020/query)",
    "qwen3-235b-a22b-thinking-2507": "Qwen 3 235B Thinking: extended reasoning open weights ($0.021/query)",
    "deepseek-r1-0528": "DeepSeek R1 0528: frontier reasoning and mathematical derivation ($0.050/query)",
    "glm-4.6": "GLM 4.6: high-capability reasoning and analysis ($0.055/query)",
    "gpt-5": "GPT-5: frontier flagship foundation model ($0.150/query)",
    "gemini-2.5-pro": "Gemini 2.5 Pro: frontier flagship reasoning with massive context ($0.250/query)",
}

INSTRUCTIONS = "Select the destination model that will answer this query correctly at the lowest cost."


def compute_cost_sensitive_target(
    model_evals: Dict[str, Dict[str, Any]],
    optimal_model: str,
    cost_beta: float = 20.0,
) -> Dict[str, float]:
    """Calculate cost-sensitive soft target distribution across models.
    
    Correct models receive exponential preference inversely proportional to their cost penalty.
    """
    correct_models = {}
    for m, ev in model_evals.items():
        if m in DESTINATION_CRITERIA and ev.get("score", 0.0) >= 1.0:
            correct_models[m] = ev.get("cost", 0.0)

    if not correct_models:
        # If no model got it right, label the designated optimal model with full weight
        return {optimal_model: 1.0}

    # Exponential weighting: cheaper correct models get higher mass
    weights = {}
    min_cost = min(correct_models.values())
    for m, cost in correct_models.items():
        # Normalized extra cost above optimal
        extra_cost = max(0.0, cost - min_cost)
        weights[m] = math.exp(-cost_beta * extra_cost)

    total_weight = sum(weights.values())
    return {m: round(w / total_weight, 5) for m, w in weights.items()}


def convert_record(raw: Dict[str, Any], max_tokens: int = 1024) -> Optional[Dict[str, Any]]:
    """Transform raw LLMRouterBench record into Kev JSONL shape.
    
    Prompts are token-truncated using tiktoken BPE to ensure consistent context bounds
    while preserving full problem formulations for >= 94.6% of queries without truncation.
    """
    prompt = raw.get("prompt") or raw.get("origin_query") or ""
    if not prompt.strip():
        return None

    # Truncate prompt to fit within empirical token budget
    state = truncate_to_tokens(prompt, max_tokens=max_tokens)

    optimal_model = raw.get("optimal_model")
    if not optimal_model or optimal_model not in DESTINATION_CRITERIA:
        # Fallback to cheapest model
        optimal_model = "deepseek-v3-0324"

    model_evals = raw.get("model_evals", {})
    target = compute_cost_sensitive_target(model_evals, optimal_model)

    return {
        "state": state,
        "questions": {
            "destination": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": DESTINATION_CRITERIA,
                "label": optimal_model,
                "target": target,
            }
        },
        "_meta": {
            "dataset": raw.get("dataset", "Unknown"),
            "optimal_cost": raw.get("optimal_cost", 0.0),
            "any_correct": raw.get("any_model_correct", False),
            "index": raw.get("index"),
        },
    }


def convert_dataset_splits(
    processed_dir: Path,
    output_dir: Path,
    calibration_split_size: int = 500,
    max_tokens: int = 1024,
):
    """Read processed data and write Kev-finetune partitioned JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": processed_dir / "train.jsonl",
        "val": processed_dir / "val.jsonl",
        "test_id": processed_dir / "test_id.jsonl",
        "test_ood": processed_dir / "test_ood.jsonl",
    }

    converted = {}
    for split_name, file_path in split_files.items():
        if not file_path.exists():
            print(f"[!] Warning: {file_path} not found. Skipping {split_name}.")
            continue

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw = json.loads(line)
                    rec = convert_record(raw, max_tokens=max_tokens)
                    if rec:
                        records.append(rec)

        converted[split_name] = records
        print(f"[*] Processed {len(records):,} records for {split_name} (max_tokens={max_tokens})")

    # Partition validation into calibration (for temperature fitting) and development (for evaluation)
    val_records = converted.get("val", [])
    if val_records:
        cal_records = val_records[:calibration_split_size]
        dev_records = val_records[calibration_split_size:]
    else:
        cal_records, dev_records = [], []

    # Write output files
    targets = {
        "train.jsonl": converted.get("train", []),
        "calibration.jsonl": cal_records,
        "development.jsonl": dev_records,
        "test_id.jsonl": converted.get("test_id", []),
        "test_ood.jsonl": converted.get("test_ood", []),
    }

    for filename, rows in targets.items():
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[+] Wrote {len(rows):,} records to {out_path.name}")

    summary = {
        "train_records": len(targets["train.jsonl"]),
        "calibration_records": len(targets["calibration.jsonl"]),
        "development_records": len(targets["development.jsonl"]),
        "test_id_records": len(targets["test_id.jsonl"]),
        "test_ood_records": len(targets["test_ood.jsonl"]),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Kev fine-tuning dataset successfully generated in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert LLMRouterBench data to official Kev fine-tuning format.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum prompt length in tokens (default: 1024, covers >=94.6% of prompts losslessly)",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=500,
        help="Number of validation records reserved for temperature calibration (default: 500)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "kev_finetune"

    convert_dataset_splits(
        processed_dir,
        output_dir,
        calibration_split_size=args.calibration_size,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
