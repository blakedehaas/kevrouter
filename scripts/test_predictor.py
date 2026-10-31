import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from kev.predictors import LocalPredictor
from kev.data import load_records

def main():
    ckpt_path = "c:/Users/realb/OneDrive/Desktop/School/AI Engineering/semester_project/models/test_dryrun"
    print("Loading LocalPredictor on CUDA...")
    pred = LocalPredictor(ckpt_path, "cuda")
    print("LocalPredictor loaded!")

    cal_records = load_records("c:/Users/realb/OneDrive/Desktop/School/AI Engineering/semester_project/data/kev_finetune/calibration.jsonl")
    sample_record = cal_records[0]

    res = pred(sample_record)
    print("\n--- PREDICTION OUTPUT ---")
    probs = res["probabilities"]["destination"]
    for m, p in sorted(probs.items(), key=lambda x: -x[1])[:5]:
        print(f"  {m:<30}: {p*100:>6.2f}%")
    print(f"Latency: {res['latency_ms']:.2f} ms")
    print(f"Input tokens: {res['input_tokens']}")

if __name__ == "__main__":
    main()

