"""Validation-driven loss function and decision rule optimization sweep.

Evaluates:
  1. Parameterized Loss Formulations (Alpha CE blend, Lambda failure penalty, Gamma excess cost)
  2. Inference Decision Rules:
      - Cost-Utility Bayes Decision Rule (mu sweep)
      - Confidence-Based Cascading / Escalation (tau sweep)
  3. Identifies the Pareto Frontier of Accuracy vs. Cost
  4. Validates generalization on held-out test splits (test_id and test_ood)
  5. Generates Pareto frontier plot and comparative report
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.benchmark_kev import (
    load_dataset_records,
    evaluate_routing_predictions,
    DESTINATION_MODELS,
)
from src.models.decision_rules import (
    rule_argmax,
    rule_cost_utility,
    rule_cascading,
    ESTIMATED_MODEL_COSTS,
)
from src.models.loss import CostSensitiveRegretLoss
from src.models.train_kev import (
    KevRouterTrainer,
    load_kev_jsonl,
    fit_temperature,
)

LOSS_CONFIGS = [
    {
        "name": "Standard CE",
        "alpha_ce": 1.0,
        "lambda_fail": 0.05,
        "gamma_cost": 0.0,
        "description": "Standard cross-entropy targeting optimal model",
    },
    {
        "name": "Accuracy-Prioritized Regret",
        "alpha_ce": 0.70,
        "lambda_fail": 0.10,
        "gamma_cost": 2.0,
        "description": "Higher failure penalty to prioritize routing correctness",
    },
    {
        "name": "Balanced Regret",
        "alpha_ce": 0.50,
        "lambda_fail": 0.05,
        "gamma_cost": 5.0,
        "description": "Balanced cost-accuracy objective matching proposal",
    },
    {
        "name": "Cost-Aggressive Regret",
        "alpha_ce": 0.20,
        "lambda_fail": 0.02,
        "gamma_cost": 10.0,
        "description": "High cost penalty to heavily penalize expensive destinations",
    },
]

MU_SWEEP = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
TAU_SWEEP = [0.4, 0.5, 0.6, 0.7, 0.8]


def train_with_custom_loss(
    trainer: KevRouterTrainer,
    train_records: List[Dict[str, Any]],
    loss_module: CostSensitiveRegretLoss,
    epochs: int = 2,
    batch_size: int = 16,
) -> float:
    """Train Kev head with parameterized regret loss."""
    trainer.head.train()
    optimizer = torch.optim.AdamW(trainer.head.parameters(), lr=trainer.head_lr, weight_decay=0.01)

    m2i = {m: i for i, m in enumerate(DESTINATION_MODELS)}
    last_loss = 0.0

    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_records), batch_size):
            batch = train_records[i:i + batch_size]
            h_d, h_o = trainer.encode_batch_representations(batch)
            logits = trainer.head(h_d, h_o)

            labels = [m2i.get(r["questions"]["destination"].get("label", "deepseek-v3-0324"), 0) for r in batch]
            labels_t = torch.tensor(labels, dtype=torch.long, device=trainer.device)

            loss = loss_module(logits, labels_t, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1
        last_loss = ep_loss / max(1, n_batches)

    return last_loss


def extract_probabilities(
    trainer: KevRouterTrainer,
    records: List[Dict[str, Any]],
    temperature: float = 1.0,
) -> List[List[float]]:
    """Compute calibrated probability distribution for each query."""
    trainer.head.eval()
    trainer.head.temperature = temperature

    all_probs = []
    for r in records:
        h_d, h_o = trainer.encode_batch_representations([r])
        with torch.no_grad():
            logits = trainer.head(h_d, h_o)
            # Option 1: Dynamic Candidate Masking (e.g. gpt-5-chat missing on tau2-bench)
            model_evals = r.get("model_evals")
            is_tau2 = (r.get("dataset") == "tau2-bench" or r.get("_meta", {}).get("dataset") == "tau2-bench")
            for m_idx, m_name in enumerate(DESTINATION_MODELS):
                if (model_evals is not None and m_name not in model_evals) or (is_tau2 and m_name == "gpt-5-chat"):
                    logits[0, m_idx] = -1e9
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
            all_probs.append(probs)

    return all_probs


def evaluate_decision_rule(
    records: List[Dict[str, Any]],
    probs_list: List[List[float]],
    rule_type: str,
    param: float,
) -> Dict[str, Any]:
    """Score predictions under a specific inference decision rule."""
    predictions = []
    for r, p in zip(records, probs_list):
        if rule_type == "argmax":
            chosen = rule_argmax(p)
        elif rule_type == "cost_utility":
            chosen = rule_cost_utility(p, mu=param)
        elif rule_type == "cascading":
            chosen = rule_cascading(p, tau=param)
        else:
            chosen = rule_argmax(p)
        predictions.append(chosen)

    return evaluate_routing_predictions(records, predictions)


def is_pareto_efficient(costs: List[float], accs: List[float]) -> List[bool]:
    """Find points on the Pareto frontier (minimizing cost, maximizing accuracy)."""
    n = len(costs)
    is_efficient = [True] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                # If j has lower or equal cost AND higher or equal accuracy (strictly better in at least one)
                if costs[j] <= costs[i] and accs[j] >= accs[i]:
                    if costs[j] < costs[i] or accs[j] > accs[i]:
                        is_efficient[i] = False
                        break
    return is_efficient


def plot_pareto_frontier(
    sweep_results: List[Dict[str, Any]],
    output_png: Path,
):
    """Draw Cost vs. Accuracy scatter plot with Pareto frontier highlighted."""
    costs = [r["avg_cost_per_query_usd"] for r in sweep_results]
    accs = [r["routing_accuracy"] for r in sweep_results]
    pareto_mask = is_pareto_efficient(costs, accs)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot all explored configurations
    plt.scatter(
        costs,
        accs,
        color="#94a3b8",
        alpha=0.6,
        s=60,
        label="Explored Configurations",
        edgecolors="none",
    )

    # Plot Pareto optimal points
    p_costs = [c for c, m in zip(costs, pareto_mask) if m]
    p_accs = [a for a, m in zip(accs, pareto_mask) if m]
    p_configs = [r for r, m in zip(sweep_results, pareto_mask) if m]

    # Sort Pareto points by cost
    sorted_pareto = sorted(zip(p_costs, p_accs, p_configs), key=lambda x: x[0])
    p_costs_s = [x[0] for x in sorted_pareto]
    p_accs_s = [x[1] for x in sorted_pareto]

    plt.plot(p_costs_s, p_accs_s, color="#2563eb", linestyle="--", linewidth=2, label="Pareto Frontier")
    plt.scatter(p_costs_s, p_accs_s, color="#dc2626", s=110, zorder=5, label="Pareto-Optimal Points")

    for c, a, cfg in sorted_pareto:
        label_text = f"{cfg['rule']} ({cfg['param']})"
        plt.annotate(
            label_text,
            (c, a),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#1e293b",
        )

    plt.title("LLM Routing Pareto Frontier: Routing Accuracy vs. Cost per Query", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Average Cost per Query ($)", fontsize=12, labelpad=10)
    plt.ylabel("Routing Accuracy (%)", fontsize=12, labelpad=10)
    plt.legend(frameon=True, loc="lower right", fontsize=10)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"[+] Saved Pareto frontier plot to: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Optimize loss function and decision rule on validation set.")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "4b", "9b"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=150)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_base = PROJECT_ROOT / args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "kev_finetune"
    train_records = load_kev_jsonl(data_dir / "train.jsonl")[:args.max_train_samples]
    cal_records = load_kev_jsonl(data_dir / "calibration.jsonl")
    dev_records = load_kev_jsonl(data_dir / "development.jsonl")

    raw_val_records = load_dataset_records(PROJECT_ROOT / "data" / "processed" / "val.jsonl")
    raw_dev_records = raw_val_records[len(cal_records):len(cal_records) + len(dev_records)]
    test_id_records = load_dataset_records(PROJECT_ROOT / "data" / "processed" / "test_id.jsonl")
    test_ood_records = load_dataset_records(PROJECT_ROOT / "data" / "processed" / "test_ood.jsonl")

    print("=" * 82)
    print(" LOSS FUNCTION & DECISION RULE OPTIMIZATION SWEEP")
    print("=" * 82)
    print(f"[*] Exploring {len(LOSS_CONFIGS)} Loss Formulations x Decision Rule Parameter Sweeps...")

    sweep_results = []
    trained_models = {}
    best_config = None
    best_cost_per_correct = 1e9

    for cfg_idx, loss_cfg in enumerate(LOSS_CONFIGS, 1):
        print(f"\n[{cfg_idx}/{len(LOSS_CONFIGS)}] Training Loss Formulation: {loss_cfg['name']}")
        print(f"    Parameters: alpha_ce={loss_cfg['alpha_ce']}, lambda={loss_cfg['lambda_fail']}, gamma={loss_cfg['gamma_cost']}")

        loss_module = CostSensitiveRegretLoss(
            lambda_fail=loss_cfg["lambda_fail"],
            gamma_cost=loss_cfg["gamma_cost"],
            alpha_ce=loss_cfg["alpha_ce"],
        )

        trainer = KevRouterTrainer(model_size=args.model_size)
        train_loss = train_with_custom_loss(
            trainer,
            train_records,
            loss_module,
            epochs=args.epochs,
        )

        # 1. Fit temperature on calibration records
        cal_logits = []
        cal_labels = []
        m2i = {m: i for i, m in enumerate(DESTINATION_MODELS)}
        trainer.head.eval()
        for r in cal_records:
            h_d, h_o = trainer.encode_batch_representations([r])
            logits = trainer.head(h_d, h_o)
            lbl = r["questions"]["destination"].get("label", "deepseek-v3-0324")
            cal_logits.append(logits)
            cal_labels.append(m2i.get(lbl, 0))

        fitted_temp = fit_temperature(cal_logits, cal_labels)
        print(f"    [+] Fitted Calibration Temperature: T = {fitted_temp:.3f}")
        trained_models[loss_cfg["name"]] = (trainer, loss_cfg, fitted_temp)

        # 2. Extract calibrated validation probabilities
        val_probs = extract_probabilities(trainer, dev_records, temperature=fitted_temp)

        # 3. Sweep Decision Rules
        # Rule 1: Greedy Argmax
        res_argmax = evaluate_decision_rule(raw_dev_records[:len(val_probs)], val_probs, "argmax", 0.0)
        entry_argmax = {
            "loss_name": loss_cfg["name"],
            "rule": "Argmax",
            "param": 0.0,
            "routing_accuracy": res_argmax["routing_accuracy"],
            "optimal_match_rate": res_argmax["optimal_match_rate"],
            "avg_cost_per_query_usd": res_argmax["avg_cost_per_query_usd"],
            "cost_per_correct_usd": res_argmax["cost_per_correct_usd"],
        }
        sweep_results.append(entry_argmax)

        # Rule 2: Cost-Utility (mu sweep)
        for mu in MU_SWEEP:
            res_mu = evaluate_decision_rule(raw_dev_records[:len(val_probs)], val_probs, "cost_utility", mu)
            entry_mu = {
                "loss_name": loss_cfg["name"],
                "rule": "Cost-Utility",
                "param": mu,
                "routing_accuracy": res_mu["routing_accuracy"],
                "optimal_match_rate": res_mu["optimal_match_rate"],
                "avg_cost_per_query_usd": res_mu["avg_cost_per_query_usd"],
                "cost_per_correct_usd": res_mu["cost_per_correct_usd"],
            }
            sweep_results.append(entry_mu)
            if res_mu["cost_per_correct_usd"] < best_cost_per_correct:
                best_cost_per_correct = res_mu["cost_per_correct_usd"]
                best_config = (trainer, loss_cfg, "Cost-Utility", mu, fitted_temp)

        # Rule 3: Cascading (tau sweep)
        for tau in TAU_SWEEP:
            res_tau = evaluate_decision_rule(raw_dev_records[:len(val_probs)], val_probs, "cascading", tau)
            entry_tau = {
                "loss_name": loss_cfg["name"],
                "rule": "Cascading",
                "param": tau,
                "routing_accuracy": res_tau["routing_accuracy"],
                "optimal_match_rate": res_tau["optimal_match_rate"],
                "avg_cost_per_query_usd": res_tau["avg_cost_per_query_usd"],
                "cost_per_correct_usd": res_tau["cost_per_correct_usd"],
            }
            sweep_results.append(entry_tau)
            if res_tau["cost_per_correct_usd"] < best_cost_per_correct:
                best_cost_per_correct = res_tau["cost_per_correct_usd"]
                best_config = (trainer, loss_cfg, "Cascading", tau, fitted_temp)

    # Calculate Pareto frontier
    costs = [r["avg_cost_per_query_usd"] for r in sweep_results]
    accs = [r["routing_accuracy"] for r in sweep_results]
    pareto_mask = is_pareto_efficient(costs, accs)
    pareto_points = [r for r, m in zip(sweep_results, pareto_mask) if m]

    print("\n" + "=" * 82)
    print(" OPTIMIZATION SUMMARY: TOP PARETO-EFFICIENT CONFIGURATIONS")
    print("=" * 82)
    print(f"{'LOSS FORMULATION':<24} | {'RULE':<12} | {'PARAM':<6} | {'ACCURACY':<10} | {'OPTIMAL':<10} | {'AVG COST':<10} | {'COST/CORRECT'}")
    print("-" * 96)
    for p in sorted(pareto_points, key=lambda x: x["avg_cost_per_query_usd"]):
        print(
            f"{p['loss_name']:<24} | {p['rule']:<12} | {p['param']:<6.1f} | "
            f"{p['routing_accuracy']:>8.2f}% | {p['optimal_match_rate']:>8.2f}% | "
            f"${p['avg_cost_per_query_usd']:>8.5f} | ${p['cost_per_correct_usd']:>10.5f}"
        )
    print("=" * 82)

    # Plot Pareto curve
    plot_file = out_base / "pareto_frontier.png"
    plot_pareto_frontier(sweep_results, plot_file)

    # Test Generalization on Held-Out Test Splits across key Pareto archetypes
    test_id_dev_records = load_kev_jsonl(data_dir / "test_id.jsonl")
    test_ood_dev_records = load_kev_jsonl(data_dir / "test_ood.jsonl")

    # Select representative points on Pareto frontier:
    # 1. Max Accuracy config
    max_acc_point = max(pareto_points, key=lambda x: x["routing_accuracy"])
    # 2. Min Cost-per-Correct config
    min_cost_point = min(pareto_points, key=lambda x: x["cost_per_correct_usd"])

    eval_archetypes = [
        ("High-Accuracy Frontier", max_acc_point),
        ("Cost-Aggressive Frontier", min_cost_point),
    ]

    archetype_results = []
    print("\n" + "=" * 82)
    print(" HELD-OUT GENERALIZATION ACROSS PARETO ARCHETYPES")
    print("=" * 82)

    for arch_name, pt in eval_archetypes:
        loss_name = pt["loss_name"]
        rule_name = pt["rule"]
        param_val = pt["param"]
        if loss_name not in trained_models:
            continue

        m_trainer, m_loss_cfg, m_temp = trained_models[loss_name]

        # Test In-Distribution
        test_id_probs = extract_probabilities(m_trainer, test_id_dev_records, temperature=m_temp)
        r_type = "cost_utility" if rule_name == "Cost-Utility" else ("cascading" if rule_name == "Cascading" else "argmax")
        id_res = evaluate_decision_rule(test_id_records[:len(test_id_probs)], test_id_probs, r_type, param_val)

        # Test Out-of-Distribution (tau2-bench)
        test_ood_probs = extract_probabilities(m_trainer, test_ood_dev_records, temperature=m_temp)
        ood_res = evaluate_decision_rule(test_ood_records[:len(test_ood_probs)], test_ood_probs, r_type, param_val)

        print(f"\n[*] Archetype: {arch_name} [{loss_name} + {rule_name} (param={param_val})]")
        print(f"    Test In-Distribution:  Accuracy = {id_res['routing_accuracy']:.2f}% | Optimal Match = {id_res['optimal_match_rate']:.2f}% | Avg Cost = ${id_res['avg_cost_per_query_usd']:.5f} | Cost/Correct = ${id_res['cost_per_correct_usd']:.5f}")
        print(f"    Test Out-of-Dist (tau2): Accuracy = {ood_res['routing_accuracy']:.2f}% | Optimal Match = {ood_res['optimal_match_rate']:.2f}% | Avg Cost = ${ood_res['avg_cost_per_query_usd']:.5f} | Cost/Correct = ${ood_res['cost_per_correct_usd']:.5f}")

        archetype_results.append({
            "archetype": arch_name,
            "loss_name": loss_name,
            "rule": rule_name,
            "param": param_val,
            "test_id": id_res,
            "test_ood": ood_res,
        })

    # Save Best Checkpoint (cost-aggressive optimized)
    save_loss_name = min_cost_point["loss_name"]
    if save_loss_name in trained_models:
        best_tr, _, _ = trained_models[save_loss_name]
    elif "Cost-Aggressive Regret" in trained_models:
        best_tr, _, _ = trained_models["Cost-Aggressive Regret"]
    else:
        best_tr = list(trained_models.values())[0][0]
    ckpt_dir = out_base / "checkpoints" / f"kev_{args.model_size}_optimized"
    best_tr.save_checkpoint(ckpt_dir)

    # Save summary JSON
    results_payload = {
        "sweep_results": sweep_results,
        "pareto_points": pareto_points,
        "archetype_results": archetype_results,
        "best_config": {
            "loss_name": min_cost_point["loss_name"],
            "rule": min_cost_point["rule"],
            "param": min_cost_point["param"],
        },
    }
    with open(out_base / "optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[+] Optimization results saved to: {out_base / 'optimization_results.json'}")


if __name__ == "__main__":
    main()
