"""Cost-sensitive regret and utility loss functions for LLM routing.

Implements the objective from Section 1 of PROJECT_PROPOSAL.md:
  - Cheapest correct model incurs 0 regret.
  - Another correct model incurs penalty proportional to additional cost (gamma * delta_cost).
  - Incorrect model incurs substantial failure penalty (lambda) plus cost.
  - Blended with Cross-Entropy for gradient stability.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

DESTINATION_MODELS = [
    "deepseek-v3-0324",
    "qwen3-235b-a22b-2507",
    "deepseek-v3.1-terminus",
    "kimi-k2-0905",
    "intern-s1",
    "gpt-5-chat",
    "qwen3-235b-a22b-thinking-2507",
    "gemini-2.5-flash",
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "glm-4.6",
    "gpt-5",
    "gemini-2.5-pro",
]


class CostSensitiveRegretLoss(nn.Module):
    """Computes expected routing regret over candidate destination models."""

    def __init__(
        self,
        lambda_fail: float = 0.05,
        gamma_cost: float = 5.0,
        alpha_ce: float = 0.5,
    ):
        super().__init__()
        self.lambda_fail = float(lambda_fail)
        self.gamma_cost = float(gamma_cost)
        self.alpha_ce = float(alpha_ce)

    def build_regret_vector(
        self,
        record: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Construct regret values for all 13 models for a single query."""
        model_evals = record.get("model_evals", {})
        optimal_m = record.get("optimal_model") or record.get("questions", {}).get("destination", {}).get("label")
        target_dict = record.get("questions", {}).get("destination", {}).get("target", {})
        meta_cost = record.get("_meta", {}).get("optimal_cost")

        # Empirical costs fallback
        from src.models.decision_rules import ESTIMATED_MODEL_COSTS

        correct_models = {}
        all_costs = {}

        if model_evals:
            for m in DESTINATION_MODELS:
                ev = model_evals.get(m, {})
                c = float(ev.get("cost", ESTIMATED_MODEL_COSTS.get(m, 0.02)))
                all_costs[m] = c
                if ev.get("score", 0.0) >= 1.0:
                    correct_models[m] = c
        else:
            # Fallback to Kev target dict and empirical costs
            for m in DESTINATION_MODELS:
                c = ESTIMATED_MODEL_COSTS.get(m, 0.02)
                all_costs[m] = c
                if m in target_dict:
                    correct_models[m] = c
            if optimal_m and optimal_m in all_costs:
                correct_models[optimal_m] = meta_cost if meta_cost is not None else all_costs[optimal_m]

        min_cost = min(correct_models.values()) if correct_models else min(all_costs.values())

        regrets = []
        for m in DESTINATION_MODELS:
            cost = all_costs[m]
            if model_evals:
                ev = model_evals.get(m, {})
                is_correct = ev.get("score", 0.0) >= 1.0
            else:
                is_correct = (m in target_dict) or (m == optimal_m)

            if is_correct:
                if m == optimal_m or abs(cost - min_cost) < 1e-6:
                    regret = 0.0
                else:
                    regret = self.gamma_cost * max(0.0, cost - min_cost)
            else:
                # Failure penalty plus cost
                regret = self.lambda_fail + cost

            regrets.append(regret)

        return torch.tensor(regrets, dtype=torch.float, device=device)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_records: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Calculate combined Cross-Entropy and Expected Regret loss.
        
        Args:
            logits: [B, K] predicted option logits
            labels: [B] target index of optimal model
            batch_records: list of original query records with model_evals
        """
        probs = F.softmax(logits, dim=-1)

        # 1. Standard cross entropy
        ce_loss = F.cross_entropy(logits, labels)

        # 2. Expected regret: sum(probs * regret_matrix)
        regret_rows = [self.build_regret_vector(r, logits.device) for r in batch_records]
        regret_matrix = torch.stack(regret_rows, dim=0)  # [B, K]
        expected_regret = torch.mean(torch.sum(probs * regret_matrix, dim=-1))

        # 3. Blended loss
        total_loss = self.alpha_ce * ce_loss + (1.0 - self.alpha_ce) * expected_regret
        return total_loss
