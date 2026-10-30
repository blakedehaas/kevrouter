"""Inference decision rules for calibrated Kev router probabilities.

Supports:
  1. Standard Greedy ArgMax
  2. Cost-Utility Optimization (Bayes-Optimal Decision Rule)
  3. Confidence-Based Cascading / Tiered Escalation
"""

from typing import Dict, List, Optional, Tuple
import torch

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

# Benchmark empirical average costs per query ($)
ESTIMATED_MODEL_COSTS = {
    "deepseek-v3-0324": 0.0020,
    "gpt-5-chat": 0.0022,
    "deepseek-v3.1-terminus": 0.0038,
    "qwen3-235b-a22b-2507": 0.0071,
    "intern-s1": 0.0095,
    "kimi-k2-0905": 0.0125,
    "claude-sonnet-4": 0.0158,
    "gemini-2.5-flash": 0.0195,
    "qwen3-235b-a22b-thinking-2507": 0.0210,
    "deepseek-r1-0528": 0.0485,
    "glm-4.6": 0.0540,
    "gpt-5": 0.1520,
    "gemini-2.5-pro": 0.2450,
}


def rule_argmax(probs: List[float]) -> str:
    """Standard greedy selection of the highest probability candidate."""
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return DESTINATION_MODELS[best_idx]


def rule_cost_utility(
    probs: List[float],
    mu: float = 1.0,
    costs: Optional[Dict[str, float]] = None,
) -> str:
    """Cost-Utility Bayes Decision Rule: argmax [ P(m | q) - mu * cost(m) ]."""
    c_map = costs or ESTIMATED_MODEL_COSTS
    best_m = DESTINATION_MODELS[0]
    best_utility = -1e9

    for i, m in enumerate(DESTINATION_MODELS):
        p = probs[i]
        c = c_map.get(m, 0.02)
        utility = p - (mu * c)
        if utility > best_utility:
            best_utility = utility
            best_m = m

    return best_m


def rule_cascading(
    probs: List[float],
    tau: float = 0.60,
    cheap_cost_threshold: float = 0.008,
    costs: Optional[Dict[str, float]] = None,
) -> str:
    """Confidence-Based Tiered Escalation:
    
    If any high-throughput cheap model has confidence >= tau, route to it immediately.
    Otherwise escalate to the overall most confident reasoning model.
    """
    c_map = costs or ESTIMATED_MODEL_COSTS
    cheap_indices = [
        i for i, m in enumerate(DESTINATION_MODELS)
        if c_map.get(m, 0.02) <= cheap_cost_threshold
    ]

    best_cheap_idx = max(cheap_indices, key=lambda i: probs[i]) if cheap_indices else 0
    if probs[best_cheap_idx] >= tau:
        return DESTINATION_MODELS[best_cheap_idx]

    # Escalate to overall best
    return rule_argmax(probs)
