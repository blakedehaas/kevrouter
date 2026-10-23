# KevRouter: Performance-Cost Tradeoff in LLM Routing

**Authors:** Blake DeHaas and Abhinav Dugar  
**Course:** AI Engineering — Fall 2026  

---

## 1. What You Will Build

We will build a model router using the decision model **Kev** (the open-source version of Jev) that assigns each query to one of 13 destination models in the **LLMRouterBench** dataset. This model produces a probability distribution over a classification problem, so in our case we will be using these 13 different destination models as the 13 classes and select the highest probability prediction from the Kev model as our routing decision.

We will compare this to a baseline **Qwen 3.5** LLM model. Kev is built on the Qwen 3.5 architecture, so we will use the corresponding Qwen 3.5 model in each model size: **0.8B, 4B, and 9B parameters**. For each model, we will evaluate both a base model and a fine-tuned version of the model.

### Fine-Tuning Methodology
- **LoRA Configuration:** Rank ($r$) = 16, LoRA alpha ($\alpha$) = 32 (matching the official Kev baseline architecture).
- **Comparison Matrix:** 12 total models to compare metrics (Base vs. Fine-Tuned across 3 model sizes and 2 model families: Kev and Qwen 3.5) to determine the best model choice to optimize the performance-cost tradeoff.
- **Cost-Sensitive Objective:**
  - For each query, the ideal routing target is the **least expensive destination model that answers correctly** (incurs zero routing loss).
  - Selecting another correct model incurs a **smaller penalty proportional to its additional cost**.
  - Selecting an incorrect model incurs a **substantially larger failure penalty plus its cost**.
  - This cost-sensitive objective encourages the router to choose the cheapest model capable of answering correctly rather than simply favoring either the most accurate or the least expensive destination.
- **Loss Function:** Implemented via a custom loss function during fine-tuning.
- **Evaluation & Tracking:** Run each router model locally and use **Opik** to capture metrics on performance, cost, latency, and generated outputs.
- **Deliverables:** Final presentation and an open-source GitHub code repository.

---

## 2. On What Data

We plan to fine-tune a version of each of the base router models using the training and validation splits of the **LLMRouterBench** dataset, and test on the test split and out-of-distribution **$\tau^2$-Bench** test set for all of the models. LLMRouterBench provides standardized destination-model responses, task scores, token counts, and costs, allowing training and evaluation of routing without issuing new API calls to the 13 destination models.

### Data Splits & Stratification
- **Primary Data Pool:** 12,168 queries sourced from:
  - AIME
  - LiveMathBench
  - LiveCodeBench
  - SWE-Bench
  - GPQA
  - HLE
  - MMLU-Pro
  - SimpleQA
  - ArenaHard
- **Splits:** Deterministic **80/10/10** training, validation, and test splits within each dataset, stratified by dataset:
  - **Training:** 9,734 queries (for fine-tuning and adapter updates)
  - **Validation:** 1,217 queries (for hyperparameter selection, early stopping, and calibration)
  - **Test (In-Distribution):** 1,217 held-out queries (for final in-distribution evaluation)
  - **$\tau^2$-Bench (OOD):** 278 queries completely held out to evaluate out-of-distribution transfer to an unseen task type.
- **Pre-processing:**
  - Normalize and hash queries to remove duplicates.
  - Construct each split separately within each dataset while preserving the approximate distribution of routing labels.
  - Retain only queries with complete results for all destination models.
- **Token Volume Estimate:**
  - Given 9,734 training queries at an average of ~1,024 tokens per query:
    $$\text{Total Training Tokens} = 9{,}734 \times 1{,}024 = 9{,}967{,}616 \approx 10\text{M tokens}$$

---

## 3. On What Compute

We will fine-tune the following six models:
1. Kev 0.8B
2. Kev 4B
3. Kev 9B
4. Qwen 3.5 0.8B
5. Qwen 3.5 4B
6. Qwen 3.5 9B

Each configuration will be trained on a **single A40 GPU (48GB VRAM)** on the Delta cluster.

### Memory Estimation (Qwen 3.5 9B Baseline)
- **Base Weights (BF16):** $\approx 20\text{ GB}$ ($9\text{B} \times 2\text{ bytes} + 2\text{GB headroom}$).
- **Available VRAM:** $48\text{ GB} - 20\text{ GB} \approx 28\text{–}30\text{ GB}$ remaining for LoRA gradients, AdamW optimizer states, and activation memory.
- **Trainable LoRA Parameters ($r=16$):**
  - Per layer: $\text{rank} \times (\text{incoming dim} + \text{outgoing dim})$
  - Total LoRA parameters: $43{,}278{,}336$
    - Full attention: $3{,}932{,}160$
    - Linear attention: $14{,}180{,}352$
    - MLP: $25{,}165{,}824$
  - Parameter memory (16 bytes/param for weights, gradients, Adam states, FP32 master copy):
    $$16 \times 43{,}278{,}336 \approx 0.65\text{ GB}$$
- **Activations & Logits (Batch size = 1, Seq length = 2048, Hidden dim = 4096, Heads = 32):**
  - Activation memory: $\approx 12\text{ GB}$ ($1 \times 2048 \times 4096 \times 32 \times 2 \times 24$ accounting for additional attention, MLP, normalization, and linear attention activations).
  - Full sequence logits: $1 \times 2048 \times 248{,}320 \times 2 \approx 0.95\text{ GB}$.
  - Loss converted to FP32: $\approx 1.89\text{ GB}$.
  - CUDA kernels, buffers, PyTorch caching, fragmentation: $\approx 8\text{ GB}$.
- **Total Memory Footprint:** $\approx 44\text{ GB}$ (fits within the 48GB A40 GPU).

### Compute & Training Time Estimate
- Rule-of-thumb from *Scaling Laws for Neural Language Models*: $6\text{ FLOPs}$ per parameter per training token.
- For 9B parameters and $9{,}967{,}616$ tokens:
  $$\text{FLOPs for 1 epoch} = 6 \times 9\times 10^9 \times 9{,}967{,}616 = 5.3825 \times 10^{17}\text{ FLOPs}$$
- A40 theoretical peak FLOPs: $1.497 \times 10^{14}\text{ FLOPs/s}$. At 40% MFU:
  $$\text{Effective Throughput} = 0.4 \times 1.497 \times 10^{14} = 5.988 \times 10^{13}\text{ FLOPs/s}$$
  $$\text{Time for 9B model} = \frac{5.3825 \times 10^{17}}{5.988 \times 10^{13}} \approx 8{,}989\text{ seconds} \approx 2.5\text{ hours}$$
- Linearly interpolating across model sizes (9B, 4B, 0.8B for Kev and Qwen):
  $$(2.5 \times 2) + \left(2.5 \times \frac{4}{9} \times 2\right) + \left(2.5 \times \frac{0.8}{9} \times 2\right) = 7.67\text{ GPU-hours}$$
  to train all six fine-tuned model variants.

---

## 4. How You Will Know It Worked

- **Success Criterion:** Collect and compare performance metrics for all **12 model configurations**.
- **Evaluated Dimensions:**
  - Router family (Kev vs. Qwen 3.5)
  - Model size (0.8B, 4B, 9B)
  - Fine-tuning effect (Base vs. LoRA Fine-Tuned)
- **Key Metrics:**
  - Routing accuracy / performance
  - Average cost per query
  - Cost per correct answer
  - Routing latency
  - Out-of-distribution generalization on $\tau^2$-Bench
- **Milestone:** Re-evaluate scope if all results are not obtained by Thanksgiving Break on November 23.
