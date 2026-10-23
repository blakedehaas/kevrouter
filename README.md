# KevRouter

KevRouter is an intelligent, cost-sensitive model router built on top of the **Kev** decision model architecture (open-source System-1 decision model based on Qwen 3.5). The router predicts the optimal destination model among 13 LLMs from the **LLMRouterBench** dataset, optimizing the performance-cost tradeoff.

For full project motivation, memory estimations, FLOPS calculations, and experimental design, see [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md).

---

## Repository Structure

```
semester_project/
├── PROJECT_PROPOSAL.md                  # Project proposal & design specification
├── README.md                            # Repository overview and quickstart
├── requirements.txt                     # Core dependencies
├── setup_env.sh                         # Linux / Delta cluster environment setup
├── setup_env.ps1                        # Windows PowerShell environment setup
├── scripts/
│   ├── download_data.py                 # Download & extract LLMRouterBench
│   ├── build_dataset.py                 # Deduplicate, align, and stratify 80/10/10 splits
│   └── run_eda.py                       # CLI exploratory data analysis report
├── src/
│   ├── data/
│   │   ├── download_and_extract.py      # Core Hugging Face downloader & extractor
│   │   ├── build_dataset.py             # Data alignment, deduplication & splitting logic
│   │   └── eda.py                       # Analytical metrics calculation
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA and visual plots
└── data/                                # (Git ignored)
    ├── raw/                             # Downloaded tar.gz and raw benchmark JSONs
    └── processed/                       # train.jsonl, val.jsonl, test_id.jsonl, test_ood.jsonl
```

---

## Quickstart

### 1. Set Up Environment
On Linux / Delta Cluster:
```bash
bash setup_env.sh
source kevrouter_venv/bin/activate
```

On Windows PowerShell:
```powershell
.\setup_env.ps1
```

### 2. Download and Extract LLMRouterBench
Downloads `bench-release.tar.gz` (~1.28 GB) from `NPULH/LLMRouterBench` and extracts to `data/raw/bench/`:
```bash
python scripts/download_data.py
```

### 3. Build Splits and Generate Cost-Sensitive Labels
Filters the 9 primary datasets (AIME, LiveMathBench, LiveCodeBench, SWE-Bench, GPQA, HLE, MMLU-Pro, SimpleQA, ArenaHard) and isolates $\tau^2$-Bench OOD queries. Aligns destination model records, removes duplicates, identifies the cheapest correct model for each query, and saves deterministic 80/10/10 stratified splits:
```bash
python scripts/build_dataset.py
```

### 4. Run Exploratory Data Analysis (EDA)
Generate the console report and JSON summary:
```bash
python scripts/run_eda.py
```
Or open the interactive Jupyter notebook in `notebooks/01_exploratory_data_analysis.ipynb`:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```
