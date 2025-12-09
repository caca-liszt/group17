# Fine-Tuning LLMs with RLHF for Bias Reduction in Resume Screening

This project demonstrates a practical application of Reinforcement Learning from Human Feedback (RLHF) to reduce demographic bias in automated resume screening systems. We develop a novel framework that fine-tunes large language models using carefully curated human preference data, effectively steering the model toward more equitable hiring decisions. Our approach addresses name-based discrimination—a well-documented form of hiring bias—by aligning AI behavior with human fairness intuitions while maintaining practical utility.

The repository contains a complete pipeline from baseline model evaluation through RLHF optimization to comprehensive fairness assessment. Our results show significant improvements: a 31.3% reduction in demographic parity difference and meaningful gains in selection rates for historically disadvantaged groups. This work contributes both a technical methodology for bias mitigation and insights into the complex patterns of algorithmic discrimination that emerge in automated hiring systems.

## Members

- [Jasmine Zhu](https://github.com/jasminezjr) 
- [Yuxuan Qin](https://github.com/caca-liszt) 
- [Cynthia Cui](https://github.com/yc6062-rgb) 

## Step-by-Step Tutorial

### 1. Setup

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/caca-liszt/group17.git
cd group17
```

Download the required database using Git LFS:
```bash
git lfs pull
```

### 2. Baseline Model
Then run the baseline model to generate initial resume matching scores:
```bash
jupyter notebook baseline.ipynb
```

### 3. Baseline Evaluation
Evaluate the fairness characteristics of the baseline model:
```bash
jupyter notebook baseline_fairness.ipynb
```
This notebook calculates demographic parity ratios and selection rates across different demographic groups.

### 4. Human Feedback Collection
Prepare and process human feedback data:
```bash
jupyter notebook human_feedback_empty.ipynb
```
This notebook contains templates for collecting and formatting human preference data used for RLHF training.

### 5. RLHF Training
Run the RLHF fine-tuning process:
```bash

```

### 6. RLHF Evaluation
Assess the fairness improvements after RLHF optimization:
```bash
jupyter notebook rlhf_fairness.ipynb
```
This notebook compares the RLHF-optimized model against the baseline and calculates fairness metric improvements.

## Notes
- Ensure all dependencies are installed using `requirements.txt`
- The human feedback data should be properly formatted before RLHF training
- All evaluation notebooks produce visualizations of fairness metrics across demographic groups
- Model checkpoints and intermediate results are saved for reproducibility



