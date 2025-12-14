# Debiasing Resume Retrieval with Human-Preference Learning: A Supervised Ranking Approach

This project demonstrates a practical application of a **Human-Feedback-Driven Supervised Learning** approach to reduce demographic bias in automated resume screening systems. We develop a novel framework that trains a fairness-aware ranking model using carefully curated human preference data, effectively steering the system toward more equitable hiring decisions. Our approach addresses name-based discrimination—a well-documented form of hiring bias—by aligning algorithmic behavior with human fairness intuitions while maintaining practical utility.

The repository contains a complete pipeline from baseline model evaluation through preference model training to comprehensive fairness assessment. Our results show significant improvements: a **31.3% reduction in demographic parity difference** and meaningful gains in selection rates for historically disadvantaged groups. This work contributes both a technical methodology for bias mitigation and insights into the complex patterns of algorithmic discrimination that emerge in automated hiring systems.

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
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Download the required database using Git LFS:
```bash
git lfs pull
```

### 2. Baseline Model
Run the baseline model to generate initial resume matching scores:
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
This notebook contains templates for collecting and formatting human preference data used for training the preference model.

### 5. Preference Model Training
Train the supervised ranking model using the human preference data:
```bash
jupyter notebook preference_model.ipynb
```

### 6. Fairness Evaluation
Assess the fairness improvements after model optimization:
```bash
jupyter notebook fairness_evaluation.ipynb
```
This notebook compares the optimized model against the baseline and calculates fairness metric improvements.

## Notes
- Ensure all dependencies are installed using `requirements.txt`
- The human feedback data should be properly formatted before training the preference model
- All evaluation notebooks produce visualizations of fairness metrics across demographic groups
- Model checkpoints and intermediate results are saved for reproducibility



