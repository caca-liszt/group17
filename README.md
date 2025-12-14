# Debiasing Resume Retrieval with Human-Preference Learning: A Supervised Ranking Approach

This project demonstrates a practical application of a **Human-Feedback-Driven Supervised Learning** approach to reduce demographic bias in automated resume screening systems. We develop a novel framework that trains a fairness-aware ranking model using carefully curated human preference data, effectively steering the system toward more equitable hiring decisions. Our approach addresses name-based discrimination—a well-documented form of hiring bias—by aligning algorithmic behavior with human fairness intuitions while maintaining practical utility.

The repository contains a complete pipeline from baseline model evaluation through preference model training to comprehensive fairness assessment. Our results show significant improvements: a **31.3% reduction in demographic parity difference** and meaningful gains in selection rates for historically disadvantaged groups. This work contributes both a technical methodology for bias mitigation and insights into the complex patterns of algorithmic discrimination that emerge in automated hiring systems.

## Datasets

This project uses two public Kaggle datasets:

- Resume Dataset (resumes)
  - https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

- Scraped Job Descriptions (job descriptions)
  - https://www.kaggle.com/datasets/marcocavaco/scraped-job-descriptions

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
#### 1.1 Filter & Align by ISCO2 with Confidence
```
python filter_by_isco2_conf.py
```
Typical outputs include:
```
resumes_filtered_isco2_conf060.csv

jds_filtered_isco2_conf060.csv
```
summary tables (coverage / before-after stats)

#### 1.2 Name Pool (Required for Name Injection)

The name injection step relies on a name pool CSV:

File: `names_first_only_per_group_pool.csv`
Purpose: provides demographic-grouped first/last names (and optionally gender) used to create controlled resume variants.

Required columns (minimum):

group (e.g., white, black, east_asian, south_asian_indian)

first_name

last_name

Optional column (if you use gendered sampling):

gender

Place this file in the project root (or update the path inside name_injection.py).

#### 1.3 Inject Names to Create Controlled Demographic Variants

This generates two resume “views” (full vs title-only) and injects names/emails for multiple demographic groups.
```
python name_injection.py
```

Typical outputs include:
```
resumes_full_injected.csv

resumes_title_only_injected.csv

variants_meta.csv
```
#### 1.4 Export Text Lines for Embedding

`embeddings_patched.py` expects line-based text files (one item per line).
Export from CSV columns into .txt files in the same row order.

Common exports:
```
descriptions.txt (from JD descriptions)

resumes_withname.txt (from injected resume text)
```
#### 1.5 Compute Embeddings

Job description embeddings:
```
python embeddings_patched.py \
  -m intfloat/e5-base-v2 \
  -q descriptions.txt \
  -o jd_embeds.pkl \
  -b 32 -l 512
```

Injected resume embeddings:
```
python embeddings_patched.py \
  -m intfloat/e5-base-v2 \
  -d resumes_withname.txt \
  -o resume_embeds.pkl \
  -b 32 -l 512
```
#### 1.6 Build Retrieval Pairs (Positives + Negatives)
```
python build_dataset.py \
  --jobs_csv jds_filtered_isco2_conf060.csv \
  --resumes_csv resumes_for_pairs.csv \
  --job_embeds_pkl jd_embeds.pkl \
  --resume_embeds_pkl resume_embeds.pkl \
  --topk 5 \
  --neg_per_pos 1 \
  --add_scores \
  --output_csv retrieval_pairs.csv
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



