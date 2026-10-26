# NeonatalBERT: a pretrained language model for predicting neonatal morbidities from clinical notes

This repository contains the code for **NeonatalBERT**, a domain-specific language model for neonatal clinical notes, described in:

> Xie F, Chung P, Reiss JD, et al. **Development and validation of a pre-trained language model for neonatal morbidities: a retrospective, multicentre, prognostic study.** *Lancet Digit Health* 2025; 7(12). doi:[10.1016/j.landig.2025.100926](https://doi.org/10.1016/j.landig.2025.100926) · [PMC12748047](https://pmc.ncbi.nlm.nih.gov/articles/PMC12748047/)

NeonatalBERT was initialised from Bio_ClinicalBERT and further pretrained (masked language modelling + next sentence prediction) on neonatal notes written during the first 3 months of life. It was then fine-tuned to predict neonatal outcomes from notes written in the **first 24 hours after birth**:

- **Primary cohort (Stanford, 2014–2021, n = 32,321):** 19 outcomes, temporal split (births 2014–2020 for development, 2021 for testing). Mean AUROC 0.910, mean AUPRC 0.291.
- **External cohort (BIDMC/MIMIC-III, n = 7,061):** 10 outcomes, random 80/20 patient-level split. Mean AUROC 0.935, mean AUPRC 0.360.

---

## ⚠️ Availability of model weights and data

**The pretrained NeonatalBERT weights are not publicly available.** NeonatalBERT was pretrained on clinical notes from the Stanford Research Data Repository (STARR). These notes contain protected health information, and language models can memorise parts of their training text, so the weights fall under the same institutional and patient-privacy restrictions as the Stanford data. Neither the Stanford data nor models trained on it can be released through open access.

What you can do at your own institution:

1. **Fine-tuning-only mode (no NeonatalBERT weights needed).** Fine-tune a public clinical encoder such as [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) on your own first-day notes using `model_fine_tune/finetune.py`. This is the "fine-tuning-only mode" for resource-limited settings described in the paper, and it reproduces the Bio_ClinicalBERT comparator.
2. **Re-create NeonatalBERT locally.** Run the domain-adaptive pretraining pipeline in `model_pretraining/` on your own neonatal notes, starting from Bio_ClinicalBERT, then fine-tune the result. This mirrors how NeonatalBERT was built.
3. **Collaborate with us.** If you are interested in evaluating the original NeonatalBERT model on your data, please contact Feng Xie (xie00469@umn.edu). Any such evaluation would require appropriate institutional agreements.

The **external MIMIC-III (BIDMC) data** are available to credentialed users through [PhysioNet](https://physionet.org/content/mimiciii/1.4/). `data_preparation/build_mimic3_cohort.py` builds the newborn cohort used in the paper.

---

## Repository structure

```
NeonatalBERT/
├── helpers.py                         # metrics, bootstrap CIs, ROC/calibration plots (PlotROCCurve)
├── data_utils.py                      # loading notes/labels, first-24h aggregation, patient-level splits
├── data_preparation/
│   └── build_mimic3_cohort.py         # external BIDMC newborn cohort from MIMIC-III v1.4
├── configs/
│   └── mimic3_icd9_outcomes.json      # ICD-9 code prefixes used to label MIMIC-III outcomes
├── model_pretraining/
│   ├── prepare_notes.py               # step 1: notes -> sentence-per-line corpus
│   ├── create_pretraining_data.py     # step 2: MLM + NSP training instances
│   └── pretrain.py                    # step 3: domain-adaptive pretraining (MLM + NSP)
├── model_fine_tune/
│   └── finetune.py                    # fine-tuning, hyperparameter selection, evaluation
├── baseline_evaluation/
│   └── llm_baseline.py                # zero-shot LLM baseline (e.g. Llama-3.1-8B-Instruct)
├── examples/
│   ├── make_synthetic_data.py         # synthetic NICU notes for a smoke test (no real data)
│   └── make_tiny_bert.py              # tiny random BERT for offline smoke tests
├── requirements.txt
└── CITATION.cff
```

## Installation

```bash
git clone https://github.com/fengx13/NeonatalBERT.git
cd NeonatalBERT
python -m venv .venv && source .venv/bin/activate
# install the PyTorch build matching your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

A GPU is strongly recommended for pretraining and fine-tuning. The paper used Python 3.9 on a Linux server with 4 NVIDIA A40 GPUs. `finetune.py --mode embedding_lr` and the synthetic example also run on a CPU.

## Input data format

All scripts take two tables (`.csv`, `.csv.gz`, `.parquet` or `.pkl`):

**`notes`: one row per note**

| column | required | description |
|---|---|---|
| `patient_id` | yes | newborn identifier |
| `note_text` | yes | note text |
| `hours_since_birth` | recommended | note time in hours after birth; notes with 0 ≤ hours ≤ `--max_hours` (default 24) are used |
| `note_type` | optional | e.g. History and physical, Progress note, Procedure note, Consult |

**`labels`: one row per patient**

| column | required | description |
|---|---|---|
| `patient_id` | yes | matches the notes table |
| one column per outcome | yes | binary 0/1 (e.g. `rds`, `bpd`, `ivh`, `nec`, `sepsis`, `rop`, `death`) |
| split column | optional | e.g. `split` with values `train` / `test`, for a predefined (e.g. temporal) split |

Notes written within the first 24 hours are concatenated per patient in chronological order. Patients without eligible notes or with missing outcome data are excluded, as in the paper. In the paper, outcomes were defined from SNOMED-CT codes (Stanford) and ICD-9 codes (MIMIC-III). The code lists are given in the paper's appendix.

> **Avoid leakage:** do not include discharge summaries or any note written after the prediction time as model input.

## Quick start on synthetic data (≈5 minutes on a CPU)

```bash
python examples/make_synthetic_data.py --out_dir examples/synthetic
python examples/make_tiny_bert.py --corpus examples/synthetic/notes.csv --out_dir examples/tiny_bert

python model_fine_tune/finetune.py \
    --notes examples/synthetic/notes.csv --labels examples/synthetic/labels.csv \
    --model_name_or_path examples/tiny_bert \
    --split random --max_length 128 --max_chunks 2 --lr_grid 1e-3 --epochs 5 \
    --output_dir results/synthetic_demo
```

The tiny model is randomly initialised, so the numbers are meaningless. This only checks that the pipeline runs. With internet access you can replace `examples/tiny_bert` with `emilyalsentzer/Bio_ClinicalBERT`.

## 1. Domain-adaptive pretraining

Use only notes from development-set patients (the paper used notes from the first 3 months of life of patients in the training set).

```bash
# Step 1: one sentence per line, blank line between notes
python model_pretraining/prepare_notes.py \
    --input_path data/pretraining_notes.pkl --output_path corpus/neonatal_notes.txt

# Step 2: generate MLM + NSP instances
python model_pretraining/create_pretraining_data.py \
    --train_corpus corpus/neonatal_notes.txt \
    --bert_model emilyalsentzer/Bio_ClinicalBERT \
    --output_dir corpus/pregenerated \
    --epochs_to_generate 3 --max_seq_len 128 --do_whole_word_mask --reduce_memory

# Step 3: continue pretraining from Bio_ClinicalBERT
python model_pretraining/pretrain.py \
    --pregenerated_data corpus/pregenerated \
    --bert_model emilyalsentzer/Bio_ClinicalBERT \
    --output_dir checkpoints/neonatalbert \
    --epochs 3 --train_batch_size 32 --learning_rate 5e-5 --fp16
```

Each epoch is saved with `save_pretrained` (`checkpoints/neonatalbert/epoch_k/`), so any checkpoint can be passed to `finetune.py`. By default the starting tokenizer's own casing setting is used (`--do_lower_case` forces lower-casing). If ClinicalBERT's [`heuristic_tokenize.py`](https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/lm_pretraining/heuristic_tokenize.py) is on your `PYTHONPATH`, `prepare_notes.py` uses it for sentence splitting. Otherwise a built-in rule-based splitter is used.

## 2. Fine-tuning and evaluation

`finetune.py` adds a logistic (single-logit) classification layer for each outcome, selects hyperparameters on a validation subset of the development data (learning-rate grid + early stopping on mean validation AUPRC), and evaluates on the held-out test set with 95% bootstrap confidence intervals.

**Primary-cohort style (temporal split defined in the labels file):**

```bash
python model_fine_tune/finetune.py \
    --notes data/notes.csv --labels data/labels.csv \
    --model_name_or_path checkpoints/neonatalbert \
    --split column --split_col split --test_value test \
    --save_models --fp16 \
    --output_dir results/primary
```

**External-cohort style (random 80/20 patient-level split, e.g. MIMIC-III):**

```bash
python data_preparation/build_mimic3_cohort.py \
    --mimic_dir /path/to/mimic-iii-clinical-database-1.4 --out_dir data/mimic3_newborn

python model_fine_tune/finetune.py \
    --notes data/mimic3_newborn/notes.csv.gz --labels data/mimic3_newborn/labels.csv \
    --model_name_or_path checkpoints/neonatalbert \
    --split random --test_size 0.2 --save_models --fp16 \
    --output_dir results/mimic3
```

**Applying saved models to a new site without re-fitting (transportability):**

```bash
python model_fine_tune/finetune.py \
    --notes site_b/notes.csv --labels site_b/labels.csv \
    --eval_only --checkpoint_dir results/primary --output_dir results/primary_to_site_b
```

Key options:

| option | default | description |
|---|---|---|
| `--mode` | `finetune` | `finetune` (end-to-end) or `embedding_lr` (frozen encoder, [CLS] embeddings + logistic regression with `GridSearchCV`) |
| `--head` | `per_outcome` | one model per outcome, or `multilabel` (shared encoder, one logit per outcome) |
| `--max_hours` | `24` | prediction window after birth (the paper also reports days 3 and 7) |
| `--max_length` / `--max_chunks` / `--stride` | `512` / `4` / `64` | long documents are split into overlapping 512-token windows. Window probabilities are pooled per patient (`--chunk_pool mean|max`). Use `--max_chunks 1` for plain truncation |
| `--lr_grid`, `--epochs`, `--patience` | `2e-5,3e-5,5e-5`, `4`, `2` | hyperparameter search |
| `--pos_weight auto` | off | up-weight positive cases for rare outcomes |
| `--threshold_from` | `test` | where the threshold for sensitivity/specificity/F1 is chosen (ROC point closest to (0, 1)); `val` uses the validation set |
| `--n_bootstraps` | `1000` | bootstrap resamples for 95% CIs |

Outputs in `--output_dir`:

- `test_metrics_summary.csv` and `test_metrics.json`: AUROC, AUPRC, AUPRC relative to prevalence, sensitivity, specificity, PPV, NPV and F1 with 95% CIs, plus the mean across outcomes
- `test_predictions.csv`: patient-level predicted probabilities
- `figures/`: ROC and calibration plots for each outcome
- `training_logs/`: the hyperparameter search history
- `models/`: saved models (`--save_models`)
- `run_config.json` and `run.log`

`helpers.PlotROCCurve(probs, y_true, ci=95, random_seed=0)` is kept for compatibility with the original analysis code. It plots the ROC curve and returns the point estimates and bootstrap CIs for AUROC, AUPRC, sensitivity and specificity.

## 3. Baseline: zero-shot LLM

```bash
python baseline_evaluation/llm_baseline.py \
    --notes data/notes.csv --labels data/labels.csv \
    --split column --split_col split --test_value test \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir results/llama31_8b
```

Each outcome is asked as a separate yes/no question. The probability of answering "1" vs "0" is read from the next-token distribution, which gives deterministic answers (for sensitivity and specificity) and a continuous score (for AUROC/AUPRC). Bio_ClinicalBERT and BioBERT baselines can be run with `finetune.py` by changing `--model_name_or_path` (e.g. `emilyalsentzer/Bio_ClinicalBERT`, `dmis-lab/biobert-base-cased-v1.2`).

## Notes for external validation

- Use the same prediction window (first 24 h), the same exclusions (no notes, missing outcomes, death within 24 h) and outcome definitions as close as possible to the paper's appendix.
- Report outcome prevalence at your site. AUPRC depends strongly on prevalence, so also report AUPRC relative to a no-skill classifier (`auprc_relative`).
- In the paper, separate models were fine-tuned for the primary and external cohorts. Use `--eval_only` if you also want to measure transportability of a model fine-tuned elsewhere.
- Hyperparameters in the paper were tuned manually on a validation subset. `finetune.py` automates this with a small grid, so results may differ slightly from the published numbers.

## Citation

```bibtex
@article{xie2025neonatalbert,
  title   = {Development and validation of a pre-trained language model for neonatal morbidities: a retrospective, multicentre, prognostic study},
  author  = {Xie, Feng and Chung, Philip and Reiss, Jonathan D and Tjoa, Erico and De Francesco, Davide and Phongpreecha, Thanaphong and Haberkorn, William and Chakraborty, Dipro and Chang, Alan Lee and James, Tomin and Kim, Yeasul and Mataraso, Samson and Espinosa, Camilo and Yang, Liu and Shu, Chi-Hung and Xue, Lei and Berson, Elo{\"i}se and Mohammadi, Neshat and Shome, Sayane and Reincke, S Momsen and Ghanem, Marc and Maric, Ivana and Gaudilliere, Brice and Angst, Martin S and Sylvester, Karl and Shaw, Gary M and Prince, Lawrence S and Stevenson, David K and Aghaeepour, Nima},
  journal = {The Lancet Digital Health},
  volume  = {7},
  number  = {12},
  year    = {2025},
  doi     = {10.1016/j.landig.2025.100926}
}
```
