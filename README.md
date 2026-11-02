# NeonatalBERT: a pretrained language model for predicting neonatal morbidities from clinical notes

**Update (version 2, October 27, 2026):** The code has been reorganised with easier access and optimized to run with current package versions. To improve reproducibility, we also added a script that generates synthetic clinical notes, so the full pipeline can be run and checked without access to any patient data (see [Quick start on synthetic data](#quick-start-on-synthetic-data-a-few-minutes-on-a-cpu)).

This repository contains the code for **NeonatalBERT**, a domain-specific language model for neonatal clinical notes, described in:

> Xie F, Chung P, Reiss JD, et al. **Development and validation of a pre-trained language model for neonatal morbidities: a retrospective, multicentre, prognostic study.** *Lancet Digit Health* 2025; 7(12). doi:[10.1016/j.landig.2025.100926](https://doi.org/10.1016/j.landig.2025.100926) · [PMC12748047](https://pmc.ncbi.nlm.nih.gov/articles/PMC12748047/)

NeonatalBERT was initialised from Bio_ClinicalBERT and further pretrained (masked language modelling + next sentence prediction) on neonatal notes written during the first 3 months of life. It was then used to predict neonatal outcomes from notes written on the **delivery day (first 24 hours after birth)**:

- **Primary cohort (Stanford, 2014–2021, n = 32,321):** 19 outcomes, temporal split (births 2014–2020 for development, 2021 for testing). Mean AUROC 0.910, mean AUPRC 0.291.
- **External cohort (BIDMC/MIMIC-III, n = 7,061):** 10 outcomes, random 80/20 patient-level split. Mean AUROC 0.935, mean AUPRC 0.360.

---

## ⚠️ Availability of data

**The pretrained NeonatalBERT weights are not publicly available.** NeonatalBERT was pretrained on clinical notes from the Stanford Research Data Repository (STARR). These notes contain protected health information, and language models can memorise parts of their training text, so the weights fall under the same institutional and patient-privacy restrictions as the Stanford data. Neither the Stanford data nor models trained on it can be released through open access.

What you can do at your own institution:

1. **Fine-tuning-only mode (no NeonatalBERT weights needed).** Run `model_fine_tune/finetune.py` with a public clinical encoder such as [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) on your own first-day notes. This is the "fine-tuning-only mode" for resource-limited settings described in the paper, and corresponds to the Bio_ClinicalBERT comparator.
2. **Re-create NeonatalBERT locally.** Run `model_pretraining/pretrain_local.py` to pretrain on your own neonatal notes, starting from Bio_ClinicalBERT, with one command (see [Option A](#option-a-one-command-pretraining-at-your-site-recommended)). Then fine-tune the result. This follows the paper's pretraining approach (masked language modelling + next sentence prediction starting from Bio_ClinicalBERT); see [Relationship to the published analysis](#relationship-to-the-published-analysis) regarding hyperparameters.
3. **Contact us or collaborate with us.** Please contact Feng Xie (current email: xie00469@umn.edu).

The **external MIMIC-III (BIDMC) data** are available to credentialed users through [PhysioNet](https://physionet.org/content/mimiciii/1.4/). `data_preparation/build_mimic3_cohort.py` builds the newborn cohort following the paper's definitions, using the ICD-9 codes from appendix eTable 2.

## Relationship to the published analysis

The analyses reported in the paper were conducted within Stanford's secure computing environment. This repository is a cleaned-up, documented implementation of that workflow, prepared so that other institutions can apply it to their own data. Its defaults follow the methods described in the paper and its appendix (eMethods):

| Paper / appendix | This repository |
|---|---|
| Domain-adaptive pretraining of Bio_ClinicalBERT with MLM + NSP on notes from the first 3 months of life, training-set patients only | `model_pretraining/` (`pretrain_local.py` holds out test patients automatically) |
| Input: all notes written on the delivery day; all note types combined (appendix eFigure 3) | `--max_hours 24`, all note types by default |
| Each note encoded separately; notes longer than 512 tokens truncated to the first 512 tokens (default) or split into 512-token sections with mean/max pooling (eTable 9) | `--long_note truncate` (default), `mean`, `max` |
| Patient representation: mean of the note embeddings (final hidden state) | mean of the [CLS] embeddings of a patient's notes |
| Classification layer with one output per outcome, trained with a class-weighted loss on top of NeonatalBERT | `--mode frozen` (default): frozen encoder, one linear output per outcome |
| Learning rate, batch size, weight decay and class weights tuned by grid search on a validation subset of the training data, with early stopping | grid search with early stopping; the best setting is chosen per outcome |
| Cases diagnosed on the delivery day excluded from analysis | per-outcome exclusion via `<outcome>_dx_hours` |
| Outcome definitions: SNOMED/OMOP concepts (eTable 1) and ICD-9 codes (eTable 2) | `configs/stanford_omop_outcome_concepts.json`, `configs/mimic3_icd9_outcomes.json` |
| AUROC, AUPRC (also relative to a no-skill classifier), F1, sensitivity, specificity; bootstrap 95% CIs; decile calibration plots | `helpers.py` |
| Baselines: logistic regression on first-day structured features; TF-IDF + SVM/XGBoost; BioBERT and Bio_ClinicalBERT; zero-shot Llama-3.1-8B-Instruct with constrained 0/1 output | `baseline_evaluation/`, and `finetune.py` with another encoder |

Settings that the paper does not report are set to reasonable defaults and may differ from the original analysis: the pretraining hyperparameters (learning rate, batch size, sequence length, number of epochs), the specific values searched in the fine-tuning grid, the method used to choose the threshold for sensitivity and specificity, and the number of bootstrap resamples. Results obtained with this code may therefore differ somewhat from the published numbers. Differences in cohorts, documentation practices, outcome definitions and software versions (the paper used Python 3.9.16, Transformers 4.25.1 and PyTorch 2.0.1) also contribute. The Stanford cohort extraction depends on STARR and is not included.

If you find a discrepancy or have a question, please open an issue.

---

## Repository structure

```
NeonatalBERT/
├── helpers.py                         # metrics, bootstrap CIs, ROC/calibration plots (PlotROCCurve)
├── data_utils.py                      # loading notes/labels, prediction window, exclusions, patient-level splits
├── data_preparation/
│   └── build_mimic3_cohort.py         # external BIDMC newborn cohort from MIMIC-III v1.4
├── configs/
│   ├── mimic3_icd9_outcomes.json      # ICD-9 codes for the MIMIC-III outcomes (appendix eTable 2)
│   └── stanford_omop_outcome_concepts.json  # concept ids for the 19 Stanford outcomes (appendix eTable 1)
├── model_pretraining/
│   ├── prepare_notes.py               # step 1: notes -> sentence-per-line corpus
│   ├── create_pretraining_data.py     # step 2: MLM + NSP training instances
│   ├── pretrain.py                    # step 3: domain-adaptive pretraining (MLM + NSP)
│   └── pretrain_local.py              # one command: steps 1-3 at your site, with test patients held out
├── model_fine_tune/
│   └── finetune.py                    # note embeddings, classification layer, hyperparameter search, evaluation
├── baseline_evaluation/
│   ├── classical_baselines.py         # structured-feature logistic regression, TF-IDF + SVM / XGBoost
│   └── llm_baseline.py                # zero-shot Llama-3.1-8B-Instruct
├── examples/
│   ├── make_synthetic_data.py         # synthetic NICU notes, labels and features for a smoke test (no real data)
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
pip install xgboost   # optional, only for the TF-IDF + XGBoost baseline
```

A GPU is strongly recommended for pretraining (the paper's pretraining took 10 days on one NVIDIA A40) and for `--mode finetune`. The default `--mode frozen` only needs one forward pass of the encoder per note, then trains a small classification layer, so it is much lighter.

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
| `<outcome>_dx_hours` | recommended | hours from birth to diagnosis for positive cases (e.g. `rds_dx_hours`); cases diagnosed within the prediction window are excluded for that outcome, as in the paper |
| split column | optional | e.g. `split` with values `train` / `test`, for a predefined (e.g. temporal) split |

Each note is encoded separately and a patient's note embeddings are averaged. Patients without eligible notes or with missing outcome data are excluded automatically. The paper also excluded infants who died within 24 h of birth; remove them when you build the labels table (`build_mimic3_cohort.py` does this). Patient ids are matched as text, the labels table must have exactly one row per patient, and the scripts stop with an error if notes and labels share no patient ids.

> **Avoid leakage:** do not include discharge summaries or any note written after the prediction time as model input. If you cannot provide `<outcome>_dx_hours`, the scripts warn that cases diagnosed on the delivery day cannot be excluded.

## Quick start on synthetic data (a few minutes on a CPU)

```bash
python examples/make_synthetic_data.py --out_dir examples/synthetic
python examples/make_tiny_bert.py --corpus examples/synthetic/notes.csv --out_dir examples/tiny_bert

python model_fine_tune/finetune.py \
    --notes examples/synthetic/notes.csv --labels examples/synthetic/labels.csv \
    --outcomes rds,bpd,ivh,nec,sepsis,jaundice,death \
    --model_name_or_path examples/tiny_bert --max_length 128 \
    --split column --split_col split --test_value test \
    --output_dir results/synthetic_demo
```

The tiny model is randomly initialised, so the numbers are meaningless. This only checks that the pipeline runs. With internet access you can replace `examples/tiny_bert` with `emilyalsentzer/Bio_ClinicalBERT`.

## 1. Domain-adaptive pretraining

The paper pretrained Bio_ClinicalBERT on 524,018 notes from the first 3 months of life of **training-set patients only**. Test patients must never be seen during pretraining. The paper does not report the pretraining hyperparameters; the defaults below are common choices for BERT domain adaptation.

### Option A: one-command pretraining at your site 

Use this when you cannot obtain the original NeonatalBERT weights. `pretrain_local.py` builds a NeonatalBERT-style model from your own notes. It:

1. defines the held-out test patients and removes **all** of their notes from the pretraining corpus
2. keeps notes from the first `--pretrain_max_days` days of life (default 90, as in the paper)
3. runs the three pretraining steps below
4. prints the matching fine-tuning command

```bash
python model_pretraining/pretrain_local.py \
    --notes data/notes.csv --labels data/labels.csv \
    --split random --test_size 0.2 \
    --base_model emilyalsentzer/Bio_ClinicalBERT \
    --work_dir pretraining_runs/site_x \
    --epochs 3 --train_batch_size 32 --learning_rate 5e-5 --fp16
```

- **Random split:** the train/test assignment is saved to `<work_dir>/labels_with_split.csv` (column `neonatalbert_split`). Fine-tune with that file and `--split column --split_col neonatalbert_split` so the evaluation uses exactly the patients held out from pretraining. The printed command already does this.
- **Predefined (e.g. temporal) split:** use `--split column --split_col split --test_value test`.
- **Resuming:** the model is written to `<work_dir>/model`. Finished steps are skipped when you re-run, so an interrupted job can resume. If the held-out patients or settings have changed since the previous run, the script stops instead of reusing it; use a new `--work_dir` or add `--overwrite`.
- **Privacy:** the intermediate files in `<work_dir>` contain note text. Keep the folder inside your secure computing environment.

### Option B: run the steps individually

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

By default (`--mode frozen`), `finetune.py` follows the paper's protocol:

1. It encodes every note separately with NeonatalBERT and takes the final hidden state of [CLS] as the note embedding. Notes longer than 512 tokens are truncated by default.
2. It averages each patient's note embeddings into one patient representation.
3. It trains a classification layer with one output per outcome, using a class-weighted loss.
4. It runs a grid search over learning rate, batch size, weight decay and class weights on a validation subset of the development data, with early stopping. The best setting is chosen per outcome.
5. It evaluates on the held-out test set with 95% bootstrap confidence intervals.

**Primary-cohort style (temporal split defined in the labels file):**

```bash
python model_fine_tune/finetune.py \
    --notes data/notes.csv --labels data/labels.csv \
    --outcomes rds,bpd,ivh,nec,sepsis,rop,death \
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
    --outcomes rds,bpd,pulmonary_haemorrhage,ivh,pda,sepsis,jaundice,nec,rop,death \
    --model_name_or_path checkpoints/neonatalbert \
    --split random --test_size 0.2 --save_models --fp16 \
    --output_dir results/mimic3
```

MIMIC-III does not record diagnosis times, so cases diagnosed on the delivery day cannot be excluded there; the script logs a warning.

**Applying saved models to a new site without re-fitting (transportability):**

```bash
python model_fine_tune/finetune.py \
    --notes site_b/notes.csv --labels site_b/labels.csv --outcomes rds,bpd,nec,death \
    --model_name_or_path /path/to/the/same/encoder \
    --eval_only --checkpoint_dir results/primary --output_dir results/primary_to_site_b
```

Key options:

| option | default | description |
|---|---|---|
| `--outcomes` | required | comma-separated binary outcome columns in the labels table |
| `--mode` | `frozen` | `frozen` (paper protocol: frozen encoder + classification layer), `embedding_lr` (frozen encoder + scikit-learn logistic regression per outcome), `finetune` (end-to-end: the encoder is also updated; needs a GPU) |
| `--max_hours` | `24` | prediction window after birth (eTable 6 reports days 1–7: use `48`, `72`, … `168`) |
| `--note_types` | all | restrict input to these note types, e.g. `"History and physical"` (eFigure 3) |
| `--long_note` | `truncate` | notes > 512 tokens: keep the first 512 tokens, or split into 512-token sections and `mean`/`max` pool them (eTable 9; `--max_sections`, default 8) |
| `--lr_grid` | `1e-3,1e-4` (frozen), `2e-5,5e-5` (finetune) | learning rates searched |
| `--batch_size_grid`, `--weight_decay_grid` | `32,128`, `0,0.01` | batch sizes and weight decay searched (frozen mode) |
| `--class_weight_grid` | `none,sqrt,balanced` | positive-class weights searched: 1, √(neg/pos), neg/pos |
| `--epochs`, `--patience` | `50`, `5` (frozen); `4`, `2` (finetune) | maximum epochs and early-stopping patience |
| `--head` | `multilabel` | `--mode finetune` only: one shared model with one output per outcome, or `per_outcome` models |
| `--threshold_from` | `val` | where the threshold for sensitivity/specificity/F1 is chosen (ROC point closest to (0, 1)): the validation set (out-of-fold predictions in `embedding_lr` mode) or `test` (optimistic). A warning is logged if the validation set cannot be used |
| `--n_bootstraps` | `1000` | bootstrap resamples for 95% CIs |

Outputs in `--output_dir`:

- `test_metrics_summary.csv` and `test_metrics.json`: AUROC, AUPRC, AUPRC relative to prevalence, sensitivity, specificity, PPV, NPV and F1 with 95% percentile-bootstrap CIs, plus the mean across outcomes (the MEAN row states how many outcomes it covers; outcomes with a single class in the test set are skipped)
- `test_predictions.csv`: patient-level predicted probabilities
- `figures/`: ROC curves and decile calibration plots for each outcome
- `training_logs/`: the hyperparameter search history and the setting selected for each outcome
- `models/`: saved models (`--save_models`)
- `run_config.json` and `run.log`

`helpers.PlotROCCurve(probs, y_true, ci=95, random_seed=0)` is kept for compatibility with the original analysis code. It plots the ROC curve and returns the point estimates and bootstrap CIs for AUROC, AUPRC, sensitivity and specificity.

BioBERT and Bio_ClinicalBERT comparators: run `finetune.py` with `--model_name_or_path dmis-lab/biobert-base-cased-v1.2` or `emilyalsentzer/Bio_ClinicalBERT` on the same split.

## 3. Baselines

**Structured features and TF-IDF (appendix eTables 1 and 8):**

```bash
# Logistic regression on first-day structured features (numeric columns; missing values median-imputed)
python baseline_evaluation/classical_baselines.py --baseline tabular_lr --features data/tabular.csv \
    --notes data/notes.csv --labels data/labels.csv --outcomes rds,bpd,nec,death \
    --split column --split_col split --test_value test --output_dir results/tabular_lr

# TF-IDF of the first-day notes + linear SVM (or --baseline tfidf_xgboost)
python baseline_evaluation/classical_baselines.py --baseline tfidf_svm \
    --notes data/notes.csv --labels data/labels.csv --outcomes rds,bpd,nec,death \
    --split column --split_col split --test_value test --output_dir results/tfidf_svm
```

These use the same patients, exclusions and split as `finetune.py`. The structured variables used in the paper are listed in appendix eTable 1.

**Zero-shot Llama-3.1-8B-Instruct (appendix "Implementation of Llama-3.1-8B"):**

```bash
python baseline_evaluation/llm_baseline.py \
    --notes data/notes.csv --labels data/labels.csv \
    --outcomes rds,bpd,ivh,nec,sepsis,rop,death \
    --note_types "History and physical" \
    --split column --split_col split --test_value test \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir results/llama31_8b
```

As in the appendix, each outcome is asked separately after the patient's notes ("Will this neonatal patient develop sepsis? Answer 1 if yes, 0 if no. Answer with only the number."), and the output is constrained to "1" or "0". The paper used the Outlines package for this. Here the constraint is applied directly to the probabilities of the two answer tokens, and the answer is sampled from them (`--decoding sample`, as in the original analysis) or taken greedily (`--decoding greedy`). Sensitivity and specificity are computed from the answers; AUROC/AUPRC from the answer probability are reported in addition.

## Notes for external validation

- Use the same prediction window (first 24 h), the same exclusions (no notes, missing outcomes, death within 24 h, cases diagnosed on the delivery day) and outcome definitions as close as possible to the appendix (`configs/`).
- Report outcome prevalence at your site. AUPRC depends strongly on prevalence, so also report AUPRC relative to a no-skill classifier (`auprc_relative`).
- In the paper, separate models were fine-tuned for the primary and external cohorts. Use `--eval_only` if you also want to measure transportability of a model fine-tuned elsewhere.
- Splits are random at the patient level and not stratified. With rare outcomes, check the number of positive cases in the training, validation and test sets (logged at the start of each run) and increase `--val_size` if the log warns that the validation set has a single class.

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
