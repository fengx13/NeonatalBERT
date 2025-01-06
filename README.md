# NeonatalBERT Workflow for Predicting Multiple Neonatal Outcomes from Clinical Notes

This repository provides a comprehensive Python-based workflow for developing and validating the NeonatalBERT model for predicting multiple neonatal outcomes from clinical notes.

## General Information

Clinical decision-making in neonatal care is crucial for improving outcomes during the early stages of life. Machine learning (ML) models, particularly those trained on clinical data, have significant potential to assist in neonatal care. With the increasing adoption of Electronic Health Records (EHR) for both neonatal and maternal data, there is an opportunity to leverage these records for more effective clinical predictions.

Our dataset comprises over **10 million neonatal-related clinical notes** from approximately **40,000 mother-baby pairs** treated at Stanford-affiliated hospitals. This dataset represents a valuable resource for research and model development.

However, a key challenge remains: the need for publicly available benchmark datasets and models to enable reproducible and comparable research. This repository addresses that challenge by providing a workflow that constructs a benchmark dataset for NeonatalBERT and evaluates baseline models for predicting neonatal outcomes.

## Repository Structure

The repository is organized as follows:

- `model_pretraining/`: Scripts for developing and pretraining the NeonatalBERT model.
- `model_fine_tune/`: Scripts for fine-tuning the NeonatalBERT model using internal and external datasets.
- `baseline_evaluation/`: Scripts for evaluating baseline models for comparison with NeonatalBERT.

## Requirements and Setup

Please note that the **Stanford newborn dataset** used in this workflow is not included in this repository and requires relevant institutional approval. The **BIDMC newborn dataset** can be requested from [PhysioNet MIMIC-III (v1.4)](https://physionet.org/content/mimiciii/1.4/).

### Prerequisites

- Python 3.x
- PyTorch and Hugging Face's `transformers` package
- Standard machine learning and data processing libraries (`numpy`, `pandas`, `scikit-learn`)
- GPU for efficient training and validation (highly recommended; without GPU, the process may be very slow)

## Workflow Overview

### 1. Model Pretraining
To pretrain NeonatalBERT using clinical notes:

```bash
python format_for_bert_children.py --input_path /path/to/input.pkl --output_name data_neo1100k_hp
```

**Arguments:**
- `--input_path`: Path to the input pickle file containing the preprocessed clinical notes.
- `--output_name`: Output filename suffix for the pretraining data.

**Output:**
- Pretraining dataset formatted for BERT input.

**Details:**
This step processes raw clinical notes and formats them into a suitable structure for BERT-based pretraining. Ensure that the data input path points to a valid `.pkl` file.

### 2. Model Fine-tuning and Validation
To fine-tune NeonatalBERT on prediction tasks for neonatal outcomes using two datasets:

```bash
python fine_tune_model.py --train_path /path/to/train.pkl --val_path /path/to/val.pkl --epochs 3 --output_dir /path/to/save
```

**Arguments:**
- `--train_path`: Path to the training dataset.
- `--val_path`: Path to the validation dataset.
- `--epochs`: Number of fine-tuning epochs.
- `--output_dir`: Directory to save the fine-tuned model.

**Output:**
- Fine-tuned NeonatalBERT model saved in the specified directory.

**Details:**
This step fine-tunes NeonatalBERT on clinical prediction tasks related to neonatal outcomes, such as mortality prediction, length of stay, and diagnosis-related outcomes.

### 3. Baseline Model Comparison
To compare NeonatalBERT with other large language models (LLMs), such as LLaMA-3.1-8B-Instruct:

**Script:**

```bash
python evaluate_baseline.py --model_name llama-3.1-8b --data_path /path/to/test.pkl --metrics accuracy f1
```

**Arguments:**
- `--model_name`: Name of the baseline model (e.g., `llama-3.1-8b`).
- `--data_path`: Path to the test dataset.
- `--metrics`: Evaluation metrics (e.g., `accuracy`, `f1`).

**Output:**
- Evaluation scores for baseline models.

**Details:**
This step compares NeonatalBERT with other baseline models using specified metrics such as accuracy and F1 score.

## Citation
This project is currently under review. Please refer to the repository for updates on the publication status.

