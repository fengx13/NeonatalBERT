# NeonatalBERT Workflow for Predicting Multiple Neonatal Outcomes from Clinical Notes

This repository provides a comprehensive Python-based workflow for developing and validating the NeonatalBERT model for predicting multiple neonatal outcomes from clinical notes.

## General Information

This study introduces **NeonatalBERT**, the first domain-specific language model pre-trained on a large corpus of neonatal clinical notes, addressing a critical gap in neonatal care research. Unlike general clinical language models, NeonatalBERT captures nuanced neonatal-specific terminologies and clinical patterns, enabling more precise risk estimation for neonatal outcomes.

A key contribution of this study is a robust evaluation framework that includes internal and external cohorts, demonstrating the model's generalizability across healthcare institutions. By leveraging unstructured clinical notes, NeonatalBERT facilitates early risk estimation for complex neonatal outcomes, such as bronchopulmonary dysplasia, necrotizing enterocolitis, sepsis, and mortality.

This model offers a transformative tool for neonatal care by supporting early risk identification and timely interventions for at-risk neonates. Its strong performance across multiple datasets suggests potential for enhancing clinical decision-making, particularly in settings with incomplete structured data. This repository addresses that challenge by providing a workflow that constructs a benchmark dataset for NeonatalBERT and evaluates baseline models for predicting neonatal outcomes.

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
To Pretrain NeonatalBERT using clinical notes:

**Details:**
This step processes raw clinical notes and formats them into a suitable structure for BERT-based pretraining. Ensure that the data input path points to a valid `.pkl` file.

### 2. Model Fine-tuning and Validation
To fine-tune NeonatalBERT on prediction tasks for neonatal outcomes using two datasets:

**Details:**
This step fine-tunes NeonatalBERT on clinical prediction tasks related to neonatal outcomes, such as mortality prediction, and diagnosis-related outcomes.

### 3. Baseline Model Comparison
To compare NeonatalBERT with other large language models (LLMs), such as LLaMA-3.1-8B-Instruct:

**Details:**
This step compares NeonatalBERT with other baseline models using specified metrics such as sensitivity and specificity.

## Citation
This project is currently under review. Please refer to the repository for updates on the publication status.

