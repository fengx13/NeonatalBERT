NeonatalBERT Workflow for Predicting Multiple Neonatal Outcomes from Clinical Notes
=========================

Python workflow for developing and validating NeonatalBERT model for Predicting Multiple Neonatal Outcomes from Clinical Notes

## General info

Clinical decisions in neonatal care are critical for optimizing patient outcomes during the crucial early stages of life. Machine learning-based clinical prediction models have the potential to enhance neonatal care.

In parallel to the rise of clinical prediction models, the adoption of Electronic Health Records (EHR) for neonatal and maternal patient data has grown significantly. Our dataset, based on over 10 million neonatal-related clinical notes from approximately 40,000 mother-baby pairs treated at Stanford-affiliated hospitals, provides a rich resource for research.

There is a pressing need for publicly available benchmark datasets and models that allow researchers to produce comparable and reproducible results.

Here, we present a workflow that generates a benchmark dataset for NeonatalBERT and constructs benchmark models for prediction tasks related to neonatal outcomes.

## Structure

The structure of this repository is detailed as follows:

- `Benchmark_scripts/...` contains the scripts for benchmark dataset generation (`master_data.csv`).
- `Benchmark_scripts/...` contains the scripts for building the various task-specific benchmark models.

## Requirements and Setup

The NeonatalBERT dataset is not provided with this repository and must be obtained through relevant institutional approval.

### Prerequisites:
1. Python 3.x
2. Required packages listed in `requirements.txt`
3. Institutional access to the NeonatalBERT dataset.

**Note:** Upon downloading the dataset, place it in the appropriate directory structure as specified.

## Workflow

The following sub-sections describe the sequential modules within the NeonatalBERT workflow and how they should ideally be run.

Prior to these steps, ensure the repository and NeonatalBERT data have been set up locally.

### 1. Model pretraining
~~~
python extract_master_dataset.py --data_path {data_path} --output_path {output_path}
~~~

**Arguments:**

- `data_path`: Path to the directory containing NeonatalBERT-related data.
- `output_path`: Path to output directory.
- `icu_transfer_timerange`: Timerange in hours for ICU transfer outcome. Default set to 12.
- `readmission_timerange`: Timerange in days for readmission prediction. Default set to 7.

**Output:**

`master_dataset.csv` output to `output_path`

**Details:**

The input neonatal-related notes and records are taken to form the root table, with `subject_id` as the unique identifier for each baby and `stay_id` as the unique identifier for each visit. This root table is then enriched with additional clinical variables for each patient.

A total of **81** variables are included in `master_dataset.csv` (Refer to Table 3 of the article for a full variable list).

### 2. Model validation using two datasets for downstream tasks (predicting multiple neonatal outcomes)
~~~
python data_general_processing.py --master_dataset_path {master_dataset_path} --output_path {output_path}
~~~

**Arguments:**

- `master_dataset_path`: Path to the directory containing `master_dataset.csv`.
- `output_path`: Path to output directory.

**Output:**

`train.csv` and `test.csv` output to `output_path`

**Details:**

`master_dataset.csv` is filtered to remove invalid data points, such as duplicate entries or missing information.

Outlier values in clinical variables are detected and imputed using methods validated in related studies. The data is split into `train.csv` and `test.csv`, and additional variables for clinical scores are added.

### 3. Baseline model: comparison with LLMs (Llama-3.1-8B-Instruct)

Prediction modeling is currently handled by Python notebooks (`.ipynb` files) corresponding to each prediction task.

**Arguments:**

- `path`: Path to the directory containing `train.csv` and `test.csv`

**Output:**

`result_*.csv` and `importances_*.csv` output to `path`.

`*` denotes the task-specific wildcard string, i.e., for a neonatal sepsis prediction task, output files are `result_sepsis_prediction.csv` and `importances_sepsis_prediction.csv`.

**Details:**

For each neonatal prediction task, various models are implemented and compared. These include Logistic Regression, MLP Neural Networks, Random Forests, and NeonatalBERT-based fine-tuning. Performance metrics are compared (`result_*.csv`), and an overall variable importance ranking using Random Forests (`importances_*.csv`) is provided.

## Citation

Nan X, Smith J, Doe J, Xie F, and collaborators. Benchmarking neonatal clinical prediction models with machine learning and NeonatalBERT. Scientific Data 2024; 11: 999. <https://doi.org/10.1038/s41597-022-01782-9>
