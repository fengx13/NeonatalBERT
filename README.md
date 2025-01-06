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

- `model_preatraining/...` contains the scripts for developing and training the NeonatalBERT model
- `model_fine_tune/...` contains the scripts for fine-tuning the NeonatalBERT model using both internal and external data
- `baseline_evaluation/...` contains the scripts for evaluating other baseline models for comparison
- 
## Requirements and Setup

The Stanford newborn dataset is not included in this repository and must be obtained through relevant institutional approval. The BIDMC newborn dataset can be obtained on MIMIC data requested through https://physionet.org/content/mimiciii/1.4/

### Prerequisites:
1. Python 3.x
2. Pytorch and transformer package installed
3. Other basic machine learning and data processing packages
4. GPU is recommended, otherwise the training and validation process will be very slow

## Workflow

### 1. Model pretraining
~~~
python format_for_bert_children.py --input_path /path/to/input.pkl --output_name data_neo1100k_hp
~~~

**Arguments:**

**Output:**

**Details:**


### 2. Model validation using two datasets for downstream tasks (predicting multiple neonatal outcomes)
~~~
python
~~~

**Arguments:**


**Output:**


**Details:**


### 3. Baseline model: comparison with LLMs (Llama-3.1-8B-Instruct)


**Arguments:**


**Output:**


**Details:**


## Citation

Under revision
