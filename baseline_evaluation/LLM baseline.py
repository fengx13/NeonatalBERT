import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import os
import sys
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

import os
import time
import random
#import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from helpers import PlotROCCurve

confidence_interval = 95
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.environ['CUDA_VISIBLE_DEVICES']= "3" #0,1,2,3, // -1
DEVICE= torch.device("cuda")


## read child's notes data
j=1
df_note = pd.read_pickle("/remote/home/fengx/fengx/data/note_child_final.pkl")
df_note = df_note[(df_note["_note_class_name"]== "History and physical note")]
df_note.drop_duplicates(subset=['note_text'], keep='first', inplace=True)
print("Working on difftime limit:",j)
print("Inluding number of notes:",len(df_note))
df_note.shape

df_note.reset_index(drop=True,inplace=True)
data = pd.read_csv("/remote/home/fengx/fengx/data/data_stru_ready_2021.csv")

##merge data
data = pd.merge(data, df_note, left_on='child_id',right_on= "person_id", how='left')
## delete NAN notes:

np.isnan(data["rds"]).sum()


df_test = data[data["birth_year"]==2021]

print(' testing size =', len(df_test))


df_cleaned = df_test.dropna(subset=['note_text'])

df_cleaned['person_id'].duplicated().sum()
df_cleaned = df_cleaned.drop_duplicates(subset='person_id', keep='first')


### prepare LLM models
from transformers import AutoTokenizer, AutoModelForCausalLM
import outlines

# Model name (e.g., LLaMA-2 7B)
model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name ="meta-llama/Meta-Llama-3-70B"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model and map to the specified GPU
#model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
#    model_name,
#    device_map = None,  # Avoid automatic mapping
#    torch_dtype=torch.float16  # Use FP16 precision for efficiency
#)

# Move model to the specific GPU
#model = model.to(DEVICE)

model = outlines.models.transformers(model_name, device=DEVICE)
# Define choices for 1 and 0
sampler = outlines.samplers.multinomial(1)
generator = outlines.generate.choice(model, ["1", "0"], sampler)
#choices = outlines.choice(["1", "0"])

# Connect generator to Hugging Face model

questions = [
    "Will the neonatal patient develop Candidiasis?",
    "Will the neonatal patient develop Bronchopulmonary Dysplasia (BPD)?",
    "Will the neonatal patient develop Retinopathy of Prematurity (ROP)?",
    "Will the neonatal patient develop Periventricular Leukomalacia(PVL)?",
    "Will the neonatal patient develop respiratory distress syndrome (RDS)?",
    "Will the neonatal patient develop intraventricular hemorrhage (IVH)?",
    "Will the neonatal patient develop necrotizing enterocolitis (NEC)?",
    "Is the neonatal patient at risk of mortality?"
]


###########################
def generate_combined_prompt(note_text, questions):
    prompt = note_text + " \n\n"
    for i, question in enumerate(questions, start=1):
        prompt += f"Question {i}: {question} 1. Yes 0. No.\n"
    prompt += "Please respond with the number corresponding to the correct answer for each question in order."
    return prompt

#Please respond with the number corresponding to the correct answer.Answer 1 if yes, 0 if no. Answer with only the number. Answer with only the number 0 or 1"

# Create prompts for each note
df_cleaned["prompt"] = df_cleaned["note_text"].apply(lambda x: generate_combined_prompt(x, questions))

# Function to generate predictions for multiple questions
def generate_predictions(prompt):
    responses = generator(prompt)
    return responses.strip().split()  # Split answers to ensure they are for each question

# Generate predictions for all prompts
df_cleaned["predictions"] = df_cleaned["prompt"].apply(generate_predictions)

# Save results to CSV
df_cleaned.to_csv("df_cleaned_llama3_multi_questions.csv", index=False)
df_cleaned[["sepsis", "rds", "ivh", "nec", "mortality"]] = pd.DataFrame(df_cleaned["predictions"].tolist(), index=df_cleaned.index)

# Save updated DataFrame
df_cleaned.to_csv("df_cleaned_llama3_multi_questions_2.csv", index=False)