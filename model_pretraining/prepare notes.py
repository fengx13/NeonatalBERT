import os
import sys
import pandas as pd
import spacy
import re
import time
from tqdm import tqdm
from heuristic_tokenize import sent_tokenize_rules  # Ensure this is installed or provided
import logging

## Adapted from ClinicalBERT Preprocessing Script
## This script is adapted from the [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/lm_pretraining/format_mimic_for_BERT.py) pretraining format workflow and customized for NeonatalBERT. Enhancements include support for neonatal-specific clinical notes



def configure_logging():
    """
    Configure logging to display informative messages during execution.

    This ensures that all major steps, such as data loading, processing, and saving,
    are logged for easier debugging and progress tracking.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

def read_input_data(input_path):
    """
    Reads input data from a pickle file.

    Args:
        input_path (str): Path to the input .pkl file.

    Returns:
        pd.DataFrame: Data loaded from the pickle file.

    The input file should be a pickle file containing a DataFrame with at least a 'text' column
    representing clinical notes. Optionally, there can be a 'note_id' column to uniquely identify
    notes; otherwise, an index is assigned.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist.")
        sys.exit(1)
    logging.info(f"Reading data from {input_path}")
    return pd.read_pickle(input_path)

def process_notes(data):
    """
    Process the notes to format them for BERT pretraining.

    This function takes clinical notes and applies sentence tokenization using heuristic rules.

    Args:
        data (pd.DataFrame): DataFrame containing notes with columns 'text' and optionally 'note_id'.

    Returns:
        pd.DataFrame: DataFrame containing processed notes with 'note_id' and list of tokenized sentences.

    This approach is consistent with our manuscript on NeonatalBERT, which focuses on domain-specific
    large language model pretraining for neonatal and maternal health. Tokenization quality is
    paramount to ensure the clinical context is preserved, particularly for complex neonatal notes.
    """
    nlp = spacy.blank("en")  # Initialize a blank spaCy language model for faster processing.
    processed_data = []
    
    logging.info("Starting note processing...")

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        note_id = row['note_id'] if 'note_id' in row else idx  # Use 'note_id' if available; otherwise, use index.
        note_text = row.get('text', "")  # Get the note text or default to an empty string.

        # Tokenize the note into sentences.
        sentences = sent_tokenize_rules(note_text)

        # Store the processed note in a dictionary.
        processed_note = {
            'note_id': note_id,
            'sentences': sentences,
        }
        processed_data.append(processed_note)

    logging.info("Completed note processing.")
    return pd.DataFrame(processed_data)

def save_processed_data(processed_data, output_path):
    """
    Save the processed data to a pickle file.

    Args:
        processed_data (pd.DataFrame): DataFrame containing processed notes.
        output_path (str): Path to save the processed pickle file.

    The processed data is saved in pickle format to preserve the structure of the DataFrame,
    including nested lists of sentences for each note. This format is suitable for subsequent
    large language model pretraining, as noted in the NeonatalBERT framework.
    """
    logging.info(f"Saving processed data to {output_path}")
    processed_data.to_pickle(output_path)

def main(input_path, output_name):
    """
    Main function to read, process, and save clinical notes.

    Args:
        input_path (str): Path to the input .pkl file.
        output_name (str): Suffix for the output file name.

    Example:
        python format_for_bert_children.py --input_path /path/to/input.pkl --output_name _neo1100k_hp

    This function orchestrates the entire process: it reads the data, processes notes into tokenized sentences,
    and saves the formatted data for BERT pretraining. This aligns with the workflow described in
    the NeonatalBERT manuscript, which emphasizes preprocessing steps to enhance downstream model performance.
    """
    configure_logging()
    output_path = f"processed_notes{output_name}.pkl"

    # Step 1: Read input data.
    data = read_input_data(input_path)

    # Step 2: Process notes.
    processed_data = process_notes(data)

    # Step 3: Save processed data.
    save_processed_data(processed_data, output_path)

    logging.info("All steps completed successfully.")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments for input and output paths.
    parser = argparse.ArgumentParser(description="Format clinical notes for BERT pretraining.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input pickle file."
    )
    parser.add_argument(
        "--output_name", type=str, required=True, help="Suffix for the output filename."
    )

    args = parser.parse_args()
    main(args.input_path, args.output_name)
