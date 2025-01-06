from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

"""
Script for pretraining BERT using PyTorch.

This script sets up training, data loading, and model checkpointing. It provides distributed training capabilities and supports both single and multi-GPU training.
"""

class BERTDataset(Dataset):
    """
    Custom Dataset class for BERT pretraining.

    Args:
        data (list): List of training instances.
        tokenizer (BertTokenizer): BERT tokenizer.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        inputs = self.tokenizer(item['text'], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        inputs['labels'] = torch.tensor(item['labels'], dtype=torch.long)
        return {key: val.squeeze(0) for key, val in inputs.items()}


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_logger():
    """
    Configure logging to display important information during training.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def save_checkpoint(model, optimizer, epoch, output_dir):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): BERT model.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch number.
        output_dir (Path): Directory to save the checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = output_dir / f'model_epoch_{epoch}.bin'
    optimizer_save_path = output_dir / f'optimizer_epoch_{epoch}.bin'

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    logging.info(f"Checkpoint saved at {output_dir}")


def load_data(file_path):
    """
    Load the training data from a JSON file.

    Args:
        file_path (Path): Path to the JSON data file.

    Returns:
        list: A list of training examples.
    """
    with file_path.open('r') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} training examples from {file_path}")
    return data


def train(args):
    """
    Train the BERT model for pretraining.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    set_seed(args.seed)
    initialize_logger()

    logging.info("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForPreTraining.from_pretrained(args.model_name_or_path)

    logging.info("Loading data...")
    train_data = load_data(args.train_file)
    train_dataset = BERTDataset(train_data, tokenizer, args.max_seq_length)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=len(train_dataloader) * args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info(f"Training on device: {device}")

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, args.output_dir)


def main():
    """
    Main function to parse arguments and start the training process.
    """
    parser = ArgumentParser()

    parser.add_argument('--train_file', type=Path, required=True, help='Path to the training data file.')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Pretrained BERT model name or path.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory to save checkpoints.')

    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training.')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every n epochs.')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
