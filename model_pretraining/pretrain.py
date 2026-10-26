#!/usr/bin/env python
"""
Step 3 of NeonatalBERT pretraining: continue pretraining a BERT model
(NeonatalBERT started from Bio_ClinicalBERT) with the original BERT objectives,
masked language modelling (MLM) and next sentence prediction (NSP), on the
instances produced by create_pretraining_data.py.

The model is saved with `save_pretrained` after every epoch, so each
checkpoint directory can be passed directly to model_fine_tune/finetune.py via
--model_name_or_path.

Usage
-----
python model_pretraining/pretrain.py \
    --pregenerated_data corpus/pregenerated \
    --bert_model emilyalsentzer/Bio_ClinicalBERT \
    --output_dir checkpoints/neonatalbert \
    --epochs 3 --train_batch_size 32 --learning_rate 5e-5 --fp16
"""

import json
import logging
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import BertForPreTraining, BertTokenizer, get_linear_schedule_with_warmup

logger = logging.getLogger("neonatalbert.pretrain")


class PregeneratedDataset(Dataset):
    """Loads one epoch_{i}.json (JSON Lines) file and converts it to id arrays."""

    def __init__(self, data_dir: Path, tokenizer, epoch: int, num_data_epochs: int):
        data_epoch = epoch % num_data_epochs
        data_file = data_dir / f"epoch_{data_epoch}.json"
        metrics = json.loads((data_dir / f"epoch_{data_epoch}_metrics.json").read_text())
        n, seq_len = metrics["num_training_examples"], metrics["max_seq_len"]
        logger.info("Loading %d instances from %s", n, data_file)

        self.input_ids = np.zeros((n, seq_len), dtype=np.int32)
        self.attention_mask = np.zeros((n, seq_len), dtype=np.int8)
        self.token_type_ids = np.zeros((n, seq_len), dtype=np.int8)
        self.labels = np.full((n, seq_len), -100, dtype=np.int32)  # -100 = ignored by the MLM loss
        self.next_sentence_label = np.zeros(n, dtype=np.int64)

        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=n, desc=f"Epoch {epoch} data")):
                ex = json.loads(line)
                ids = tokenizer.convert_tokens_to_ids(ex["tokens"])
                L = len(ids)
                self.input_ids[i, :L] = ids
                self.attention_mask[i, :L] = 1
                self.token_type_ids[i, :L] = ex["segment_ids"]
                if ex["masked_lm_positions"]:
                    self.labels[i, ex["masked_lm_positions"]] = tokenizer.convert_tokens_to_ids(ex["masked_lm_labels"])
                # Hugging Face convention: 0 = B follows A, 1 = B is random
                self.next_sentence_label[i] = int(ex["is_random_next"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": torch.as_tensor(self.input_ids[i], dtype=torch.long),
            "attention_mask": torch.as_tensor(self.attention_mask[i], dtype=torch.long),
            "token_type_ids": torch.as_tensor(self.token_type_ids[i], dtype=torch.long),
            "labels": torch.as_tensor(self.labels[i], dtype=torch.long),
            "next_sentence_label": torch.as_tensor(self.next_sentence_label[i], dtype=torch.long),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_instances(data_dir: Path, epochs: int, num_data_epochs: int) -> int:
    total = 0
    for e in range(epochs):
        m = json.loads((data_dir / f"epoch_{e % num_data_epochs}_metrics.json").read_text())
        total += m["num_training_examples"]
    return total


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--pregenerated_data", type=Path, required=True)
    ap.add_argument("--bert_model", type=str, required=True, help="Starting checkpoint, e.g. emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--do_lower_case", action="store_true", default=None,
                    help="Force lower-casing (default: use the tokenizer's own setting)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_proportion", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_data_epochs = len(list(args.pregenerated_data.glob("epoch_*_metrics.json")))
    if num_data_epochs == 0:
        raise SystemExit(f"No pregenerated data found in {args.pregenerated_data}")
    if num_data_epochs < args.epochs:
        logger.warning("Only %d data epochs for %d training epochs; data will be reused", num_data_epochs, args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    tok_kwargs = {} if args.do_lower_case is None else {"do_lower_case": True}
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, **tok_kwargs)
    model = BertForPreTraining.from_pretrained(args.bert_model).to(device)
    train_model = torch.nn.DataParallel(model) if n_gpu > 1 else model

    total_examples = count_instances(args.pregenerated_data, args.epochs, num_data_epochs)
    num_steps = max(1, total_examples // (args.train_batch_size * args.gradient_accumulation_steps))
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_proportion * num_steps), num_steps)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger.info("Device %s (n_gpu=%d) | examples=%d | optimisation steps=%d", device, n_gpu, total_examples, num_steps)
    global_step = 0
    for epoch in range(args.epochs):
        dataset = PregeneratedDataset(args.pregenerated_data, tokenizer, epoch, num_data_epochs)
        loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.train_batch_size, num_workers=args.num_workers)
        train_model.train()
        running, n_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = train_model(**batch).loss
            if loss.dim() > 0:  # DataParallel returns one loss per GPU
                loss = loss.mean()
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running += loss.item() * args.gradient_accumulation_steps
            n_batches += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            pbar.set_postfix(loss=f"{running / n_batches:.4f}")

        logger.info("Epoch %d finished | mean loss %.4f | steps %d", epoch + 1, running / max(n_batches, 1), global_step)
        ckpt = args.output_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        logger.info("Saved checkpoint to %s", ckpt)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Final model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
