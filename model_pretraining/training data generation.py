from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
from multiprocessing import Pool
from random import random, randrange, randint, shuffle, choice
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import json
import collections

"""
Pretraining Data Creation Script for BERT

This script takes a corpus of text and generates training instances for BERT pretraining.
The script supports whole word masking, handles large datasets by reducing memory usage when needed,
creates masked language model (MLM) and next sentence prediction (NSP) tasks, and outputs training files in JSON format.
"""

class DocumentDatabase:
    """
    A database for storing tokenized documents, either in memory or on disk to reduce memory usage.

    Attributes:
        reduce_memory (bool): Whether to store documents on disk to reduce memory usage.
        temp_dir (TemporaryDirectory): Temporary directory for storing on-disk database.
        working_dir (Path): Path to the temporary directory.
        document_shelf_filepath (Path): Path to the shelf database file.
        document_shelf (shelve.DbfilenameShelf): On-disk storage for documents.
        documents (list): In-memory storage for documents if not using on-disk storage.
        doc_lengths (list): List of document lengths for sampling purposes.
    """

    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath), flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        """
        Adds a tokenized document to the database.

        Args:
            document (list of list of str): A document represented as a list of tokenized sentences.

        Example:
            db.add_document([["This", "is", "a", "sentence"], ["Another", "sentence"]])
        """
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        """
        Precalculate cumulative sums of document lengths for efficient sampling.
        """
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        """
        Samples a random document for Next Sentence Prediction.

        Args:
            current_idx (int): Index of the current document.
            sentence_weighted (bool): Whether to sample proportionally to sentence length.

        Returns:
            list: A sampled document.
        """
        if sentence_weighted:
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        return self.document_shelf[str(sampled_doc_index)] if self.reduce_memory else self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        return self.document_shelf[str(item)] if self.reduce_memory else self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """
    Truncates a pair of sequences to a maximum length to fit within BERT's input size.

    Args:
        tokens_a (list of str): First sequence of tokens.
        tokens_b (list of str): Second sequence of tokens.
        max_num_tokens (int): Maximum number of tokens allowed.

    The function randomly truncates from either the front or back for diversity.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """
    Creates masked language model predictions.

    Args:
        tokens (list of str): List of tokens to be masked.
        masked_lm_prob (float): Probability of masking each token.
        max_predictions_per_seq (int): Maximum number of tokens to mask.
        whole_word_mask (bool): Whether to mask whole words instead of sub-word units.
        vocab_list (list of str): List of vocabulary words.

    Returns:
        tuple: Updated tokens, masked positions, and masked labels.

    Implements the 80/10/10 rule: 80% [MASK], 10% unchanged, 10% random word.
    """
    cand_indices = []
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]"]:
            continue
        if whole_word_mask and len(cand_indices) >= 1 and token.startswith("##"):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        if any(index in covered_indexes for index in index_set):
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = "[MASK]" if random() < 0.8 else (tokens[index] if random() < 0.5 else choice(vocab_list))
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    return tokens, [p.index for p in masked_lms], [p.label for p in masked_lms]

def create_instances_from_document(doc_database, doc_idx, max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """
    Creates training instances from a document.

    Args:
        doc_database (DocumentDatabase): Database containing tokenized documents.
        doc_idx (int): Index of the document.
        max_seq_length (int): Maximum sequence length.
        short_seq_prob (float): Probability of creating a short sequence.
        masked_lm_prob (float): Probability of masking tokens.
        max_predictions_per_seq (int): Maximum number of masked tokens.
        whole_word_mask (bool): Whether to use whole word masking.
        vocab_list (list of str): List of vocabulary words.

    Returns:
        list: Training instances.
    """
    document = doc_database[doc_idx]
    max_num_tokens = max_seq_length - 3
    target_seq_length = max_num_tokens if random() >= short_seq_prob else randint(2, max_num_tokens)

    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = randrange(1, len(current_chunk)) if len(current_chunk) >= 2 else 1
                tokens_a = [token for segment in current_chunk[:a_end] for token in segment]
                tokens_b = []

                if len(current_chunk) == 1 or random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)
                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels
                }
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances

def create_training_file(docs, vocab_list, args, epoch_num):
    """
    Creates a training file for a single epoch.

    Args:
        docs (DocumentDatabase): Database containing documents.
        vocab_list (list of str): List of vocabulary words.
        args (Namespace): Command-line arguments.
        epoch_num (int): Epoch number.
    """
    epoch_filename = args.output_dir / f"epoch_{epoch_num}.json"
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        for doc_idx in trange(len(docs), desc="Document"):
            doc_instances = create_instances_from_document(
                docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
            for instance in doc_instances:
                epoch_file.write(json.dumps(instance) + '\n')
                num_instances += 1

    metrics_file = args.output_dir / f"epoch_{epoch_num}_metrics.json"
    with metrics_file.open('w') as metrics_file:
        metrics = {"num_training_examples": num_instances, "max_seq_len": args.max_seq_len}
        metrics_file.write(json.dumps(metrics))

def main():
    """
    Main function that parses command-line arguments and generates training data for BERT pretraining.
    """
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model (e.g., bert-base-uncased)")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true", help="Use whole word masking.")
    parser.add_argument("--reduce_memory", action="store_true", help="Reduce memory usage by storing data on disk.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--epochs_to_generate", type=int, default=3, help="Number of epochs of data to generate.")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1, help="Probability of short sequences.")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="Probability of masking tokens.")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20, help="Max number of masked tokens.")

    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers with reduced memory.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())

    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                else:
                    doc.append(tokenizer.tokenize(line))
            if doc:
                docs.add_document(doc)

        if len(docs) <= 1:
            exit("ERROR: No document breaks found in the input file. Add blank lines to indicate breaks between documents.")

        args.output_dir.mkdir(exist_ok=True)

        if args.num_workers > 1:
            with Pool(min(args.num_workers, args.epochs_to_generate)) as pool:
                pool.starmap(create_training_file, [(docs, vocab_list, args, epoch) for epoch in range(args.epochs_to_generate)])
        else:
            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                create_training_file(docs, vocab_list, args, epoch)

if __name__ == '__main__':
    main()
