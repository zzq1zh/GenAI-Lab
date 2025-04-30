import os
import pandas as pd
import itertools
from collections import Counter
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from BiMambaForMaskedLM import BiMambaForMaskedLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch
import math
import csv
from tqdm import tqdm
import sys
from itertools import islice
import random

# Load and clean eccDNA sequences

csv.field_size_limit(sys.maxsize)
csv_file_1 = "/users/zliu328/GenAI-Lab/dataset/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens_new_clean.csv"
csv_file_2 = "dataset/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens_clean.csv"
sequences = []

with open(csv_file_1, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in tqdm(reader, desc="Reading sequences"):
        seq = row.get("sequence")
        if seq and len(seq) < 10000 and "N" not in seq:
            sequences.append(seq.upper())
print(f"Loaded {len(sequences)} valid eccDNA sequences")


with open(csv_file_2, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader, desc="Reading sequences"):
        seq = row.get("Sequence")
        if seq and len(seq) < 10000 and "N" not in seq:
            sequences.append(seq.upper())
print(f"Loaded {len(sequences)} valid eccDNA sequences")

# Save sequences to a text file
os.makedirs("tmp", exist_ok=True)
seq_file = "tmp/sequences.txt"
with open(seq_file, "w", encoding="utf-8") as f:
    for seq in sequences:
        f.write(seq + "\n")

# Special tokens to match original setup
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]
                 

# Path to save the tokenizer
tokenizer_path = "saved_model/bpe_tokenizer"
vocab_size = 4096

# Initialize tokenizer and training configuration
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=special_tokens
)

# Train from sequence
tokenizer.train([seq_file], trainer)

# Save trained tokenizer
os.makedirs(tokenizer_path, exist_ok=True)
tokenizer.save(f"{tokenizer_path}/tokenizer.json")

# Wrap in HuggingFace's PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json")
fast_tokenizer.add_special_tokens({
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
})

# Build HuggingFace Dataset and apply the tokenizer
raw_dataset = Dataset.from_dict({"text": sequences})
tokenized_dataset = raw_dataset.map(
    lambda batch: fast_tokenizer(batch["text"], add_special_tokens=True, truncation=False),
    batched=True,
    remove_columns=["text"]
)
tokenized_dataset.save_to_disk("tokenized_dataset/")
