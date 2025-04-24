import pandas as pd
import itertools
from collections import Counter
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
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
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch
import math
import csv
from tqdm import tqdm
import sys
from itertools import islice
import os
from BiMambaForMaskedLM import BiMambaForMaskedLM

# Load and clean eccDNA sequences
csv.field_size_limit(sys.maxsize)
csv_file = "datasets/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens.csv"
sequences = []

with open(csv_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader, desc="Reading sequences"):
        seq = row.get("Sequence")
        if seq and len(seq) < 100000:
            sequences.append(seq.upper())
print(f"Loaded {len(sequences)} valid eccDNA sequences")

# Build k-mer vocab
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import reduce
from tqdm import tqdm

k = 6
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[EOS]", "[MASK]"]

def count_kmers_in_seq(seq, k=6):
    local_counter = Counter()
    append = local_counter.update
    kmers = []
    seq_len = len(seq)
    for i in range(seq_len - k + 1):
        kmer = seq[i:i+k]
        if 'N' not in kmer:
            kmers.append(kmer)
    append(kmers)
    return local_counter

def count_kmers_star(args):
    return count_kmers_in_seq(*args)

def merge_counters_with_progress(counters):
    total = Counter()
    for c in tqdm(counters, desc="Merging Counters"):
        total.update(c)
    return total
    
def parallel_kmer_count(sequences, k=6, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers) as pool:
        args_iter = ((seq, k) for seq in sequences)
        results = list(tqdm(pool.imap_unordered(count_kmers_star, args_iter, chunksize=100), total=len(sequences), desc="Processing Sequences"))

    total_counts = merge_counters_with_progress(results)
    return total_counts

def get_kmers(seq, k=6):
    return [seq[i:i+k] for i in range(len(seq) - k + 1) if seq[i:i+k].find("N") == -1]

def save_vocab(vocab: dict, filepath: str = "saved_model/vocab.txt"):
    """Save vocabulary dictionary to a text file in 'token\tindex' format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for token, index in vocab.items():
            f.write(f"{token}\t{index}\n")
    print(f"Vocabulary saved to {filepath}")
    
kmer_counts = parallel_kmer_count(sequences, k=k)

vocab = {tok: i for i, tok in enumerate(special_tokens)}
vocab.update({kmer: i + len(special_tokens) for i, kmer in enumerate(sorted(kmer_counts))})

save_vocab(vocab, "saved_model/vocab.txt")

# Build tokenizer
tokenizer_model = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer_model.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_model,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[CLS]",
    eos_token="[EOS]",
    mask_token="[MASK]",
)

# Encode sequences for causal LM
def encode_batch(batch):
    batch_input_ids = []
    for seq in batch["text"]:

        if isinstance(seq, list):
            seq = "".join(seq)
        kmers = get_kmers(seq, k)
        ids = [vocab.get(kmer, vocab["[UNK]"]) for kmer in kmers]
        ids = [vocab["[CLS]"]] + ids
        batch_input_ids.append(ids)
    
    return {"input_ids": batch_input_ids}

raw_dataset = Dataset.from_dict({"text": sequences})
tokenized_dataset = raw_dataset.map(encode_batch, batched=True, remove_columns=["text"])
tokenized_dataset.save_to_disk("tokenized_dataset/")

# Data collator for autoregressive training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
    
# Load bidirectional Mamba
def load_mamba(vocab_size):
    config = AutoConfig.from_pretrained("state-spaces/mamba-130m")
    config.vocab_size = vocab_size
    config.pad_token_id = 0
    
    model = BiMambaForMaskedLM(config)

    return model

model = load_mamba(len(vocab))

# ========== Training ==========
training_args = TrainingArguments(
    output_dir="./weights/",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=10000,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    save_total_limit=1

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Save model
trainer.save_model("saved_model/")
trainer.save_state()              