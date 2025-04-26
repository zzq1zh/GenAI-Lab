import pandas as pd
import itertools
from collections import Counter
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
import math
import csv
from tqdm import tqdm
import sys
from itertools import islice
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load and clean eccDNA sequences
csv.field_size_limit(sys.maxsize)
csv_file = "/users/zliu328/GenAI-Lab/dataset/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens_clean.csv"
sequences = []

def find_and_print_nulls(path, max_reports=10):
    with open(path, 'rb') as f:
        for i, chunk in enumerate(f):
            if b'\x00' in chunk:
                print(f"NUL byte found on line {i+1}: {chunk!r}")
                max_reports -= 1
                if max_reports <= 0:
                    break

# Path to your CSV
find_and_print_nulls(csv_file)


with open(csv_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    max_records = 10000
    for row in tqdm(islice(reader, max_records), desc="Reading sequences"):
        seq = row.get("Sequence")
        if seq and len(seq) < 262144:
            sequences.append(seq.upper())
print(f"Loaded {len(sequences)} valid eccDNA sequences")

# Build k-mer vocab
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import reduce
from tqdm import tqdm

k = 6
special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

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
    bos_token="[BOS]",
    eos_token="[EOS]",
)

# Encode sequences for causal LM
def encode_batch(batch):
    batch_input_ids = []
    for seq in batch["text"]:

        if isinstance(seq, list):
            seq = "".join(seq)
        kmers = get_kmers(seq, k)
        ids = [vocab.get(kmer, vocab["[UNK]"]) for kmer in kmers]
        ids = [vocab["[BOS]"]] + ids
        batch_input_ids.append(ids)
    
    return {"input_ids": batch_input_ids}

raw_dataset = Dataset.from_dict({"text": sequences})
tokenized_dataset = raw_dataset.map(encode_batch, batched=True, remove_columns=["text"])
tokenized_dataset.save_to_disk("tokenized_dataset/")

# Data collator for autoregressive training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Circular Positional Encoding
def circular_positional_encoding(seq_len, embed_dim, device="cpu", alpha=1):
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angle = 2 * math.pi * positions / seq_len
    pe = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
    repeats = math.ceil(embed_dim / 2)
    pe = pe.repeat(1, repeats)[:, :embed_dim]
    return alpha * pe

# Load single-directional Mamba with PE
def load_mamba_with_pe(vocab_size, alpha=1):
    config = AutoConfig.from_pretrained("state-spaces/mamba-130m")
    config.vocab_size = vocab_size
    config.output_hidden_states = True
    model = AutoModelForCausalLM.from_config(config)

    # Patch forward to add circular PE
    original_forward = model.forward
    def patched_forward(*args, **kwargs):
        if 'inputs_embeds' in kwargs:
            inputs_embeds = kwargs['inputs_embeds']
        else:
            inputs_embeds = model.get_input_embeddings()(kwargs['input_ids'])
        seq_len = inputs_embeds.size(1)
        pe = circular_positional_encoding(seq_len, inputs_embeds.size(-1), device=inputs_embeds.device, alpha=alpha)
        inputs_embeds = inputs_embeds + pe.unsqueeze(0)
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs.pop('input_ids', None)

        return original_forward(*args, **kwargs)
    model.forward = patched_forward
    return model

model = load_mamba_with_pe(len(vocab), alpha=1)

# ========== Training ==========
training_args = TrainingArguments(
    output_dir="./weights/",
    per_device_train_batch_size=2,
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
