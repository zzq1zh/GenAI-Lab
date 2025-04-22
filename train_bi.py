import pandas as pd
import itertools
from collections import Counter
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PretrainedConfig,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForMaskedLM,
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

# Load and clean eccDNA sequences
csv.field_size_limit(sys.maxsize)
csv_file = "datasets/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens.csv"
sequences = []

with open(csv_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    max_records = 10000
    for row in tqdm(islice(reader, max_records), desc="Reading sequences"):
        seq = row.get("Sequence")
        if seq:
            sequences.append(seq.upper())
print(f"Loaded {len(sequences)} valid eccDNA sequences")

# Build k-mer vocab
k = 6
def get_kmers(seq, k=6):
    return [seq[i:i+k] for i in range(len(seq)-k+1) if "N" not in seq[i:i+k]]

all_kmers = list(itertools.chain.from_iterable(get_kmers(seq, k) for seq in sequences))
special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS], [MASK]"]
kmer_counts = Counter(all_kmers)
vocab = {tok: i for i, tok in enumerate(special_tokens)}
for i, kmer in enumerate(sorted(kmer_counts.keys())):
    vocab[kmer] = i + len(special_tokens)

# Build tokenizer
tokenizer_model = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer_model.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_model,
    unk_token="[UNK]",
    mask_token="[MASK]",
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

# Data collator for autoregressive training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
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
    config = AutoConfig.from_pretrained("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16")
    config.vocab_size = vocab_size
    config.pad_token_id = -100

    model = AutoModelForMaskedLM.from_config(config)

    print(config)
    # Patch forward to add circular PE
    original_forward = model.forward
    def patched_forward(*args, **kwargs):
        kwargs.pop("attention_mask", None)
        kwargs.pop("num_items_in_batch", None)

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
    output_dir="./mamba2_autoregressive_kmer",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Training autoregressive Mamba...")
trainer.train()