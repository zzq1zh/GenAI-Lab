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

# Save sequences to a text file
os.makedirs("tmp", exist_ok=True)
seq_file = "tmp/sequences.txt"
with open(seq_file, "w", encoding="utf-8") as f:
    for seq in sequences:
        f.write(seq + "\n")

# Special tokens to match original setup
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[EOS]", "[MASK]"]

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
    "bos_token": "[CLS]",
    "eos_token": "[EOS]",
    "mask_token": "[MASK]",
})

# Encoding function for noncausal language modeling
def encode_batch(batch):
    encodings = fast_tokenizer(batch["text"], add_special_tokens=False, truncation=False)
    input_ids = [[fast_tokenizer.convert_tokens_to_ids("[CLS]")] + ids for ids in encodings["input_ids"]]
    return {"input_ids": input_ids}

# Build HuggingFace Dataset and apply the tokenizer
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