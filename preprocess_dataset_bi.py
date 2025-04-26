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
csv_file_1 = "dataset/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens.csv"
csv_file_2 = "dataset/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens_clean.csv"
sequences = []

with open(csv_file_1, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(islice(reader, 10000), desc="Reading sequences"):
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
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[EOS]", "[MASK]"] + [f"<extra_id_{i}>" for i in range(100)]
                 

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
    "eos_token": "[EOS]",
    "mask_token": "[MASK]",
    "additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]
})

# Encoding function for noncausal language modeling
def encode_batch(batch):
    eos_token_id = fast_tokenizer.eos_token_id
    encodings = fast_tokenizer(batch["text"], add_special_tokens=False, truncation=False)
    input_ids_list = []
    
    for ids in encodings["input_ids"]:
        cls = [fast_tokenizer.cls_token_id]
        eos = [fast_tokenizer.eos_token_id]
        
        # Take first 256 tokens (or as many as available)
        head_len = min(256, len(ids))
        head = ids[:head_len]
        
        # Augment: original + head
        augmented_ids = cls + ids + head + eos
        input_ids_list.append(augmented_ids)

    return {"input_ids": input_ids_list}

# Build HuggingFace Dataset and apply the tokenizer
raw_dataset = Dataset.from_dict({"text": sequences})
tokenized_dataset = raw_dataset.map(encode_batch, batched=True, remove_columns=["text"])
tokenized_dataset.save_to_disk("tokenized_dataset/")

# Data collator for span masking training
class SpanMaskingCollator:
    def __init__(self, tokenizer, noise_density=0.15, mean_span_length=3):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length

    def __call__(self, examples):
        batch = [self.mask_span(example["input_ids"]) for example in examples]
        input_ids_list, labels_list = zip(*batch)

        # Find the maximum length in this batch
        max_input_length = max(len(ids) for ids in input_ids_list)
        max_label_length = max(len(ids) for ids in labels_list)

        # Pad input_ids and labels to the max length of the batch
        input_ids = torch.stack([
            self.pad_sequence(ids, max_input_length, pad_token_id=self.tokenizer.pad_token_id)
            for ids in input_ids_list
        ])
        labels = torch.stack([
            self.pad_sequence(ids, max_label_length, pad_token_id=-100)
            for ids in labels_list
        ])

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def pad_sequence(self, ids, max_length, pad_token_id):
        padded = ids + [pad_token_id] * (max_length - len(ids))
        return torch.tensor(padded, dtype=torch.long)

    def mask_span(self, input_ids):
        length = len(input_ids)
        num_masked = max(1, int(round(length * self.noise_density)))

        # Randomly sample masked spans
        mask_spans = []
        i = 0
        while i < num_masked:
            span_length = min(random.poisson(lam=self.mean_span_length), num_masked - i)
            start = random.randint(0, max(0, length - span_length))
            mask_spans.append((start, start + span_length))
            i += span_length

        # Merge overlapping spans
        mask_spans = sorted(mask_spans)
        merged_spans = []
        for start, end in mask_spans:
            if not merged_spans or start > merged_spans[-1][1]:
                merged_spans.append([start, end])
            else:
                merged_spans[-1][1] = max(merged_spans[-1][1], end)

        # Build new input_ids and labels
        new_input_ids = []
        labels = []
        extra_id = 0
        prev_end = 0
        for start, end in merged_spans:
            # Copy unmasked part
            new_input_ids.extend(input_ids[prev_end:start])
            # Insert <extra_id_X>
            new_input_ids.append(self.tokenizer.convert_tokens_to_ids(f"<extra_id_{extra_id}>"))

            # Label: <extra_id_X> + masked content
            labels.append(self.tokenizer.convert_tokens_to_ids(f"<extra_id_{extra_id}>"))
            labels.extend(input_ids[start:end])

            extra_id += 1
            prev_end = end

        # Copy the remaining part
        new_input_ids.extend(input_ids[prev_end:])
        labels.append(self.tokenizer.eos_token_id)

        return new_input_ids, labels
        
data_collator = SpanMaskingCollator(
    tokenizer=fast_tokenizer,
    noise_density=0.15,
    mean_span_length=3
)
    
# Load bidirectional Mamba
def load_mamba(vocab_size):
    config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
    config.vocab_size = vocab_size
    config.pad_token_id = 0
    config.tie_embeddings = False

    model = BiMambaForMaskedLM(config)

    return model

model = load_mamba(vocab_size)

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
