import os
import sys
import math
import csv
import torch
from tqdm import tqdm
from itertools import islice
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from safetensors.torch import load_file as load_safetensors

from BiMambaForMaskedLM import BiMambaForMaskedLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="saved_model/tokenizer.json")

tokenizer.pad_token = "[PAD]"
tokenizer.unk_token = "[UNK]"
tokenizer.mask_token = "[MASK]"
tokenizer.bos_token = "[BOS]"
tokenizer.eos_token = "[EOS]"

# Load model 
vocab = {}
with open("saved_model/vocab.txt", "r") as f:
    for line in f:
        token, idx = line.strip().split("\t")
        vocab[token] = int(idx)
id_to_token = {v: k for k, v in vocab.items()}

config = AutoConfig.from_pretrained("state-spaces/mamba-130m")
config.vocab_size = len(vocab)
config.pad_token_id = 0

# Reconstruct training arguments
model = BiMambaForMaskedLM(config)
state_dict = load_safetensors("saved_model/model.safetensors")
model.load_state_dict(state_dict)


training_args = TrainingArguments(
    output_dir="./saved_model/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=10000,
    save_total_limit=1,
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
    disable_tqdm=True
)


# Load dataset
tokenized_dataset = load_from_disk("tokenized_dataset/")


# Reconstruct trainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Resume training
trainer.train()