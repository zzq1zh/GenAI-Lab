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

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("saved_model/")

# Reconstruct training arguments
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Reconstruct trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Resume training
trainer.train()
