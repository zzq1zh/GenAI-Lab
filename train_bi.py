import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"



import sys
import math
import csv
from tqdm import tqdm
from itertools import islice
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from safetensors.torch import load_file as load_safetensors
from BiMambaForMaskedLM import BiMambaForMaskedLM

# import wandb
# from transformers.integrations import WandbCallback
# wandb.init(project="eccDNA-bimamba", name="bimamba-4gpu")


# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="saved_model/tokenizer.json")

tokenizer.pad_token = "[PAD]"
tokenizer.unk_token = "[UNK]"
tokenizer.mask_token = "[MASK]"
tokenizer.cls_token = "[CLS]"
tokenizer.eos_token = "[EOS]"

config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
config.vocab_size = tokenizer.vocab_size
config.pad_token_id = 0

# Reconstruct training arguments
model = BiMambaForMaskedLM(config)
state_dict = load_safetensors("saved_model/model.safetensors")
model.load_state_dict(state_dict)
model.cuda()    # move to the default GPU

# print("=== CUDA & Precision Check ===")
# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"Using Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# # Check if bfloat16 is supported
# device_cap = torch.cuda.get_device_capability()
# bf16_supported = device_cap >= (8, 0)  # Ampere or newer
# print(f"BF16 Supported: {bf16_supported} (CUDA Compute Capability: {device_cap})")

# # Check model dtype (will be float32 before AMP is applied)
# print(f"Model parameter dtype (first layer): {next(model.parameters()).dtype}")
# print("================================")


training_args = TrainingArguments(
    output_dir="./saved_model/",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=10000,                  # Save every 10k steps
    save_total_limit=1,                # Keep only the last checkpoint
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",                 # Enable WandB logging
    run_name="bimamba-4gpu",           # Same as before
    gradient_checkpointing=False,      # Off to avoid extra memory usage
    ddp_find_unused_parameters=True
)




tokenized_dataset = load_from_disk("tokenized_dataset/")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Resume training
trainer.train()
trainer.save_model("saved_model/final/")
print("Model saved to saved_model/final/")
