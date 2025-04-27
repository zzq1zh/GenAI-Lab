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
tokenizer.additional_special_tokens = [f"<extra_id_{i}>" for i in range(100)]

config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
config.vocab_size = tokenizer.vocab_size
config.pad_token_id = 0
config.tie_embeddings = False

# Reconstruct training arguments
model = BiMambaForMaskedLM(config)
model.cuda()    # move to the default GPU

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
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_span_length=3
) 

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
