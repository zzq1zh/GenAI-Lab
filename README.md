# GenAI-Lab

This repository is designated for the CSCI2952N group project.

# eccDNAMamba

A reproducible Python 3.9 environment using Conda and pip-based requirements.

## Environment Setup

### 1. Create the Conda environment with Python 3.9

```bash
conda create -n eccDNAMamba python=3.9
conda activate eccDNAMamba
```

### 2. Install PyTorch 2.1.0 with CUDA 11.8

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 
```

### 3. Install nvcc via Conda

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt --no-deps
```
```bash
pip install datasets pyfaidx==0.8.1.3 mamba_ssm==2.2.4 causal_conv1d==1.5.0.post8 --no-cache-dir --no-build-isolation
```

### 5. Training
```bash
python main.py
```

### 6. Fine-tuning
Download the **pretrained model weights** and **tokenizer files** from the following link, and place them into a folder named `weights` in your project directory:

[Pretrained Model Weights and Tokenizer](https://drive.google.com/drive/folders/1m1iUJX1v1go77Ztzre7isOh8U12sYVoP?usp=sharing)

```bash
python task1_finetune.py
```
```bash
python task2_finetune.py
```

### 7. Inference
Download the **finetuned model weights** and **tokenizer files** for **Task1** and **Task2** from the following links, and place them into in your project directory:

[Finetuned Model Weights and Tokenizer for Task1](https://drive.google.com/drive/folders/10ELKlSUJVmR30HYCi-ICjY4ncEgDTARB?usp=sharing)

[Finetuned Model Weights and Tokenizer for Task2](https://drive.google.com/drive/folders/15-KxmmNHmCqoyTdxhBFoB8pg5rAQe4Xs?usp=sharing)

```bash
python task1_inference.py
```
```bash
python task2_inference.py
```
