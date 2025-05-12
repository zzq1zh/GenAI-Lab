# GenAI-Lab

This repository is designated for the CSCI2952G group project.

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
pip install torch==2.2.0+cu121 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install nvcc via Conda

```bash
conda install -c "nvidia/label/cuda-12.1.105" cuda-nvcc
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt --no-deps
```
```bash
pip install datasets numpy==1.26.3 pyfaidx==0.8.1.3 mamba_ssm==2.2.4 causal_conv1d==1.5.0.post8 --no-cache-dir --no-build-isolation
```

### 5. Training
```bash
python main.py
```

### 6. Fine-tuning
Download the **pretrained model weights** and **tokenizer files** from the following link, and place them into a folder named `weights` in your project directory:

[Pretrained Model Weights and Tokenizer](https://drive.google.com/drive/folders/1JUrzrE01Ud0Im7nSv0kJxpPlH6ub8GMd?usp=sharing)

```bash
python task1_finetune.py
```
```bash
python task2_finetune.py
```

### 7. Inference
Download the **finetuned model weights** and **tokenizer files** for **Task1** and **Task2** from the following links, and place them into your project directory under the respective folders:
`saved_model_classifier_task1` and `saved_model_classifier_task2`:

[Finetuned Model Weights and Tokenizer for Task1](https://drive.google.com/drive/folders/1DqIc70KIN0j1FnekWynaX8y7m5oW8CcM?usp=sharing)

[Finetuned Model Weights and Tokenizer for Task2](https://drive.google.com/drive/folders/15-KxmmNHmCqoyTdxhBFoB8pg5rAQe4Xs?usp=sharing)

[Finetuned Model Weights and Tokenizer for Task3](https://drive.google.com/drive/folders/1dDsHGSo2AEJB_K_YgE5dRdLt4jBeWPpO?usp=sharing)

```bash
python task1_inference.py
```
```bash
python task2_inference.py
```
```bash
python task3_inference.py
```
