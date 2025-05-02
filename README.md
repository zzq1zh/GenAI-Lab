# GenAI-Lab

This repository is designated for the CSCI2952N group project.

# eccDNAMamba

A reproducible Python 3.9 environment using Conda and pip-based requirements.

## Environment Setup

### 1. Create the Conda environment with Python 3.9

```bash
conda create -n cycloMamba python=3.9
conda activate cycloMamba
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
pip install -r requirements.txt
```
