from src.download_datasets import download_datasets
from src.extract_sequences import extract_sequences
from src.tokenize_sequences import tokenize_sequences
from train_model import train_model

# Tokenize datasets
tokenize_sequences()

# Train model
train_model()