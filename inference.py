import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
import math

# ==== 1. Load tokenizer from saved tokenizer.json ====
tokenizer = PreTrainedTokenizerFast.from_pretrained("saved_model/")

# Load vocab from vocab.txt for reverse lookup (id -> token)
vocab = {}
with open("saved_model/vocab.txt", "r") as f:
    for line in f:
        token, idx = line.strip().split("\t")
        vocab[token] = int(idx)
id_to_token = {v: k for k, v in vocab.items()}

# ==== 2. Circular Positional Encoding ====
def circular_positional_encoding(seq_len, embed_dim, device="cpu", alpha=1):
    """Generate circular positional encoding (sine/cosine)"""
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angle = 2 * math.pi * positions / seq_len
    pe = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
    repeats = math.ceil(embed_dim / 2)
    pe = pe.repeat(1, repeats)[:, :embed_dim]
    return alpha * pe

# ==== 3. Load model with patched forward to inject circular PE ====
def load_mamba_with_pe(vocab_size, alpha=1):
    config = AutoConfig.from_pretrained("state-spaces/mamba-130m")
    config.vocab_size = vocab_size
    config.output_hidden_states = True
    model = AutoModelForCausalLM.from_pretrained("saved_model/")

    # Patch the model's forward method to add circular PE
    original_forward = model.forward
    def patched_forward(*args, **kwargs):
        if 'inputs_embeds' in kwargs:
            inputs_embeds = kwargs['inputs_embeds']
        else:
            inputs_embeds = model.get_input_embeddings()(kwargs['input_ids'])
        seq_len = inputs_embeds.size(1)
        pe = circular_positional_encoding(seq_len, inputs_embeds.size(-1), device=inputs_embeds.device, alpha=alpha)
        inputs_embeds = inputs_embeds + pe.unsqueeze(0)
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs.pop('input_ids', None)
        return original_forward(*args, **kwargs)

    model.forward = patched_forward
    return model

# ==== 4. k-mer encoding ====
def get_kmers(seq, k=6):
    """Split a DNA sequence into k-mers"""
    return [seq[i:i+k] for i in range(len(seq) - k + 1) if "N" not in seq[i:i+k]]

def encode_sequence(sequence, k=6):
    """Encode a DNA string to input IDs"""
    kmers = get_kmers(sequence.upper(), k)
    input_ids = [vocab["[BOS]"]] + [vocab.get(kmer, vocab["[UNK]"]) for kmer in kmers]
    return torch.tensor(input_ids).unsqueeze(0)  # Shape: (1, seq_len)

# ==== 5. Inference function ====
def inference(sequence, max_length=128, k=6):
    model = load_mamba_with_pe(vocab_size=len(vocab), alpha=1)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = encode_sequence(sequence, k).to(model.device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,           # Use sampling for generation
        top_p=0.95,               # Nucleus sampling
        top_k=50,                 # Top-k filtering
        pad_token_id=vocab["[PAD]"],
        eos_token_id=vocab["[EOS]"]
    )

    # Decode token IDs back to k-mer tokens
    output_ids = list(output[0].detach().cpu().flatten().tolist())
    decoded = [id_to_token.get(idx, "[UNK]") for idx in output_ids]
    return decoded

# ==== 6. Example usage ====
if __name__ == "__main__":
    test_seq = "ATCGAGTTCGATCGATGCGAT"  # Replace with your eccDNA sequence
    result = inference(test_seq, max_length=100)
    print("Generated k-mers:\n", result)