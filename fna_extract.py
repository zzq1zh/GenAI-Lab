import pandas as pd
from Bio import SeqIO

csv_file = "datasets/raw/eccDNA_Atlas/Homo_sapiens/Homo_sapiens.csv"
fna_file = "datasets/raw/eccDNA_Atlas/Homo_sapiens/fna/GCF_000001405.13_GRCh37_genomic.fna"
output_csv = "datasets/preprocess/eccDNA_Atlas/Homo_sapiens/Homo_sapiens_id_sequence.csv"

# Chromosome ID mapping
chr_mapping = {
    "NC_000001.10": "1",
    "NC_000002.11": "2",
    "NC_000003.11": "3",
    "NC_000004.11": "4",
    "NC_000005.9": "5",
    "NC_000006.11": "6",
    "NC_000007.13": "7",
    "NC_000008.10": "8",
    "NC_000009.11": "9",
    "NC_000010.10": "10",
    "NC_000011.9": "11",
    "NC_000012.11": "12",
    "NC_000013.10": "13",
    "NC_000014.8": "14",
    "NC_000015.9": "15",
    "NC_000016.9": "16",
    "NC_000017.10": "17",
    "NC_000018.9": "18",
    "NC_000019.9": "19",
    "NC_000020.10": "20",
    "NC_000021.8": "21",
    "NC_000022.10": "22",
    "NC_000023.10": "X",
    "NC_000024.9": "Y",
    "NC_012920.1": "MT"
}

# Load reference genome
print("Loading reference genome sequences...")
chrom_dict = {}
for record in SeqIO.parse(fna_file, "fasta"):
    chrom_name = chr_mapping.get(record.id)
    if chrom_name:
        chrom_dict[chrom_name] = record.seq
print(f"Load complete. Total chromosomes loaded: {len(chrom_dict)}")

# Read table
df = pd.read_csv(csv_file, usecols=["eccDNA ID", "eccDNA type", "Chr", "Start", "End"])
df = df.dropna(subset=["Start", "End"])
df = df[df["eccDNA type"] == "eccDNA"]

# Sequence extraction function
def get_sequence(chrom, start, end):
    chrom_key = str(chrom).replace("chr", "").upper()
    fasta_id = chrom_key
    if fasta_id and fasta_id in chrom_dict:
        seq = chrom_dict[fasta_id]
        if 1 <= start <= end <= len(seq):
            return str(seq[start-1:end])
    return None

# Build result
results = []
for idx, row in df.iterrows():
    try:
        seq = get_sequence(row["Chr"], int(row["Start"]), int(row["End"]))
        if not seq:
            print(f"Skipping {row['eccDNA ID']}, invalid interval or chromosome not found")
        results.append({
            "eccDNA ID": row["eccDNA ID"],
            "Sequence": seq
        })
    except Exception as e:
        print(f"Skipping {row['eccDNA ID']}, error: {e}")
        results.append({
            "eccDNA ID": row["eccDNA ID"],
            "Sequence": None
        })

# Save results
output_df = pd.DataFrame(results)
output_df.to_csv(output_csv, index=False)
print(f"Saved to: {output_csv}")