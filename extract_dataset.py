import pandas as pd
from Bio import SeqIO
import os
from glob import glob
import re

# Input EccDNA
excel_file = "datasets/raw/eccDNA_Atlas/Yeast/Yeast.xlsx"
fsa_folder = "datasets/raw/eccDNA_Atlas/Yeast/fsa"
output_csv = "datasets/preprocess/eccDNA_Atlas/Yeast/Yeast.csv"

# Read the eccDNA
df = pd.read_excel(excel_file)

# Read fsa files
genome = {}
fsa_files = glob(os.path.join(fsa_folder, "*.fsa"))
print(f"Found {len(fsa_files)} fsa files")

for fsa in fsa_files:
    print(f"Reading {os.path.basename(fsa)}")
    for record in SeqIO.parse(fsa, "fasta"):
        description = record.description
        match = re.search(r"\[chromosome=(.*?)\]", description)
        if match:
            chrom_key = match.group(1)  # Extract chromosome name like I, II, III...
            genome[chrom_key] = record.seq
            genome["chr" + chrom_key] = record.seq  # Also add version with "chr" prefix
        else:
            print(f"Could not identify chromosome name: {description}")

# Process the table and extract sequences
results = []

for idx, row in df.iterrows():
    ecc_id = row['eccDNA ID']
    chrom = row['Chr']
    
    # Check if Start and End are valid numbers
    try:
        start = int(row['Start'])
        end = int(row['End'])
    except (ValueError, TypeError):
        print(f"Skipping eccDNA ID {ecc_id} because Start or End is not a valid number: Start={row['Start']}, End={row['End']}")
        continue

    found = False
    # Try different chromosome name formats
    for key in [chrom, chrom.replace("chr", ""), "chr" + chrom]:
        if key in genome:
            
            sequence = genome[key][start - 1:end]
            results.append({
                'eccDNA ID': ecc_id,
                'Chr': key,
                'Start': start,
                'End': end,
                'Sequence': str(sequence)
            })
            found = True
            break

    if not found:
        print(f"Sequence not found for chromosome {chrom} (eccDNA ID: {ecc_id})")

# Save the results
output_df = pd.DataFrame(results)
output_df.to_csv(output_csv, index=False)
print(f"Extraction completed. Results saved to {output_csv}")