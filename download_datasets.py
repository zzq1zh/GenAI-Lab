import os
import requests
from tqdm import tqdm

# EccDNA directory path
target_path = os.path.join("dataset", "preprocess", "eccDNA_Atlas", "Homo_sapiens")
os.makedirs(target_path, exist_ok=True)

# File URL and output path
csv_url = "https://drive.usercontent.google.com/download?id=1QrDjwfzg0L8M2eons97N0UPWJTnuB0jr&export=download&authuser=0&confirm=t&uuid=029b1ddb-2ffa-490d-96f8-0c815cc13a45&at=APcmpoxnXvkgKt4XilEMzmmjWEd9%3A1744124388589"
csv_file_path = os.path.join(target_path, "Homo_sapiens.csv")

# Stream download with tqdm progress bar
try:
    response = requests.get(csv_url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(csv_file_path, 'wb') as file, tqdm(
        total=total_size, unit='B', unit_scale=True, desc="Downloading",
        ncols=80, ascii=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    print(f"File downloaded successfully: {csv_file_path}")
except Exception as e:
    print(f"Failed to download the file: {e}")