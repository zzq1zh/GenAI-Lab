import os
import urllib.request

# EccDNA directory path
target_path = os.path.join("datasets", "preprocess", "eccDNA_Atlas", "Homo_sapiens")
os.makedirs(target_path, exist_ok=True)

# Save the EccDNA CSV
csv_url = "https://drive.usercontent.google.com/download?id=1QrDjwfzg0L8M2eons97N0UPWJTnuB0jr&export=download&authuser=0&confirm=t&uuid=029b1ddb-2ffa-490d-96f8-0c815cc13a45&at=APcmpoxnXvkgKt4XilEMzmmjWEd9%3A1744124388589"
csv_file_path = os.path.join(target_path, "Homo_sapiens.csv")

# Download the EccDNA file 
try:
    urllib.request.urlretrieve(csv_url, csv_file_path)
    print(f"File downloaded successfully: {csv_file_path}")
except Exception as e:
    print(f"Failed to download the file: {e}")
