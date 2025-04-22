import os
import urllib.request

# EccDNA directory path
target_path = os.path.join("datasets", "preprocess", "eccDNA_Atlas", "Homo_sapiens")
os.makedirs(target_path, exist_ok=True)

# EccDNA CSV file URL
csv_url = "https://drive.google.com/file/d/1QrDjwfzg0L8M2eons97N0UPWJTnuB0jr/view?usp=sharing"
csv_file_path = os.path.join(target_path, "Homo_sapiens.csv")

# Download the EccDNA file 
try:
    urllib.request.urlretrieve(csv_url, csv_file_path)
    print(f"File downloaded successfully: {csv_file_path}")
except Exception as e:
    print(f"Failed to download the file: {e}")