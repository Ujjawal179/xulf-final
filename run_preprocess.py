"""
Docstring for run_preprocess:

Workflow:

first step: load the dataset from the specified path (Unzipping)
second step: preprocess the dataset (Resizing, cropping)
third step: save the preprocessed dataset as img/1_ohwx_man/img0001.png and so on

"""
import zipfile
import os

def unzip_dataset(url_path, extract_path):
    """
    Unzips the dataset from the specified URL path to the extract path.
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with zipfile.ZipFile(url_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def main():
    url_path = "https://media.looktara.com/tmp/Akanksha.zip"
    extract_path = "Dataset/Imgs"

    unzip_dataset(url_path, extract_path)
    print(f"Dataset unzipped to {extract_path}")

