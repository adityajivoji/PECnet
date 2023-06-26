import os
import importlib
import zipfile
import subprocess

spec = importlib.util.find_spec("py7zr")
if spec is None:
    print("py7zr library not found. Installing it now...")
    subprocess.check_call(['pip', 'install', 'py7zr'])
import py7zr

def install_gdown():
    subprocess.check_call(['pip', 'install', 'gdown'])

def extract_zip_file(zip_file_path, destination_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

def extract_7z_file(archive_file_path, destination_folder):
    with py7zr.SevenZipFile(archive_file_path, 'r') as archive_ref:
        archive_ref.extractall(destination_folder)

def download_file_from_google_drive(file_id, destination_folder):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    destination_folder = os.path.join(script_dir, destination_folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    zip_file_path = os.path.join(destination_folder, "indoor_trajectory_forecasting_dataset.zip")
    
    if not os.path.exists(zip_file_path):
        spec = importlib.util.find_spec("gdown")
        if spec is None:
            print("gdown library not found. Installing it now...")
            install_gdown()
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_file_path, quiet=False)
    else:
        print("File Exist, Proceeding To Extraction...")
    
    dataset_sub_dir = 'supermarket'
    dataset_sub_dir_path = os.path.join(destination_folder, dataset_sub_dir)
    
    if os.path.exists(dataset_sub_dir_path):
        # Deleting all files in the 'supermarket' folder
        for root, dirs, files in os.walk(dataset_sub_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # print(f"deleting {file_path}")
                os.remove(file_path)
    else:
        os.makedirs(dataset_sub_dir_path)

    # Extract contents from the downloaded file
    print("Extractng...")
    extract_zip_file(zip_file_path, dataset_sub_dir_path)
    os.remove(zip_file_path)

    # Find .7z files and extract their contents
    for root, dirs, files in os.walk(dataset_sub_dir_path):
        for file in files:
            if file.endswith(".7z"):
                archive_file_path = os.path.join(root, file)
                print(f"Extracting {archive_file_path}")
                extract_7z_file(archive_file_path, root)
                os.remove(archive_file_path)

file_id = "10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe"
destination_folder = "./"
download_file_from_google_drive(file_id, destination_folder)

def delete_dataset(dataset_sub_dir_path):
    if os.path.exists(dataset_sub_dir_path):
        # Deleting all files in the 'supermarket' folder
        for root, dirs, files in os.walk(dataset_sub_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # print(f"deleting {file_path}")
                os.remove(file_path)
# file id 10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe

# Link to the google drive https://drive.google.com/file/d/10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe/view?usp=sharing