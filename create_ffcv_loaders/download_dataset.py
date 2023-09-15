import os
import tarfile
import requests

def download_cub_dataset(url, save_path):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download the dataset
    filename = url.split("/")[-1]
    download_path = os.path.join(save_path, filename)

    if not os.path.exists(download_path):
        response = requests.get(url, stream=True)
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    # Extract the dataset
    with tarfile.open(download_path, 'r') as tar:
        tar.extractall(save_path)

    # Remove the downloaded tar file
    os.remove(download_path)

if __name__ == "__main__":
    cub_dataset_url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    image_folder = "CUB_200_2011"

    download_cub_dataset(cub_dataset_url, image_folder)
    print("CUB dataset has been downloaded and extracted to the folder:", image_folder)
