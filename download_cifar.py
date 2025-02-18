import torchvision
import os

def download_cifar():
    data_dir = '/sciclone/geograd/GPU_THRASH_TIMING_TEST/data'
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading CIFAR10 dataset to {data_dir}")
    
    # Download training data
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    
    # Download test data
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    print("Download complete!")

if __name__ == "__main__":
    download_cifar()
