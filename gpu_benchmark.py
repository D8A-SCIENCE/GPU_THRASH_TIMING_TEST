import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import logging
import socket
import psutil
import argparse

def get_gpu_type():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Extract common GPU model names
        if 'A40' in gpu_name:
            return 'A40'
        elif 'L4' in gpu_name:
            return 'L4'
        elif 'V100' in gpu_name:
            return 'V100'
        elif 'A100' in gpu_name:
            return 'A100'
        elif 'T4' in gpu_name:
            return 'T4'
        else:
            # Remove spaces and special characters for filename compatibility
            return ''.join(c for c in gpu_name if c.isalnum())
    return 'noGPU'

def setup_logging(cpu_count, memory_gb):
    # Create the logs directory if it doesn't exist
    log_dir = '/sciclone/geograd/GPU_THRASH_LOGS'
    os.makedirs(log_dir, exist_ok=True)
    
    # Get GPU type for filename
    gpu_type = get_gpu_type()
    
    # Format the log filename
    log_file = f'gpu_benchmark_logs_{gpu_type}_{cpu_count}cpu_{memory_gb}gb.LOG'
    log_path = os.path.join(log_dir, log_file)
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def log_environment():
    """Log system environment details"""
    logging.info("=== Environment Information ===")
    node_name = os.environ.get('NODE_NAME', 'unknown')
    pod_name = os.environ.get('POD_NAME', 'unknown')
    logging.info(f"Node Name: {node_name}")
    logging.info(f"Pod Name: {pod_name}")
    logging.info(f"CPU Count: {psutil.cpu_count()}")
    logging.info(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # GPU Information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)  # Convert to MB
        gpu_max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # Convert to MB
        
        logging.info(f"GPU: {gpu_name}")
        logging.info(f"GPU Memory Total: {gpu_max_memory:.0f} MB")
        logging.info(f"GPU Memory Reserved: {gpu_memory_reserved:.0f} MB")
        logging.info(f"GPU Memory Allocated: {gpu_memory_allocated:.0f} MB")
    else:
        logging.info("No GPU available")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GPU Benchmark with configurable CPU and Memory logging')
    parser.add_argument('--cpu', type=int, help='Number of CPUs allocated')
    parser.add_argument('--memory', type=int, help='Amount of memory in GB allocated')
    args = parser.parse_args()
    
    # Setup logging with the configuration parameters
    setup_logging(args.cpu if args.cpu else 'NA', args.memory if args.memory else 'NA')
    
    # Log environment details
    log_environment()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading timing
    start_time = time.time()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create data directory in our mounted path
    data_dir = '/sciclone/geograd/GPU_THRASH_TIMING_TEST/data'
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # First try without downloading
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                              download=False, transform=transform)
        logging.info("Using existing CIFAR10 dataset")
    except RuntimeError as e:
        logging.error(f"Dataset not found in {data_dir} and download failed: {str(e)}")
        logging.info("Please ensure the CIFAR10 dataset is pre-downloaded to the data directory")
        raise
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)
    data_load_time = time.time() - start_time
    logging.info(f"Data loading time: {data_load_time:.2f} seconds")

    # Model initialization timing
    start_time = time.time()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model_init_time = time.time() - start_time
    logging.info(f"Model initialization time: {model_init_time:.2f} seconds")

    # Training timing
    num_epochs = 5
    total_train_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        logging.info(f"Epoch {epoch + 1} training time: {epoch_time:.2f} seconds")

    logging.info(f"Average epoch training time: {total_train_time/num_epochs:.2f} seconds")
    logging.info(f"Total training time: {total_train_time:.2f} seconds")

if __name__ == "__main__":
    main()
