import os
from spikingjelly.datasets import cifar10_dvs

# Create data directory if it doesn't exist
os.makedirs('/mnt/Data_2/cifar10-dvs', exist_ok=True)

# Initialize the dataset which will trigger download and processing
dataset = cifar10_dvs.CIFAR10DVS(
    root='/mnt/Data_2/cifar10-dvs',
    data_type='frame',  # Convert events to frames
    frames_number=16,   # Number of time steps per sample
    split_by='number',  # Split events by number of frames
    transform=None,     # No additional transforms needed
    target_transform=None
)

print("CIFAR10-DVS dataset has been downloaded and prepared successfully!")
print(f"Dataset size: {len(dataset)} samples")
print(f"Sample shape: {dataset[0][0].shape}")  # Print shape of first sample 