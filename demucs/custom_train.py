import torch
from demucs.hdemucs import HDemucs  # Import HDemucs
from demucs.pretrained import get_model
from demucs.data import Dataset
from torch.utils.data import DataLoader
import os

# Path and configuration
TRAINING_FOLDERS = [
    '/DATA/Training Data/beat stems'  # Currently using only this folder
]
BATCH_SIZE = 4

def load_custom_data(training_folders, batch_size):
    """
    Function to set up the dataloader for training data across the specified folders
    with variable sample rates.
    """
    print("Loading data...")
    datasets = []
    for folder in training_folders:
        if os.path.exists(folder):
            print(f"Loading data from: {folder}")
            # Initialize the dataset for each folder
            dataset = Dataset(
                root=folder,
                streams=["drums"],  # Focus on drums
                channels=2,         # Stereo audio
                duration=10,        # Segment duration in seconds
                samplerate=None,    # Use the sample rate of the files (no resampling)
                stride=5,           # Overlap between segments
                augment=True        # Apply data augmentation
            )
            datasets.append(dataset)
        else:
            print(f"Folder not found: {folder}")
    
    if datasets:
        print("Combining datasets...")
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Adjust based on your system's capabilities
            pin_memory=True
        )
        return dataloader
    else:
        print("No valid datasets found.")
        return None

def test_data_loading():
    """
    Function to test if data from the specified folder is loaded correctly and 
    to check if the Hybrid Demucs model is loaded properly.
    """
    print("Script started...")  # This will confirm the script is running

    print("Before loading data...")
    dataloader = load_custom_data(TRAINING_FOLDERS, BATCH_SIZE)
    print("After loading data...")

    # Try loading the pretrained Hybrid Demucs model
    try:
        model = get_model('htdemucs')  # Load the 'htdemucs' variant
        print("Hybrid Demucs model loaded successfully.")
    except Exception as e:
        print(f"Error loading Hybrid Demucs model: {e}")
        return

    # Check if data is loaded properly
    if dataloader:
        print("Dataloader created successfully.")
        try:
            for i, (inputs, targets) in enumerate(dataloader):
                print(f"Batch {i+1}: Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
                if i == 0:
                    print("Data loading appears to be working correctly.")
                    break  # Test only the first batch for now
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print("No data loaded. Please check the folders and try again.")

if __name__ == '__main__':
    test_data_loading()
