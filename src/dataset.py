from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pathlib import Path
import pickle
from collections import deque
import random

class SWDEDataset2(Dataset):
    def __init__(self, dataset_path, domain="university",split="train", buffer_size=10):
        """
        Custom streaming dataset with buffered shuffling.

        Args:
            file_paths (list): List of pickle file paths.
            buffer_size (int): Number of samples to hold in memory for shuffling.
        """
        self.path = Path(dataset_path) / domain
        self.file_paths = self._get_split(sorted(self.path.glob("**/*.pkl")),split)
        self.buffer_size = buffer_size
        self.current_file_idx = 0
        self.current_data = iter([])  # Empty iterator initially
        self.buffer = deque(maxlen=buffer_size)  # Shuffle buffer
        
        # Load the first file
        self._load_next_file()

    def _get_split(self,files, split,seed=42):
        train, test = train_test_split(files,test_size=0.2,random_state=seed)
        if split == "train":
            return train
        return test

    def _load_next_file(self):
        """Loads the next pickle file and fills the shuffle buffer."""
        if self.current_file_idx >= len(self.file_paths):
            return  # No more files left

        file = self.file_paths[self.current_file_idx]
        with open(file, "rb") as f:
            data = pickle.load(f)
            random.shuffle(data)  # Shuffle within the file to add more randomness
            self.current_data = iter(data)

        self.current_file_idx += 1  # Move to the next file

    def __getitem__(self, index):
        """Returns a shuffled item from the buffer and refills it when needed."""
        if len(self.buffer) == 0:
            # If buffer is empty, refill it from the current file
            try:
                for _ in range(self.buffer_size):
                    self.buffer.append(next(self.current_data))
            except StopIteration:
                self._load_next_file()  # Load next file when exhausted

        # Randomly pick an item from the buffer
        if len(self.buffer) > 0:
            item_idx = random.randint(0, len(self.buffer) - 1)
            return self.buffer[item_idx]
        else:
            raise IndexError("Dataset exhausted.")

    def __len__(self):
        """Returns an approximate length."""
        return sum(len(pickle.load(open(f, "rb"))) for f in self.file_paths)
    
class SWDEDataset(Dataset):
    def __init__(self, dataset_path, domain="university",split="train"):
        self.path = Path(dataset_path) / domain
        self.files = self._get_split(sorted(self.path.glob("**/*.pkl")),split)
        self._idx2file = []        
        for file_id, file in enumerate(self.files):            
            with open(file,'rb') as f:
                features = pickle.load(f)    
            prev_len = len(self._idx2file)        
            self._idx2file.extend(((prev_len,file_id) for _ in range(len(features))) )            
        self.current_features = None
        self.current_file_idx = None

    def _get_split(self,files, split,seed=42):
        train, test = train_test_split(files,test_size=0.2,random_state=seed)
        if split == "train":
            return train
        return test

    def __len__(self):
        return len(self._idx2file)

    def __getitem__(self, idx):        
        idx = len(self) + idx if idx < 0 else idx

        min_idx, file_idx = self._idx2file[idx]
        
        if self.current_file_idx != file_idx:            
            with open(self.files[file_idx],'rb') as f:
                features = pickle.load(f) 
            self.current_features = features  
            self.current_file_idx = file_idx

        return self.current_features[idx - min_idx]


class SWDEAttributeExtractionDataset(SWDEDataset):
    pass
