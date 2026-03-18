"""
MNIST Behavioral Dataset for RTNet Task - Log Normalization Version

Uses log-scale normalization for RT values, which better matches 
the right-skewed distribution of human reaction times.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTBehavioralDatasetLog(Dataset):
    def __init__(self, behavioral_csv_path, mnist_root='./mnist-data', 
                 train=True, transform=None, rt_filter=(0.2, 5.0),
                 image_size=227):
        self.behavioral_data = pd.read_csv(behavioral_csv_path)
        self.train = train
        self.rt_filter = rt_filter
        self.image_size = image_size
        
        if transform is None:
            if image_size == 28:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
        else:
            self.transform = transform
        
        self.mnist_dataset = datasets.MNIST(
            root=mnist_root, 
            train=train, 
            download=True,
            transform=self.transform
        )
        
        self._preprocess_behavioral_data()
        self._build_mnist_index()
        
        print(f"MNIST Behavioral Dataset (Log Normalization) loaded:")
        print(f"  Total trials: {len(self.filtered_data)}")
        print(f"  RT range: {self.rt_min:.3f} - {self.rt_max:.3f} seconds")
        print(f"  Log RT range: {self.log_rt_min:.3f} - {self.log_rt_max:.3f}")
    
    def _preprocess_behavioral_data(self):
        self.filtered_data = self.behavioral_data.copy()
        
        self.filtered_data = self.filtered_data[
            (self.filtered_data['resp_rt'] >= self.rt_filter[0]) & 
            (self.filtered_data['resp_rt'] <= self.rt_filter[1])
        ]
        
        self.filtered_data = self.filtered_data.dropna(subset=['resp_rt'])
        self.filtered_data = self.filtered_data.reset_index(drop=True)
        
        rt_values = self.filtered_data['resp_rt'].values
        self.rt_min = np.min(rt_values)
        self.rt_max = np.max(rt_values)
        
        log_rt_values = np.log(rt_values)
        self.log_rt_min = np.min(log_rt_values)
        self.log_rt_max = np.max(log_rt_values)
        self.log_rt_range = self.log_rt_max - self.log_rt_min
        
        self.filtered_data['rt_normalized'] = (log_rt_values - self.log_rt_min) / self.log_rt_range
        
        print(f"Filtered {len(self.behavioral_data) - len(self.filtered_data)} trials")
    
    def _build_mnist_index(self):
        self.mnist_images = {}
        self.mnist_labels = {}
        
        for idx in range(len(self.mnist_dataset)):
            img, label = self.mnist_dataset[idx]
            self.mnist_images[idx] = img
            self.mnist_labels[idx] = label
    
    def denormalize_rt(self, normalized_rt):
        if isinstance(normalized_rt, torch.Tensor):
            normalized_rt = normalized_rt.cpu().numpy()
        log_rt = normalized_rt * self.log_rt_range + self.log_rt_min
        return np.exp(log_rt)
    
    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, idx):
        row = self.filtered_data.iloc[idx]
        
        mnist_idx = row['mnist_index']
        label = int(row['stim']) - 1
        rt_normalized = row['rt_normalized']
        rt_original = row['resp_rt']
        correct = row['correct']
        response = int(row['response']) - 1 if row['response'] > 0 else 0
        difficulty = row['difficulty'] if 'difficulty' in row else 'unknown'
        
        image = self.mnist_images.get(mnist_idx)
        if image is None:
            image, _ = self.mnist_dataset[mnist_idx]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'rt_normalized': torch.tensor(rt_normalized, dtype=torch.float32),
            'rt_original': torch.tensor(rt_original, dtype=torch.float32),
            'correct': torch.tensor(correct, dtype=torch.bool),
            'response': torch.tensor(response, dtype=torch.long),
            'mnist_index': mnist_idx,
            'difficulty': difficulty
        }
