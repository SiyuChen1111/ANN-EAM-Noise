"""
MNIST Behavioral Dataset for RTNet Task

Task: RTNet (Reaction Time Network) - Human perceptual decision-making task
Raw Data Source: RTNet behavioral dataset (human responses and reaction times)
Stimuli Source: MNIST digits (0-9, mapped to 8 classes for RTNet)

This dataset combines:
1. MNIST images as visual stimuli
2. RTNet behavioral data containing human responses and reaction times
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTBehavioralDataset(Dataset):
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
        
        print(f"MNIST Behavioral Dataset loaded:")
        print(f"  Total trials: {len(self.filtered_data)}")
        print(f"  Correct trials: {self.filtered_data['correct'].sum()}")
        print(f"  Error trials: {(self.filtered_data['correct'] == 0).sum()}")
        print(f"  RT range: {self.rt_min:.3f} - {self.rt_max:.3f} seconds")
    
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
        self.rt_range = self.rt_max - self.rt_min
        
        self.filtered_data['rt_normalized'] = (rt_values - self.rt_min) / self.rt_range
        
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
        return normalized_rt * self.rt_range + self.rt_min
    
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
            'mnist_index': mnist_idx
        }


def preprocess_behavioral_data(csv_path, output_path=None):
    df = pd.read_csv(csv_path)
    
    print("Original data shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    print("\n" + "="*60)
    print("Data Statistics:")
    print("="*60)
    print(f"Total trials: {len(df)}")
    print(f"Unique subjects: {df['subject'].nunique()}")
    print(f"RT range: {df['resp_rt'].min():.3f} - {df['resp_rt'].max():.3f} seconds")
    print(f"Mean RT: {df['resp_rt'].mean():.3f} seconds")
    print(f"Accuracy: {df['correct'].mean()*100:.2f}%")
    
    print("\n" + "="*60)
    print("RT Distribution by Correctness:")
    print("="*60)
    correct_rt = df[df['correct'] == 1]['resp_rt']
    incorrect_rt = df[df['correct'] == 0]['resp_rt']
    print(f"Correct trials RT: {correct_rt.mean():.3f} ± {correct_rt.std():.3f} seconds (n={len(correct_rt)})")
    print(f"Incorrect trials RT: {incorrect_rt.mean():.3f} ± {incorrect_rt.std():.3f} seconds (n={len(incorrect_rt)})")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess MNIST behavioral data')
    parser.add_argument('--csv_path', type=str, 
                        default='RTNet_Dataset/behavioral data.csv',
                        help='Path to behavioral data CSV')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save processed data')
    parser.add_argument('--test_dataset', action='store_true',
                        help='Test the dataset class')
    
    args = parser.parse_args()
    
    df = preprocess_behavioral_data(args.csv_path, args.output_path)
    
    if args.test_dataset:
        print("\n" + "="*60)
        print("Testing MNISTBehavioralDataset")
        print("="*60)
        
        dataset = MNISTBehavioralDataset(args.csv_path)
        
        print(f"\nDataset length: {len(dataset)}")
        
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"RT (normalized): {sample['rt_normalized']:.4f}")
        print(f"RT (original): {sample['rt_original']:.4f} seconds")
        print(f"Correct: {sample['correct']}")
