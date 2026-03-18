"""
VAM Lost in Migration Dataset Preprocessing

Task: Flanker task variant - identify the direction of the center bird
Stimuli: Birds pointing in 4 directions (L/R/U/D) with flanker birds
Behavioral Data: Human responses and reaction times

This script processes the VAM Lost in Migration data and creates
training/test datasets for the ConvLSTM model.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import shutil


DIRECTION_MAP = {'L': 0, 'R': 1, 'U': 2, 'D': 3}

LAYOUT_SPACERS = {
    0: [51, 0],    # horizontal line
    1: [0, 51],    # vertical line
    2: [51, 51],   # cross
    3: [34, 34],   # V left
    4: [34, 34],   # V right
    5: [34, 34],   # V down
    6: [34, 34],   # V up
}


def get_distractor_positions(targ_pos, layout, spacer):
    """Calculate distractor bird positions based on layout."""
    if layout == 0:  # horizontal line
        return [
            (targ_pos[0] - 2 * spacer[0], targ_pos[1]),
            (targ_pos[0] - spacer[0], targ_pos[1]),
            (targ_pos[0] + spacer[0], targ_pos[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1]),
        ]
    elif layout == 1:  # vertical line
        return [
            (targ_pos[0], targ_pos[1] - 2 * spacer[1]),
            (targ_pos[0], targ_pos[1] - spacer[1]),
            (targ_pos[0], targ_pos[1] + spacer[1]),
            (targ_pos[0], targ_pos[1] + 2 * spacer[1]),
        ]
    elif layout == 2:  # cross
        return [
            (targ_pos[0] - spacer[0], targ_pos[1]),
            (targ_pos[0] + spacer[0], targ_pos[1]),
            (targ_pos[0], targ_pos[1] - spacer[1]),
            (targ_pos[0], targ_pos[1] + spacer[1]),
        ]
    elif layout == 3:  # V left
        return [
            (targ_pos[0] + spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    elif layout == 4:  # V right
        return [
            (targ_pos[0] - spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] - spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    elif layout == 5:  # V down
        return [
            (targ_pos[0] - spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
        ]
    elif layout == 6:  # V up
        return [
            (targ_pos[0] - spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    return []


def create_stimulus_image(background, target_bird, distractor_bird, 
                          target_pos, distractor_positions, canvas_size=(640, 480)):
    """
    Create a stimulus image by compositing birds onto background.
    
    Args:
        background: PIL Image of background
        target_bird: PIL Image of target bird
        distractor_bird: PIL Image of distractor bird
        target_pos: (x, y) position of target bird (centered coordinates)
        distractor_positions: list of (x, y) positions for distractors
        canvas_size: size of the canvas
    
    Returns:
        PIL Image of the composed stimulus
    """
    img = background.copy()
    
    center_x = canvas_size[0] // 2
    center_y = canvas_size[1] // 2
    
    for dx, dy in distractor_positions:
        bird_x = center_x + dx - distractor_bird.width // 2
        bird_y = center_y - dy - distractor_bird.height // 2
        img.paste(distractor_bird, (int(bird_x), int(bird_y)), distractor_bird)
    
    target_x = center_x + target_pos[0] - target_bird.width // 2
    target_y = center_y - target_pos[1] - target_bird.height // 2
    img.paste(target_bird, (int(target_x), int(target_y)), target_bird)
    
    return img


class VAMDataset(Dataset):
    """
    VAM Lost in Migration Dataset for ConvLSTM training.
    
    Uses LOG NORMALIZATION for RT values (consistent with MNIST).
    
    Each sample contains:
    - image: 3-channel stimulus image (128x128)
    - label: correct direction (0-3 for L/R/U/D)
    - response: human response direction (0-3)
    - rt_normalized: LOG-normalized reaction time
    - correct: whether response matches target
    - congruency: whether flankers match target (0=congruent, 1=incongruent)
    """
    
    def __init__(self, data_dir, users=None, max_trials_per_user=25000,
                 min_rt=250, image_size=128, transform=None,
                 rt_filter=(0.2, 5.0), split='train', train_ratio=0.8,
                 random_seed=42, precompute_images=True, cache_dir=None,
                 use_log_norm=True):
        """
        Args:
            data_dir: path to VAM_Lost-in-Migration directory
            users: list of user IDs to include (None = all users)
            max_trials_per_user: maximum trials to load per user
            min_rt: minimum RT in ms to include
            image_size: size to resize images to
            transform: additional transforms to apply
            rt_filter: (min, max) RT filter in seconds
            split: 'train' or 'test'
            train_ratio: ratio of data for training
            random_seed: random seed for reproducibility
            precompute_images: whether to precompute all images
            cache_dir: directory to cache precomputed images
            use_log_norm: use log normalization for RT (default: True)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.rt_filter = rt_filter
        self.split = split
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.precompute_images = precompute_images
        self.use_log_norm = use_log_norm
        
        self.graphics_dir = os.path.join(data_dir, 'graphics')
        self.gameplay_dir = os.path.join(data_dir, 'gameplay_data')
        
        if cache_dir is None:
            cache_dir = os.path.join(data_dir, 'processed_cache')
        self.cache_dir = cache_dir
        
        self._load_graphics()
        
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            if users is None:
                users = self.metadata['user_id'].tolist()
        else:
            if users is None:
                users = [int(f.replace('user', '').replace('df.csv', '')) 
                        for f in os.listdir(self.gameplay_dir) if f.endswith('.csv')]
        
        self.users = users
        self._load_behavioral_data(max_trials_per_user, min_rt)
        
        self._compute_rt_normalization()
        
        if self.precompute_images:
            self._precompute_all_images()
        
        self._print_stats()
    
    def _load_graphics(self):
        """Load bird images and background."""
        self.background = Image.open(os.path.join(self.graphics_dir, 'bkgrnd.png')).convert('RGB')
        self.birds = {}
        for direction, idx in DIRECTION_MAP.items():
            self.birds[idx] = Image.open(os.path.join(self.graphics_dir, f'bird{idx}.png')).convert('RGBA')
    
    def _load_behavioral_data(self, max_trials, min_rt):
        """Load and preprocess behavioral data from all users."""
        all_data = []
        
        for user_id in tqdm(self.users, desc="Loading user data"):
            csv_path = os.path.join(self.gameplay_dir, f'user{user_id}df.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping")
                continue
            
            df = pd.read_csv(csv_path, nrows=max_trials)
            
            df = df[df['response_time'] >= min_rt]
            
            df['user_id'] = user_id
            all_data.append(df)
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        self.raw_data = pd.concat(all_data, ignore_index=True)
        
        self._filter_rt_outliers()
        
        self._add_derived_columns()
        
        self._split_data()
    
    def _filter_rt_outliers(self, mult=10):
        """Filter RT outliers using median absolute deviation."""
        rt = self.raw_data['response_time'].values
        median = np.median(rt)
        mad = np.median(np.abs(rt - median))
        
        mask = np.abs(rt - median) < mult * mad
        n_filtered = len(self.raw_data) - mask.sum()
        
        if n_filtered > 0:
            print(f"Filtered {n_filtered} RT outliers")
            self.raw_data = self.raw_data[mask].reset_index(drop=True)
    
    def _add_derived_columns(self):
        """Add derived columns to the dataframe."""
        self.raw_data['target_dir'] = self.raw_data['target_direction'].map(DIRECTION_MAP)
        self.raw_data['flanker_dir'] = self.raw_data['flanker_direction'].map(DIRECTION_MAP)
        self.raw_data['response_dir'] = self.raw_data['response_direction'].map(DIRECTION_MAP)
        
        self.raw_data['correct'] = (self.raw_data['target_dir'] == self.raw_data['response_dir']).astype(int)
        
        self.raw_data['congruency'] = (self.raw_data['target_dir'] != self.raw_data['flanker_dir']).astype(int)
        
        self.raw_data['rt_seconds'] = self.raw_data['response_time'] / 1000.0
        
        win_size = (640, 480)
        self.raw_data['xpos_centered'] = self.raw_data['xpos'] - win_size[0] / 2
        self.raw_data['ypos_centered'] = -self.raw_data['ypos'] + win_size[1] / 2
    
    def _split_data(self):
        """Split data into train/test sets by trials (not by users)."""
        np.random.seed(self.random_seed)
        
        n_total = len(self.raw_data)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * self.train_ratio)
        
        if self.split == 'train':
            self.data = self.raw_data.iloc[indices[:n_train]].reset_index(drop=True)
        else:
            self.data = self.raw_data.iloc[indices[n_train:]].reset_index(drop=True)
    
    def _compute_rt_normalization(self):
        """Compute RT normalization parameters using LOG normalization."""
        rt_values = self.data['rt_seconds'].values
        
        self.rt_min = np.min(rt_values)
        self.rt_max = np.max(rt_values)
        self.rt_range = self.rt_max - self.rt_min
        
        if self.use_log_norm:
            log_rt_values = np.log(rt_values)
            self.log_rt_min = np.min(log_rt_values)
            self.log_rt_max = np.max(log_rt_values)
            self.log_rt_range = self.log_rt_max - self.log_rt_min
            
            self.data['rt_normalized'] = (log_rt_values - self.log_rt_min) / self.log_rt_range
        else:
            self.data['rt_normalized'] = (rt_values - self.rt_min) / self.rt_range
    
    def _precompute_all_images(self):
        """Precompute and cache all stimulus images."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.image_cache = {}
        cache_file = os.path.join(self.cache_dir, f'{self.split}_images.npy')
        index_file = os.path.join(self.cache_dir, f'{self.split}_index.npy')
        
        if os.path.exists(cache_file) and os.path.exists(index_file):
            print(f"Loading cached images from {cache_file}")
            self.image_cache = np.load(cache_file, allow_pickle=True).item()
            return
        
        print(f"Precomputing {len(self.data)} images...")
        
        for idx in tqdm(range(len(self.data)), desc="Creating stimuli"):
            row = self.data.iloc[idx]
            
            img = self._create_stimulus(row)
            
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            self.image_cache[idx] = img_array
        
        np.save(cache_file, self.image_cache)
        np.save(index_file, np.array(list(self.image_cache.keys())))
        
        print(f"Cached images saved to {cache_file}")
    
    def _create_stimulus(self, row):
        """Create a stimulus image for a single trial."""
        target_dir = row['target_dir']
        flanker_dir = row['flanker_dir']
        layout = row['stimulus_layout']
        xpos = row['xpos_centered']
        ypos = row['ypos_centered']
        
        target_bird = self.birds[target_dir]
        distractor_bird = self.birds[flanker_dir]
        
        spacer = LAYOUT_SPACERS[layout]
        distractor_positions = get_distractor_positions((xpos, ypos), layout, spacer)
        
        img = create_stimulus_image(
            self.background, target_bird, distractor_bird,
            (xpos, ypos), distractor_positions
        )
        
        return img
    
    def _print_stats(self):
        """Print dataset statistics."""
        print(f"\nVAM Dataset ({self.split}):")
        print(f"  Total trials: {len(self.data)}")
        print(f"  Users: {self.data['user_id'].nunique()}")
        print(f"  Correct trials: {self.data['correct'].sum()} ({self.data['correct'].mean()*100:.1f}%)")
        print(f"  Congruent trials: {(self.data['congruency']==0).sum()} ({(self.data['congruency']==0).mean()*100:.1f}%)")
        print(f"  RT range: {self.rt_min:.3f} - {self.rt_max:.3f} seconds")
        print(f"  Mean RT: {self.data['rt_seconds'].mean():.3f} seconds")
        
        if self.use_log_norm:
            print(f"  Log RT range: {self.log_rt_min:.3f} - {self.log_rt_max:.3f}")
        
        print(f"\n  RT by Correctness:")
        correct_rt = self.data[self.data['correct']==1]['rt_seconds']
        incorrect_rt = self.data[self.data['correct']==0]['rt_seconds']
        print(f"    Correct: {correct_rt.mean():.3f} ± {correct_rt.std():.3f} s (n={len(correct_rt)})")
        print(f"    Incorrect: {incorrect_rt.mean():.3f} ± {incorrect_rt.std():.3f} s (n={len(incorrect_rt)})")
        
        print(f"\n  RT by Congruency:")
        congruent_rt = self.data[self.data['congruency']==0]['rt_seconds']
        incongruent_rt = self.data[self.data['congruency']==1]['rt_seconds']
        print(f"    Congruent: {congruent_rt.mean():.3f} ± {congruent_rt.std():.3f} s (n={len(congruent_rt)})")
        print(f"    Incongruent: {incongruent_rt.mean():.3f} ± {incongruent_rt.std():.3f} s (n={len(incongruent_rt)})")
    
    def denormalize_rt(self, normalized_rt):
        """Convert normalized RT back to seconds."""
        if isinstance(normalized_rt, torch.Tensor):
            normalized_rt = normalized_rt.cpu().numpy()
        
        if self.use_log_norm:
            log_rt = normalized_rt * self.log_rt_range + self.log_rt_min
            return np.exp(log_rt)
        else:
            return normalized_rt * self.rt_range + self.rt_min
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.precompute_images:
            image = self.image_cache[idx]
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            img = self._create_stimulus(row)
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(row['target_dir'], dtype=torch.long),
            'response': torch.tensor(row['response_dir'], dtype=torch.long),
            'rt_normalized': torch.tensor(row['rt_normalized'], dtype=torch.float32),
            'rt_original': torch.tensor(row['rt_seconds'], dtype=torch.float32),
            'correct': torch.tensor(row['correct'], dtype=torch.bool),
            'congruency': torch.tensor(row['congruency'], dtype=torch.long),
            'user_id': torch.tensor(row['user_id'], dtype=torch.long),
        }


def create_vam_datasets(data_dir, **kwargs):
    """Create train and test datasets."""
    train_dataset = VAMDataset(data_dir, split='train', **kwargs)
    test_dataset = VAMDataset(data_dir, split='test', **kwargs)
    
    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='Preprocess VAM Lost in Migration data')
    parser.add_argument('--data_dir', type=str, 
                        default='VAM_Lost-in-Migration',
                        help='Path to VAM data directory')
    parser.add_argument('--output_dir', type=str, 
                        default='data/processed/vam',
                        help='Output directory for processed data')
    parser.add_argument('--max_trials', type=int, default=25000,
                        help='Maximum trials per user')
    parser.add_argument('--min_rt', type=int, default=250,
                        help='Minimum RT in ms')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test_dataset', action='store_true',
                        help='Test the dataset class')
    parser.add_argument('--users', type=int, nargs='*', default=None,
                        help='List of user IDs to process')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VAM Lost in Migration Data Preprocessing")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_dataset = VAMDataset(
        args.data_dir,
        users=args.users,
        max_trials_per_user=args.max_trials,
        min_rt=args.min_rt,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        split='train'
    )
    
    test_dataset = VAMDataset(
        args.data_dir,
        users=args.users,
        max_trials_per_user=args.max_trials,
        min_rt=args.min_rt,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        split='test'
    )
    
    if args.test_dataset:
        print("\n" + "="*60)
        print("Testing Dataset")
        print("="*60)
        
        sample = train_dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Response: {sample['response']}")
        print(f"RT (normalized): {sample['rt_normalized']:.4f}")
        print(f"RT (original): {sample['rt_original']:.4f} seconds")
        print(f"Correct: {sample['correct']}")
        print(f"Congruency: {sample['congruency']}")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        batch = next(iter(train_loader))
        print(f"\nBatch image shape: {batch['image'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
