import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass  # Use default style
sns.set_palette("husl")


# ==================== NOISE FUNCTION ====================
def add_noise(x, mask_p=0.0, std=0.0, rescale_after_dropout=True):
    """
    Add noise to input (supports multiple shapes)

    Parameters:
        x: Input tensor
           - Image: (B, C, H, W) or (T, B, C, H, W)
           - Evidence trajectory: (B, T) or (T, B)
        mask_p: dropout probability (0-1)
        std: Gaussian noise standard deviation
        rescale_after_dropout: whether to rescale after dropout
            - True: standard dropout, maintains expectation (suitable for feature maps)
            - False: direct masking, accumulated sum becomes smaller (recommended for evidence trajectory)

    Returns:
        Noisy tensor
    """
    if mask_p == 0 and std == 0:
        return x

    x_noisy = x.clone()

    # 1. Random masking (dropout)
    if mask_p > 0:
        mask = torch.bernoulli(
            torch.ones_like(x) * (1 - mask_p)
        )
        x_noisy = x_noisy * mask

        # Optional rescaling
        if rescale_after_dropout:
            x_noisy = x_noisy / (1 - mask_p + 1e-8)

    # 2. Gaussian noise
    if std > 0:
        noise = torch.randn_like(x) * std
        x_noisy = x_noisy + noise

    return x_noisy


# ==================== SHAPE GENERATOR ====================
class ShapeGenerator:
    def __init__(self, square_side, matrix_size=64):
        self.square_side = square_side
        self.matrix_size = matrix_size

    def generate_symbol_matrix(self, symbol, area):
        symbol_matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=int)
        shorter_area = area // 8
        center_x = self.matrix_size // 2
        center_y = self.matrix_size // 2

        if symbol == '||':
            bar_width = 1
            bar_height = shorter_area // (2 * bar_width)
            start_x = center_x - bar_height // 2
            end_x = start_x + bar_height
            start_y1 = center_y - bar_width - 2
            end_y1 = start_y1 + bar_width
            start_y2 = center_y + 2
            end_y2 = start_y2 + bar_width
            symbol_matrix[start_x:end_x, start_y1:end_y1] = 1
            symbol_matrix[start_x:end_x, start_y2:end_y2] = 1
        elif symbol == '=':
            bar_width = 1
            bar_length = shorter_area // (2 * bar_width)
            start_y = center_y - bar_length // 2
            end_y = start_y + bar_length
            start_x1 = center_x - bar_width - 2
            end_x1 = start_x1 + bar_width
            start_x2 = center_x + 2
            end_x2 = start_x2 + bar_width
            symbol_matrix[start_x1:end_x1, start_y:end_y] = 1
            symbol_matrix[start_x2:end_x2, start_y:end_y] = 1
        elif symbol == '+':
            enlarged_area = shorter_area * 4
            bar_width = 1
            bar_length = int(np.sqrt(enlarged_area - bar_width ** 2))
            start_y = center_y - bar_length // 2
            end_y = start_y + bar_length
            start_x = center_x - bar_width // 2
            end_x = start_x + bar_width
            symbol_matrix[start_x:end_x, start_y:end_y] = 1
            start_y = center_y - bar_width // 2
            end_y = start_y + bar_width
            start_x = center_x - bar_length // 2
            end_x = start_x + bar_length
            symbol_matrix[start_x:end_x, start_y:end_y] = 1

        return symbol_matrix.reshape(1, self.matrix_size, self.matrix_size)

    def generate_square(self):
        matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=int)
        start_x = (self.matrix_size - self.square_side) // 2
        start_y = (self.matrix_size - self.square_side) // 2
        end_x = start_x + self.square_side
        end_y = start_y + self.square_side
        matrix[start_x:end_x, start_y:end_y] = 1
        return matrix.reshape(1, self.matrix_size, self.matrix_size)

    def generate_triangle(self):
        triangle_area = self.square_side ** 2
        side_length = int(np.sqrt(4 * triangle_area / np.sqrt(3)))
        matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=int)
        center_x, center_y = self.matrix_size // 2, self.matrix_size // 2
        height = int(np.sqrt(3) / 2 * side_length)
        top_x = center_x - height // 2
        for i in range(height):
            row_width = int((i + 1) / height * side_length)
            start_col = center_y - row_width // 2
            end_col = start_col + row_width
            matrix[top_x + i, start_col:end_col] = 1
        return matrix.reshape(1, self.matrix_size, self.matrix_size)

    def generate_circle(self):
        circle_area = self.square_side ** 2
        radius = int(np.sqrt(circle_area / np.pi))
        matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=int)
        center_x, center_y = self.matrix_size // 2, self.matrix_size // 2
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
                    matrix[i, j] = 1
        return matrix.reshape(1, self.matrix_size, self.matrix_size)


# ==================== HUMAN ALIGNED DATASET WITH ERROR TRIALS ====================
class HumanAlignedDataset(Dataset):
    """
    Modified dataset class to handle Exp1_postpro.csv format which includes error trials.

    Key differences:
    - Uses 'shape_en' column instead of 'shape'
    - Handles 'matchness' as string ("match"/"mismatch") instead of "tensor([1])"
    - Always uses ground truth labels (not human responses)
    - Includes both correct trials (93%) and error trials (7%)
    """
    def __init__(self, human_data_path, generator):
        self.human_data = pd.read_csv(human_data_path)
        self.generator = generator

        self.shape_matrices = {
            'square': self.generator.generate_square(),
            'triangle': self.generator.generate_triangle(),
            'circle': self.generator.generate_circle()
        }

        self.symbol_matrices = {
            'square': self.generator.generate_symbol_matrix('||', generator.square_side ** 2),
            'triangle': self.generator.generate_symbol_matrix('+', generator.square_side ** 2),
            'circle': self.generator.generate_symbol_matrix('=', generator.square_side ** 2)
        }

        # Filter NaN RT values
        self.human_data = self.human_data.dropna(subset=['rt'])

        # Filter RT range 200-1200 ms
        self.human_data = self.human_data[
            (self.human_data['rt'] >= 200) & (self.human_data['rt'] <= 1200)
        ]

        print(f"Filtered dataset: {len(self.human_data)} trials remaining")

        # Reset index after filtering
        self.human_data = self.human_data.reset_index(drop=True)

        rt_values = self.human_data['rt'].values
        rt_tensor = torch.tensor(rt_values, dtype=torch.float32)

        # Save normalization parameters for inverse transformation (ignore NaN values)
        valid_rt = rt_tensor[~torch.isnan(rt_tensor)]
        self.rt_min = valid_rt.min().item()
        self.rt_max = valid_rt.max().item()
        self.rt_range = self.rt_max - self.rt_min

        # Normalized RT (0-1 range)
        self.rt_converted = (rt_tensor - self.rt_min) / self.rt_range

        # Store original RT values for reference
        self.rt_original = rt_values

        # Detect column format (handles both old and new CSV formats)
        self.shape_column = 'shape_en' if 'shape_en' in self.human_data.columns else 'shape'
        self.uses_string_matchness = 'shape_en' in self.human_data.columns  # New format uses string matchness

        print(f"Dataset format detected: shape_column='{self.shape_column}', string_matchness={self.uses_string_matchness}")

        # Count error trials for reporting
        if 'correct' in self.human_data.columns:
            n_errors = (~self.human_data['correct']).sum()
            n_total = len(self.human_data)
            print(f"Dataset loaded: {n_total} trials ({n_errors} errors, {n_errors/n_total*100:.1f}%)")
    
    def denormalize_rt(self, normalized_rt):
        """
        Convert normalized RT (0-1) back to original RT scale (milliseconds)
        
        Parameters:
        -----------
        normalized_rt : float, np.ndarray, or torch.Tensor
            Normalized RT value(s) in range [0, 1]
        
        Returns:
        --------
        Original RT value(s) in milliseconds
        """
        if isinstance(normalized_rt, torch.Tensor):
            normalized_rt = normalized_rt.cpu().numpy()
        
        return normalized_rt * self.rt_range + self.rt_min

    def __len__(self):
        return len(self.human_data)

    def __getitem__(self, idx):
        row = self.human_data.iloc[idx]
        # Use detected shape column (handles both 'shape' and 'shape_en')
        shape = row[self.shape_column]
        word = row['word']

        word_mapping = {'方形': 'square', '三角': 'triangle', '圆形': 'circle'}
        symbol = word_mapping.get(word, 'square')

        img1 = torch.tensor(self.shape_matrices[shape], dtype=torch.float32)
        img2 = torch.tensor(self.symbol_matrices[symbol], dtype=torch.float32)

        # Handle matchness based on detected format
        # IMPORTANT: Always use ground truth label, not human response
        matchness_str = row['matchness']
        if self.uses_string_matchness:
            # New format: "match"/"mismatch" strings
            if matchness_str == 'match':
                label = 1
            else:  # 'mismatch'
                label = 0
        else:
            # Old format: "tensor([1])" strings
            label = int(matchness_str.split('[')[1].split(']')[0])
        rt = self.rt_converted[idx]

        return img1, img2, torch.tensor(label, dtype=torch.long), rt


# ==================== IMAGE ENCODER ====================
class ImageEncoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, z_dim)

    def forward(self, x):
        cnn_feat = self.cnn_backbone(x)
        pooled = self.global_pool(cnn_feat).squeeze(-1).squeeze(-1)
        z = self.fc(pooled)
        return z


# ==================== CONVOLUTIONAL LSTM ====================
class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=0):
        super().__init__()
        self._conv = nn.Conv2d(input_channel + num_filter, num_filter * 4, kernel_size, stride, padding)
        self._batch_size, self._state_height, self._state_width = b_h_w
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self._input_channel = input_channel
        self._num_filter = num_filter

    def forward(self, inputs=None, states=None, seq_len=20):
        device = inputs.device
        actual_batch_size = inputs.size(1)
        if states is None:
            c = torch.zeros((actual_batch_size, self._num_filter, self._state_height, self._state_width), device=device)
            h = torch.zeros_like(c)
        else:
            h, c = states
        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)
            i, f, new_c, o = torch.chunk(conv_x, 4, dim=1)
            i = torch.sigmoid(i + self.Wci * c)
            f = torch.sigmoid(f + self.Wcf * c)
            c = f * c + i * torch.tanh(new_c)
            o = torch.sigmoid(o + self.Wco * c)
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=0), (h, c)


# ==================== DIFFERENTIABLE DECISION ====================
class DiffDecision(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory):
        mask = trajectory > 0
        decision_time = mask.float().argmax(dim=1).float()
        decision_time[mask.sum(dim=1) == 0] = float(trajectory.shape[1] - 1)
        ctx.save_for_backward(dsdt_trajectory, decision_time, trajectory)
        return decision_time

    @staticmethod
    def backward(ctx, grad_output):
        dsdt_trajectory, decision_times, trajectory = ctx.saved_tensors
        device = dsdt_trajectory.device
        mask = trajectory > 0
        idx1 = (mask.sum(dim=1) == 0)
        idx2 = dsdt_trajectory[torch.arange(dsdt_trajectory.size(0), device=device), decision_times.long()] < 0
        idx = torch.logical_and(idx1, idx2)
        grads = torch.zeros_like(dsdt_trajectory)
        batch_indices = torch.arange(decision_times.size(0), device=device)
        grads[batch_indices, decision_times.long()] = -1.0 / (dsdt_trajectory[batch_indices, decision_times.long()] + 1e-6)
        grads[batch_indices[idx], decision_times[idx].long()] = 1e-6
        grads = grads * grad_output.unsqueeze(1)
        return grads, None


# ==================== RTIFY LSTM ====================
class RTify_LSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, output_size,
                 time_steps=20, sigma=2.0,
                 noise_position='input',
                 mask_p=0.0,
                 gaussian_std=0.0,
                 evidence_noise_std=0.0,
                 evidence_mask_p=0.0,
                 evidence_dropout_rescale=False,
                 evidence_scale=1.0,
                 threshold=6.0):

        super().__init__()
        self.evidence_dropout_rescale = evidence_dropout_rescale
        self.time_steps = time_steps
        self.noise_position = noise_position
        self.mask_p = mask_p
        self.gaussian_std = gaussian_std
        self.evidence_noise_std = evidence_noise_std
        self.evidence_mask_p = evidence_mask_p
        self.evidence_dropout_rescale = evidence_dropout_rescale
        self.evidence_scale = evidence_scale

        self.convlstm = ConvLSTM(input_channel, num_filter, b_h_w, kernel_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filter, output_size)

        # Add LayerNorm for input normalization (prevents tanh saturation)
        self.input_norm = nn.LayerNorm(input_channel)

        # Evidence network: LayerNorm prevents LSTM input saturation,
        # Tanh constrains per-step evidence to [-1, 1] for stable multi-step accumulation
        self.evidence = nn.Sequential(
            nn.Linear(num_filter, num_filter),  # 64 → 64
            nn.ReLU(),
            nn.Linear(num_filter, 1),
            nn.Tanh()                            # Bounds evidence to [-1, 1]
        )
        self.register_buffer('threshold', torch.tensor(threshold))  # Fixed, not learnable
        self.sigma = sigma

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.squeeze(-1).squeeze(-1)
        x_flat_normed = self.input_norm(x_flat)
        x = x_flat_normed.unsqueeze(-1).unsqueeze(-1)

        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)

        if self.noise_position in ['input', 'both']:
            x_seq = add_noise(x_seq, self.mask_p, self.gaussian_std)

        hidden_states, _ = self.convlstm(x_seq, seq_len=self.time_steps)

        time_steps, B, num_filter, H, W = hidden_states.shape
        hidden_2d = hidden_states.view(time_steps * B, num_filter, H, W)
        pooled_2d = self.pool(hidden_2d).squeeze()
        hidden_states = pooled_2d.view(time_steps, B, num_filter)

        logit_trajectory = self.fc(hidden_states).permute(1, 0, 2)
        s_traj = self.evidence(hidden_states).squeeze(-1).permute(1, 0) * self.evidence_scale

        if self.noise_position in ['evidence', 'both']:
            s_traj = add_noise(
                s_traj,
                self.evidence_mask_p,
                self.evidence_noise_std,
                rescale_after_dropout=self.evidence_dropout_rescale
            )

        s_accumulated = torch.cumsum(s_traj, dim=1)
        dsdt_trajectory = torch.diff(s_accumulated, dim=1)
        dsdt_trajectory = torch.cat([dsdt_trajectory[:, :1], dsdt_trajectory], dim=1)

        decision_time = DiffDecision.apply(s_accumulated - self.threshold, dsdt_trajectory)

        soft_index = torch.exp(-0.5 * (decision_time.unsqueeze(1) -
                               torch.arange(self.time_steps, device=x.device)) ** 2 / self.sigma ** 2)
        soft_index = soft_index / soft_index.sum(dim=-1, keepdim=True)
        decision_logits = (logit_trajectory * soft_index.unsqueeze(-1)).sum(dim=1)

        return decision_logits, (decision_time + 1) / self.time_steps


# ==================== DUAL IMAGE CONTRASTIVE MODEL ====================
class DualImageContrastiveModel(nn.Module):
    def __init__(self, z_dim=128, share_backbone=True):
        super().__init__()
        self.encoder1 = ImageEncoder(z_dim)
        self.encoder2 = self.encoder1 if share_backbone else ImageEncoder(z_dim)

    def forward(self, img1, img2):
        z1 = self.encoder1(img1)
        z2 = self.encoder2(img2)
        distance = torch.norm(z1 - z2, p=2, dim=1)
        return z1, z2, distance


# ==================== ENCODER RTIFY MODEL ====================
class EncoderRTifyModel(nn.Module):
    """
    Complete decision time prediction model

    Parameters:
        noise_position: Noise injection position
            - 'input': Only add noise before ConvLSTM input
            - 'evidence': Only add noise to evidence trajectory (default, maintains original behavior)
            - 'both': Add noise to both
            - None: No noise
        mask_p: dropout probability (for input noise)
        gaussian_std: Gaussian noise standard deviation (for input noise)
        evidence_noise_std: Evidence noise standard deviation (for evidence noise)
    """
    def __init__(self, pretrained_encoder_path=None, z_dim=128, num_filter=64,
                 output_size=2, time_steps=20, sigma=2.0,
                 freeze_encoder=True, share_encoder_backbone=True,
                 noise_position='evidence',
                 mask_p=0.0,
                 gaussian_std=0.0,
                 evidence_noise_std=0.0,
                 evidence_mask_p=0.0,
                 evidence_dropout_rescale=False,
                 evidence_scale=1.0,
                 threshold=6.0):
        super().__init__()
        self.z_dim = z_dim
        self.freeze_encoder = freeze_encoder

        # Load or create encoder
        if pretrained_encoder_path:
            self._load_pretrained_encoder(pretrained_encoder_path, share_encoder_backbone)
        else:
            self.encoder1 = ImageEncoder(z_dim)
            self.encoder2 = self.encoder1 if share_encoder_backbone else ImageEncoder(z_dim)
            print(f"Created encoder with z_dim={z_dim}")
            
        # Freeze encoder (if needed)
        if freeze_encoder and pretrained_encoder_path:
            self._freeze_encoder()

        # RTify module
        self.rtify = RTify_LSTM(
            input_channel=2 * z_dim,
            num_filter=num_filter,
            b_h_w=(1, 1, 1),
            kernel_size=1,
            output_size=output_size,
            time_steps=time_steps,
            sigma=sigma,
            noise_position=noise_position,
            mask_p=mask_p,
            gaussian_std=gaussian_std,
            evidence_noise_std=evidence_noise_std,
            evidence_mask_p=evidence_mask_p,
            evidence_dropout_rescale=evidence_dropout_rescale,
            evidence_scale=evidence_scale,
            threshold=threshold
        )


    def _load_pretrained_encoder(self, path, share_backbone):
        temp_model = DualImageContrastiveModel(z_dim=self.z_dim, share_backbone=share_backbone)
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        temp_model.load_state_dict(state_dict)
        self.encoder1 = temp_model.encoder1
        self.encoder2 = temp_model.encoder2

    def _freeze_encoder(self):
        for param in self.encoder1.parameters():
            param.requires_grad = False
        if self.encoder2 is not self.encoder1:
            for param in self.encoder2.parameters():
                param.requires_grad = False

    def forward(self, img1, img2):
        B = img1.size(0)
        z1 = self.encoder1(img1)
        z2 = self.encoder2(img2)
        z_concat = torch.cat([z1, z2], dim=1)
        z_4d = z_concat.unsqueeze(-1).unsqueeze(-1)

        decision_logits, decision_time = self.rtify(z_4d)
        return decision_logits, decision_time, z1, z2

    def forward_with_diagnostics(self, img1, img2):
        B = img1.size(0)
        z1 = self.encoder1(img1)
        z2 = self.encoder2(img2)
        z_concat = torch.cat([z1, z2], dim=1)
        z_4d = z_concat.unsqueeze(-1).unsqueeze(-1)

        decision_logits, decision_time = self.rtify(z_4d)
        z_dist = torch.norm(z1 - z2, p=2, dim=1, keepdim=True)
        return decision_logits, decision_time, z1, z2, z_dist.squeeze(-1)


# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, test_loader=None, num_epochs=5, lr=1e-3, device='cpu',
                use_rt_loss=False, speed_penalty=0.0, save_path=None,
                accuracy_threshold=0.7, save_best=True, test_dataset=None,
                output_dir='./output', filename='model', time_steps=20,
                condition_name=None, show_incorrect=False, save_epoch=None, random_seed=None,
                save_plots=True, save_model=True):
    """
    Train the model with optional per-epoch evaluation and best model tracking

    Args:
        model: The EncoderRTifyModel
        train_loader: Training data loader
        test_loader: Test data loader (for per-epoch evaluation)
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        use_rt_loss: Whether to use RT supervision
        speed_penalty: Penalty coefficient for slow decisions (0=no penalty, higher=favor faster)
        save_path: Path to save the trained model
        accuracy_threshold: Minimum accuracy for saving best model
        save_best: Whether to track and save best model based on fast-same effect

    Returns:
        history: Dictionary containing training metrics
        best_epoch_info: Information about the best epoch (if test_loader provided)
    """
    from scipy import stats
    
    label_criterion = nn.CrossEntropyLoss()
    rt_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'rt_loss': [], 'label_loss': [], 'total_loss': [], 'acc': [], 'corr': []
    }
    
    # Track best model based on fast-same effect with new priority
    best_model_state = None
    best_epoch_info = {
        'epoch': -1,
        'accuracy': 0.0,
        'fast_same_effect_size': -999.0,  # Start with very negative value
        'p_value': 1.0,
        'match_mean': 0.0,
        'mismatch_mean': 0.0,
        'has_significant_fast_same': False,
        'meets_accuracy': False
    }
    
    # Track ALL "NEW BEST" epochs for batch visualization after training
    all_best_epochs = []  # Will store model states and metadata for each NEW BEST
    
    # Track ALL epochs that meet accuracy threshold for immediate output generation
    accuracy_threshold_epochs = []  # Will store epochs that meet accuracy threshold
    
    # Store complete results for all epochs that meet accuracy threshold
    all_threshold_results = []  # Will store complete analysis results for filtering

    model.to(device)

    # Create training log file
    log_path = os.path.join(output_dir, f'{filename}_training_log.txt')
    with open(log_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING LOG\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {filename}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"RT Supervision: {'Yes' if use_rt_loss else 'No'}\n")
        f.write(f"Speed Penalty: {speed_penalty}\n")
        f.write("="*70 + "\n\n")

    print(f"\nStarting Training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"RT Supervision: {'Yes' if use_rt_loss else 'No'}")
    print(f"Speed Penalty: {speed_penalty} {'(favoring faster decisions)' if speed_penalty > 0 else '(no speed preference)'}")
    print(f"Training log: {log_path}")
    if test_loader and save_best:
        print(f"Per-Epoch Evaluation: Enabled")
        print(f"Accuracy Threshold: {accuracy_threshold*100:.1f}%")
        print(f"Selection Priority:")
        print(f"  1. Significant fast-same effect (p < 0.05)")
        print(f"  2. Larger effect size")
        print(f"  3. Higher accuracy")
    print("="*60)

    for epoch in range(num_epochs):
        # ============ TRAINING PHASE ============
        model.train()
        epoch_metrics = {'rt_loss': [], 'label_loss': [], 'acc': [], 'corr': []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for img1, img2, labels, rt in pbar:
            # Use non_blocking for faster GPU transfer when pin_memory is enabled
            non_blocking = device.type == 'cuda'
            img1 = img1.to(device, non_blocking=non_blocking)
            img2 = img2.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            rt = rt.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            decision_logits, rt_pred, _, _ = model(img1, img2)

            # Calculate losses
            rt_loss = rt_criterion(rt_pred, rt)
            label_loss = label_criterion(decision_logits, labels)

            # Speed penalty: penalize slower decisions
            if speed_penalty > 0:
                speed_loss = speed_penalty * rt_pred.mean()
            else:
                speed_loss = 0.0

            # Decide whether to use RT loss for supervision
            if use_rt_loss:
                total_loss = rt_loss + label_loss + speed_loss
            else:
                total_loss = label_loss + speed_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record metrics
            epoch_metrics['rt_loss'].append(rt_loss.item())
            epoch_metrics['label_loss'].append(label_loss.item())
            acc = (decision_logits.argmax(-1) == labels).float().mean().item()
            epoch_metrics['acc'].append(acc)

            # Calculate RT correlation
            rt_pred_np = rt_pred.detach().cpu().numpy().flatten()
            rt_np = rt.cpu().numpy().flatten()
            corr_temp = np.corrcoef(rt_pred_np, rt_np)[0, 1] if len(rt_pred_np) > 1 else 0.0
            epoch_metrics['corr'].append(np.nan_to_num(corr_temp))

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'acc': f'{acc:.3f}',
                'corr': f'{np.nan_to_num(corr_temp):.3f}'
            })

        # Record epoch-level training metrics
        for key in epoch_metrics:
            history[key].append(np.mean(epoch_metrics[key]))

        if use_rt_loss:
            history['total_loss'].append(history['rt_loss'][-1] + history['label_loss'][-1])
        else:
            history['total_loss'].append(history['label_loss'][-1])

        # Print epoch training summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Training Summary:")
        print(f"  RT Loss: {history['rt_loss'][-1]:.4f}")
        print(f"  Label Loss: {history['label_loss'][-1]:.4f}")
        print(f"  Accuracy: {history['acc'][-1]*100:.2f}%")
        print(f"  RT Correlation: {history['corr'][-1]:.4f}")

        # Write to log file
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"  RT Loss: {history['rt_loss'][-1]:.4f}\n")
            f.write(f"  Label Loss: {history['label_loss'][-1]:.4f}\n")
            f.write(f"  Accuracy: {history['acc'][-1]*100:.2f}%\n")
            f.write(f"  RT Correlation: {history['corr'][-1]:.4f}\n")
            f.write("-"*70 + "\n")
        
        # ============ EVALUATION PHASE ============
        if test_loader is not None:
            print(f"\n  Evaluating on test set...")
            model.eval()
            
            model_match_times = []
            model_mismatch_times = []
            model_match_times_ms = []  # Denormalized RT in milliseconds
            model_mismatch_times_ms = []  # Denormalized RT in milliseconds
            correct_predictions = 0
            total_predictions = 0
            
            # Check if we can denormalize
            can_denormalize = (test_dataset is not None and 
                              hasattr(test_dataset, 'denormalize_rt'))
            
            with torch.no_grad():
                for img1, img2, labels, rt_human in test_loader:
                    non_blocking = device.type == 'cuda'
                    img1 = img1.to(device, non_blocking=non_blocking)
                    img2 = img2.to(device, non_blocking=non_blocking)
                    labels = labels.to(device, non_blocking=non_blocking)

                    decision_logits, decision_time, _, _ = model(img1, img2)
                    pred_labels = decision_logits.argmax(dim=-1)

                    for i in range(len(labels)):
                        true_label = labels[i].item()
                        pred_label = pred_labels[i].item()
                        model_dt = decision_time[i].item()

                        if pred_label == true_label:
                            correct_predictions += 1
                        total_predictions += 1

                        # Denormalize to milliseconds
                        if can_denormalize:
                            model_dt_ms = test_dataset.denormalize_rt(model_dt)
                        else:
                            model_dt_ms = None

                        if true_label == 1:
                            model_match_times.append(model_dt)
                            if can_denormalize:
                                model_match_times_ms.append(model_dt_ms)
                        else:
                            model_mismatch_times.append(model_dt)
                            if can_denormalize:
                                model_mismatch_times_ms.append(model_dt_ms)
            
            # Calculate test metrics (on normalized values for consistency)
            test_accuracy = correct_predictions / total_predictions
            match_mean = np.mean(model_match_times)
            mismatch_mean = np.mean(model_mismatch_times)
            
            # Perform t-test (on normalized values)
            ttest_result = stats.ttest_ind(model_match_times, model_mismatch_times)
            
            # Calculate fast-same effect size (positive = match faster than mismatch)
            effect_size = mismatch_mean - match_mean
            has_fast_same = effect_size > 0
            is_significant = ttest_result.pvalue < 0.05
            
            # Check if this epoch meets basic criteria
            meets_accuracy = test_accuracy >= accuracy_threshold
            has_significant_fast_same = has_fast_same and is_significant
            
            # Print results - show both normalized and denormalized values
            print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
            
            if can_denormalize:
                # Calculate millisecond-scale statistics
                match_mean_ms = np.mean(model_match_times_ms)
                mismatch_mean_ms = np.mean(model_mismatch_times_ms)
                effect_size_ms = mismatch_mean_ms - match_mean_ms
                
                # Perform t-test on millisecond values too
                ttest_result_ms = stats.ttest_ind(model_match_times_ms, model_mismatch_times_ms)
                
                print(f"  Match Mean DT: {match_mean:.4f} (normalized) = {match_mean_ms:.2f} ms")
                print(f"  Mismatch Mean DT: {mismatch_mean:.4f} (normalized) = {mismatch_mean_ms:.2f} ms")
                print(f"  Fast-Same Effect Size: {effect_size:.4f} (normalized) = {effect_size_ms:.2f} ms {'✓' if has_fast_same else '✗'}")
                print(f"  T-test (normalized): t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4f} "
                      f"{'***' if ttest_result.pvalue < 0.001 else '**' if ttest_result.pvalue < 0.01 else '*' if ttest_result.pvalue < 0.05 else 'ns'}")
                print(f"  T-test (ms): t={ttest_result_ms.statistic:.4f}, p={ttest_result_ms.pvalue:.4f}")
            else:
                print(f"  Match Mean DT: {match_mean:.4f}")
                print(f"  Mismatch Mean DT: {mismatch_mean:.4f}")
                print(f"  Fast-Same Effect Size: {effect_size:.4f} {'✓' if has_fast_same else '✗'}")
                print(f"  T-test: t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4f} "
                      f"{'***' if ttest_result.pvalue < 0.001 else '**' if ttest_result.pvalue < 0.01 else '*' if ttest_result.pvalue < 0.05 else 'ns'}")
            
            print(f"  Significant Fast-Same: {'✓ YES' if has_significant_fast_same else '✗ NO'}")
            print(f"  Meets Accuracy Threshold: {'✓ YES' if meets_accuracy else '✗ NO'}")

            # Check if we should save outputs for this epoch
            # If save_epoch is None, save for all epochs meeting accuracy threshold
            # If save_epoch is specified, only save when current epoch matches save_epoch
            should_save = meets_accuracy and (save_epoch is None or (epoch + 1) == save_epoch)

            # If meets accuracy threshold and matches save_epoch, generate immediate outputs
            if should_save:
                print(f"\n  " + "="*56)
                print(f"  ★★★ ACCURACY THRESHOLD MET - GENERATING OUTPUTS ★★★")
                print(f"  " + "="*56)
                
                # Create unique filename for this epoch
                epoch_filename = f"{filename}_ep{epoch+1:02d}_acc{test_accuracy:.3f}_effect{effect_size:.4f}"
                epoch_output_path = os.path.join(output_dir, f"{epoch_filename}.png")
                
                print(f"  Output directory: {output_dir}")
                print(f"  Output filename: {epoch_filename}")
                print(f"  Full path: {epoch_output_path}")
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                print(f"  ✓ Output directory confirmed")
                
                # Generate complete analysis and plots for this epoch
                try:
                    print(f"  → Calling analyze_and_plot_decision_time...")
                    epoch_results = analyze_and_plot_decision_time(
                        model, test_loader,
                        test_dataset=test_dataset,
                        device=device,
                        save_path=epoch_output_path,
                        time_steps=time_steps,
                        accuracy_threshold=accuracy_threshold,
                        model_save_path=None,  # Don't save model here, just save analysis
                        full_model=None,
                        return_full_data=True,  # Return full test data for later analysis
                        condition_name=condition_name,
                        show_incorrect=show_incorrect,
                        random_seed=random_seed,
                        save_plots=save_plots,
                        save_model=save_model
                    )
                    
                    # Verify files were created
                    expected_files = [
                        epoch_output_path,  # PNG
                        epoch_output_path.replace('.png', '.pdf'),  # PDF
                        epoch_output_path.replace('.png', '_decision_times.csv'),  # CSV
                        epoch_output_path.replace('.png', '_significance.csv')  # Significance CSV
                    ]
                    
                    print(f"\n  Verifying generated files:")
                    for fpath in expected_files:
                        if os.path.exists(fpath):
                            file_size = os.path.getsize(fpath) / 1024  # KB
                            print(f"    ✓ {os.path.basename(fpath)} ({file_size:.1f} KB)")
                        else:
                            print(f"    ✗ {os.path.basename(fpath)} - NOT FOUND!")
                    
                    # Store complete epoch information for later filtering
                    epoch_snapshot = {
                        'epoch': epoch + 1,
                        'model_state': model.state_dict().copy(),
                        'accuracy': test_accuracy,
                        'fast_same_effect_size': effect_size,
                        'p_value': ttest_result.pvalue,
                        'match_mean': match_mean,
                        'mismatch_mean': mismatch_mean,
                        'has_significant_fast_same': has_significant_fast_same,
                        'meets_accuracy': meets_accuracy,
                        'ttest_statistic': ttest_result.statistic,
                        'filename': epoch_filename,
                        'output_path': epoch_output_path,
                        'correlation': epoch_results.get('correlation', 0.0),
                        # Store complete test data for this epoch
                        'all_trials': epoch_results.get('all_trials', []),
                        'model_match_times': epoch_results.get('model_match_times', []),
                        'model_mismatch_times': epoch_results.get('model_mismatch_times', []),
                        'human_match_times': epoch_results.get('human_match_times', []),
                        'human_mismatch_times': epoch_results.get('human_mismatch_times', []),
                        'model_match_times_ms': epoch_results.get('model_match_times_ms'),
                        'model_mismatch_times_ms': epoch_results.get('model_mismatch_times_ms'),
                        'human_match_times_ms': epoch_results.get('human_match_times_ms'),
                        'human_mismatch_times_ms': epoch_results.get('human_mismatch_times_ms')
                    }
                    
                    accuracy_threshold_epochs.append(epoch_snapshot)
                    
                    # Also store the complete analysis results in the all_threshold_results list
                    all_threshold_results.append(epoch_snapshot.copy())
                    print(f"\n  ✓✓✓ SUCCESS! All outputs generated for Epoch {epoch+1}")
                    print(f"  " + "="*56 + "\n")
                    
                except Exception as e:
                    import traceback
                    print(f"\n  ✗✗✗ ERROR generating outputs for Epoch {epoch+1}:")
                    print(f"  Error type: {type(e).__name__}")
                    print(f"  Error message: {str(e)}")
                    print(f"\n  Full traceback:")
                    print(traceback.format_exc())
                    print(f"  " + "="*56 + "\n")
            
            # Update best model with new priority:
            # Priority 1: Significant fast-same effect (p < 0.05 and effect > 0)
            # Priority 2: Larger effect size
            # Priority 3: Higher accuracy
            if save_best:
                is_better = False
                reason = ""
                
                # Get previous best info
                prev_significant = best_epoch_info.get('has_significant_fast_same', False)
                prev_effect = best_epoch_info.get('fast_same_effect_size', -999)
                prev_accuracy = best_epoch_info.get('accuracy', 0)
                
                # Priority 1: Significant fast-same effect
                if has_significant_fast_same and not prev_significant:
                    # Current has significance, previous doesn't
                    is_better = True
                    reason = "First epoch with significant fast-same effect!"
                elif has_significant_fast_same and prev_significant:
                    # Both have significance: compare effect sizes (Priority 2)
                    if effect_size > prev_effect:
                        is_better = True
                        reason = f"Larger effect size ({effect_size:.4f} > {prev_effect:.4f})"
                    elif abs(effect_size - prev_effect) < 0.001:
                        # Effect sizes essentially equal: compare accuracy (Priority 3)
                        if test_accuracy > prev_accuracy:
                            is_better = True
                            reason = f"Similar effect, higher accuracy ({test_accuracy*100:.2f}% > {prev_accuracy*100:.2f}%)"
                elif not has_significant_fast_same and not prev_significant:
                    # Neither has significance: still track best by effect size
                    if effect_size > prev_effect:
                        is_better = True
                        reason = f"Better effect size (but not significant yet)"
                    elif abs(effect_size - prev_effect) < 0.001 and test_accuracy > prev_accuracy:
                        is_better = True
                        reason = f"Similar effect, higher accuracy (not significant yet)"
                # If prev has significance but current doesn't: is_better stays False
                
                if is_better:
                    # Update best epoch info
                    best_epoch_info = {
                        'epoch': epoch + 1,
                        'accuracy': test_accuracy,
                        'fast_same_effect_size': effect_size,
                        'p_value': ttest_result.pvalue,
                        'match_mean': match_mean,
                        'mismatch_mean': mismatch_mean,
                        'has_significant_fast_same': has_significant_fast_same,
                        'meets_accuracy': meets_accuracy,
                        'ttest_statistic': ttest_result.statistic
                    }
                    best_model_state = model.state_dict().copy()
                    print(f"  ★ NEW BEST: {reason}")
                    
                    # Only save epoch snapshot if it has SIGNIFICANT fast-same effect
                    if has_significant_fast_same:
                        epoch_snapshot = {
                            'epoch': epoch + 1,
                            'model_state': model.state_dict().copy(),
                            'accuracy': test_accuracy,
                            'fast_same_effect_size': effect_size,
                            'p_value': ttest_result.pvalue,
                            'match_mean': match_mean,
                            'mismatch_mean': mismatch_mean,
                            'has_significant_fast_same': has_significant_fast_same,
                            'meets_accuracy': meets_accuracy,
                            'ttest_statistic': ttest_result.statistic,
                            'reason': reason
                        }
                        all_best_epochs.append(epoch_snapshot)
                        print(f"    ✓ SIGNIFICANT - Saved for visualization (#{len(all_best_epochs)})")
                    else:
                        print(f"    ✗ Not significant (p={ttest_result.pvalue:.4f}) - Not saved")
        
        print("-"*60)

        # Clear GPU cache to prevent memory accumulation
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save final model if path provided and save_model is True
    if save_path and save_model:
        torch.save(model.state_dict(), save_path)
        print(f"\nFinal model saved to: {save_path}")

    # Save best model if we have one and save_model is True
    if save_best and best_model_state is not None and save_path and save_model:
        best_model_path = save_path.replace('.pth', '_BEST_fast_same.pth')
        
        # Save with full information
        torch.save({
            'model_state_dict': best_model_state,
            'epoch': best_epoch_info['epoch'],
            'accuracy': best_epoch_info['accuracy'],
            'fast_same_effect_size': best_epoch_info['fast_same_effect_size'],
            'p_value': best_epoch_info['p_value'],
            'match_mean': best_epoch_info['match_mean'],
            'mismatch_mean': best_epoch_info['mismatch_mean'],
            'has_significant_fast_same': best_epoch_info['has_significant_fast_same'],
            'meets_accuracy': best_epoch_info['meets_accuracy'],
            'ttest_statistic': best_epoch_info['ttest_statistic'],
            'history': history
        }, best_model_path)
        
        print(f"\n" + "="*60)
        print("BEST MODEL SUMMARY")
        print("="*60)
        print(f"Best Epoch: {best_epoch_info['epoch']}/{num_epochs}")
        print(f"\nSelection Priority Applied:")
        print(f"  1. Significant Fast-Same: {'✓ YES' if best_epoch_info['has_significant_fast_same'] else '✗ NO'} (p={best_epoch_info['p_value']:.4f})")
        print(f"  2. Effect Size: {best_epoch_info['fast_same_effect_size']:.4f}")
        print(f"  3. Accuracy: {best_epoch_info['accuracy']*100:.2f}%")
        print(f"\nDetails:")
        print(f"  Match Mean: {best_epoch_info['match_mean']:.4f}")
        print(f"  Mismatch Mean: {best_epoch_info['mismatch_mean']:.4f}")
        print(f"  Meets Accuracy Threshold: {'✓ YES' if best_epoch_info['meets_accuracy'] else '✗ NO'}")
        print(f"\nSaved to: {best_model_path}")
        print("="*60)

    # Note: Final filtering is now handled in the main function to avoid re-evaluation
    print(f"\nTraining completed. Found {len(accuracy_threshold_epochs)} epochs meeting accuracy threshold.")
    print(f"Final filtering will be performed in main function using collected results.")

    return history, best_epoch_info, all_best_epochs, accuracy_threshold_epochs, all_threshold_results


# ==================== ANALYSIS AND PLOTTING FUNCTION ====================
# APA Style Setup
def setup_apa_style():
    """Configure APA 7th edition style"""
    sns.set_style("white")
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False,
    })
    
    # Professional color palette (inspired by scientific publications)
    colors = {
        'match': '#297270',        # Professional blue for match
        'mismatch': '#e66d50',     # Professional red for mismatch
        'match_light': '#299d8f',
        'mismatch_light': '#f3a361'
    }
    return colors


def despine(ax):
    """Remove top and right spines"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)


def create_half_raincloud_plot(ax, data, position, color, side='right', width=0.6):
    """
    Create a proper half-raincloud plot with distribution on one side and points on the other
    
    Parameters:
    -----------
    ax : matplotlib axes
    data : array-like
        The data to plot
    position : float
        X-axis position
    color : str
        Color for the plot
    side : str
        'right' = distribution on right, points on left
        'left' = distribution on left, points on right
    width : float
        Width of the violin plot
    """
    from scipy import stats as scipy_stats

    data = np.array(data)

    # Filter out NaN and Inf values
    valid_mask = np.isfinite(data)
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"Warning: Removing {n_invalid} NaN/Inf values from data (total: {len(data)})")
        data = data[valid_mask]

    # Check if we have any valid data left
    if len(data) == 0:
        print("Error: No valid data points after filtering NaN/Inf values")
        return

    # Check if data has variance (not all identical)
    data_std = np.std(data)
    if data_std < 1e-6:  # Essentially constant data
        # Handle constant data: just draw a horizontal line
        data_mean = np.mean(data)
        y_range = np.array([data_mean])
        density = np.array([width])  # Full width for the line
    else:
        # Calculate KDE for smooth distribution
        kde = scipy_stats.gaussian_kde(data)

        # Create range for KDE
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        y_range = np.linspace(data_min - 0.1 * data_range,
                             data_max + 0.1 * data_range, 200)
        density = kde(y_range)
    
    # Normalize density to fit width
    density = density / density.max() * width
    
    # Create half violin on specified side
    if side == 'right':
        # Distribution on right side
        x_density = position + density
        ax.fill_betweenx(y_range, position, x_density, 
                         alpha=0.6, color=color, edgecolor='black', linewidth=1)
        
        # Points on left side
        jitter_base = position - width * 0.4
        jitter_amount = width * 0.15
    else:
        # Distribution on left side
        x_density = position - density
        ax.fill_betweenx(y_range, x_density, position,
                         alpha=0.6, color=color, edgecolor='black', linewidth=1)
        
        # Points on right side
        jitter_base = position + width * 0.4
        jitter_amount = width * 0.15
    
    # Add box plot in the center (very thin)
    bp = ax.boxplot([data], positions=[position], widths=width*0.15,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2),
                    medianprops=dict(color='black', linewidth=2))
    
    # Add strip plot on opposite side with jitter
    np.random.seed(42)  # For reproducibility
    jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(data))
    ax.scatter([jitter_base]*len(data) + jitter, data, 
              alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.5)


def create_raincloud_plot(ax, data, position, color, width=0.3):
    """Deprecated: Use create_half_raincloud_plot instead"""
    create_half_raincloud_plot(ax, data, position, color, side='right', width=width)



def analyze_and_plot_decision_time(model, test_loader, train_loader=None, train_dataset=None,
                                   test_dataset=None, device='cpu', save_path=None,
                                   time_steps=20, accuracy_threshold=0.7,
                                   model_save_path=None, full_model=None, return_full_data=False,
                                   condition_name='baseline', show_incorrect=False, random_seed=None,
                                   save_plots=True, save_model=True):
    """
    Enhanced APA-style evaluation with:
    - Raincloud plots  
    - T-test significance analysis
    - CSV export of decision times (both normalized and original scale)
    - Conditional model saving based on accuracy and fast-same effect
    - RT denormalization for human-scale comparison
    - Option to return full test data for later analysis
    """
    from scipy import stats
    
    # Debug info
    print(f"\n  [analyze_and_plot_decision_time] Called with:")
    print(f"    save_path: {save_path}")
    print(f"    test_dataset: {test_dataset is not None}")
    if test_dataset:
        print(f"    has denormalize_rt: {hasattr(test_dataset, 'denormalize_rt')}")
    print(f"    return_full_data: {return_full_data}")
    
    # Setup APA style
    apa_colors = setup_apa_style()
    
    model.eval()
    model.to(device)

    # Storage for results with detailed information
    model_match_times = []
    model_mismatch_times = []
    human_match_times = []
    human_mismatch_times = []

    # Storage for correct/incorrect predictions (for ablation analysis)
    model_match_correct_times = []
    model_match_incorrect_times = []
    model_mismatch_correct_times = []
    model_mismatch_incorrect_times = []

    # Storage for denormalized (original scale) RT
    model_match_times_ms = []
    model_mismatch_times_ms = []
    human_match_times_ms = []
    human_mismatch_times_ms = []

    # Storage for correct/incorrect predictions (denormalized)
    model_match_correct_times_ms = []
    model_match_incorrect_times_ms = []
    model_mismatch_correct_times_ms = []
    model_mismatch_incorrect_times_ms = []
    
    # For CSV export
    all_trials = []

    correct_predictions = 0
    total_predictions = 0

    all_model_times = []
    all_human_times = []
    
    # Check if we can denormalize
    can_denormalize = (test_dataset is not None and 
                      hasattr(test_dataset, 'denormalize_rt'))

    print("\nEvaluating model on test set...")
    if can_denormalize:
        print(f"RT Normalization Info:")
        print(f"  Original RT range: {test_dataset.rt_min:.2f} - {test_dataset.rt_max:.2f} ms")
        print(f"  RT range: {test_dataset.rt_range:.2f} ms")

    with torch.no_grad():
        for batch_idx, (img1, img2, labels, rt_human) in enumerate(tqdm(test_loader, desc="Testing")):
            non_blocking = device.type == 'cuda'
            img1 = img1.to(device, non_blocking=non_blocking)
            img2 = img2.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            decision_logits, decision_time, _, _ = model(img1, img2)
            pred_labels = decision_logits.argmax(dim=-1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = pred_labels[i].item()
                model_dt = decision_time[i].item()
                human_rt = rt_human[i].item()
                
                is_correct = (pred_label == true_label)
                condition = 'match' if true_label == 1 else 'mismatch'

                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Denormalize RT values to original scale (milliseconds)
                if can_denormalize:
                    model_dt_ms = test_dataset.denormalize_rt(model_dt)
                    human_rt_ms = test_dataset.denormalize_rt(human_rt)
                else:
                    model_dt_ms = None
                    human_rt_ms = None
                
                # Store for CSV
                trial_data = {
                    'trial_id': batch_idx * test_loader.batch_size + i,
                    'condition': condition,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'correct': is_correct,
                    'model_decision_time_normalized': model_dt,
                    'human_rt_normalized': human_rt
                }
                
                # Add denormalized values if available
                if can_denormalize:
                    trial_data['model_decision_time_ms'] = float(model_dt_ms)
                    trial_data['human_rt_ms'] = float(human_rt_ms)
                
                all_trials.append(trial_data)

                # Store data by condition (match/mismatch) and correctness
                if true_label == 1:  # Match trials
                    model_match_times.append(model_dt)
                    human_match_times.append(human_rt)
                    if can_denormalize:
                        model_match_times_ms.append(model_dt_ms)
                        human_match_times_ms.append(human_rt_ms)

                    # Separate by correctness for ablation analysis
                    if is_correct:
                        model_match_correct_times.append(model_dt)
                        if can_denormalize:
                            model_match_correct_times_ms.append(model_dt_ms)
                    else:
                        model_match_incorrect_times.append(model_dt)
                        if can_denormalize:
                            model_match_incorrect_times_ms.append(model_dt_ms)
                else:  # Mismatch trials
                    model_mismatch_times.append(model_dt)
                    human_mismatch_times.append(human_rt)
                    if can_denormalize:
                        model_mismatch_times_ms.append(model_dt_ms)
                        human_mismatch_times_ms.append(human_rt_ms)

                    # Separate by correctness for ablation analysis
                    if is_correct:
                        model_mismatch_correct_times.append(model_dt)
                        if can_denormalize:
                            model_mismatch_correct_times_ms.append(model_dt_ms)
                    else:
                        model_mismatch_incorrect_times.append(model_dt)
                        if can_denormalize:
                            model_mismatch_incorrect_times_ms.append(model_dt_ms)

                all_model_times.append(model_dt)
                all_human_times.append(human_rt)

    # Calculate metrics
    accuracy = correct_predictions / total_predictions
    correlation = np.corrcoef(all_model_times, all_human_times)[0, 1]

    # Perform t-tests (on normalized values for consistency)
    model_ttest = stats.ttest_ind(model_match_times, model_mismatch_times)
    human_ttest = stats.ttest_ind(human_match_times, human_mismatch_times)

    # ==================== REPRESENTATION DISTANCE ANALYSIS ====================
    match_dists, mismatch_dists = [], []

    print("\nComputing representation distance analysis...")
    with torch.no_grad():
        for img1, img2, labels, _ in tqdm(test_loader, desc="Distance Analysis", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            _, _, _, _, z_dist = model.forward_with_diagnostics(img1, img2)
            dist_np = z_dist.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                if labels_np[i] == 1:  # Match
                    match_dists.append(dist_np[i])
                else:                  # Mismatch
                    mismatch_dists.append(dist_np[i])

    mean_match_dist = np.mean(match_dists) if match_dists else 0.0
    mean_mismatch_dist = np.mean(mismatch_dists) if mismatch_dists else 0.0
    delta_dist = mean_mismatch_dist - mean_match_dist

    # Representation distance contribution to FSE
    match_mean = np.mean(model_match_times)
    mismatch_mean = np.mean(model_mismatch_times)
    delta_rt = mismatch_mean - match_mean  # FSE = Mismatch slower than Match

    print(f"\n  [Representation Distance Analysis]")
    print(f"  Match dist:      {mean_match_dist:.4f}")
    print(f"  Mismatch dist:   {mean_mismatch_dist:.4f}")
    print(f"  ΔDist:           {delta_dist:.4f}")

    # ==================== EVALUATE ON TRAINING DATA ====================
    train_match_times_ms = []
    train_mismatch_times_ms = []

    if train_loader is not None and train_dataset is not None:
        print("\nEvaluating on training set...")
        model.eval()
        with torch.no_grad():
            for img1, img2, labels, rt_human in tqdm(train_loader, desc="Training Set", leave=False):
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)

                decision_logits, decision_time, _, _ = model(img1, img2)

                for i in range(len(labels)):
                    true_label = labels[i].item()
                    human_rt = rt_human[i].item()

                    # Denormalize
                    if hasattr(train_dataset, 'denormalize_rt'):
                        human_rt_ms = train_dataset.denormalize_rt(human_rt)
                    else:
                        human_rt_ms = human_rt

                    # Separate by condition
                    if true_label == 1:  # match
                        train_match_times_ms.append(human_rt_ms)
                    else:  # mismatch
                        train_mismatch_times_ms.append(human_rt_ms)

    
    # Also perform t-tests on denormalized values if available
    if can_denormalize:
        model_ttest_ms = stats.ttest_ind(model_match_times_ms, model_mismatch_times_ms)
        human_ttest_ms = stats.ttest_ind(human_match_times_ms, human_mismatch_times_ms)
    
    # Check for fast-same effect (match < mismatch)
    model_fast_same = np.mean(model_match_times) < np.mean(model_mismatch_times)
    human_fast_same = np.mean(human_match_times) < np.mean(human_mismatch_times)
    
    # Check if model shows significant fast-same effect
    model_significant_fast_same = (model_ttest.pvalue < 0.05) and model_fast_same

    results = {
        'accuracy': accuracy,
        'correlation': correlation,
        'model_match': {
            'mean': np.mean(model_match_times),
            'std': np.std(model_match_times),
            'n': len(model_match_times),
            'data': model_match_times
        },
        'model_mismatch': {
            'mean': np.mean(model_mismatch_times),
            'std': np.std(model_mismatch_times),
            'n': len(model_mismatch_times),
            'data': model_mismatch_times
        },
        'human_match': {
            'mean': np.mean(human_match_times),
            'std': np.std(human_match_times),
            'n': len(human_match_times),
            'data': human_match_times
        },
        'human_mismatch': {
            'mean': np.mean(human_mismatch_times),
            'std': np.std(human_mismatch_times),
            'n': len(human_mismatch_times),
            'data': human_mismatch_times
        },
        'model_ttest': {'statistic': model_ttest.statistic, 'pvalue': model_ttest.pvalue},
        'human_ttest': {'statistic': human_ttest.statistic, 'pvalue': human_ttest.pvalue},
        'model_fast_same_effect': model_fast_same,
        'human_fast_same_effect': human_fast_same,
        'model_significant_fast_same': model_significant_fast_same,
        'distance_analysis': {
            'mean_match_dist': mean_match_dist,
            'mean_mismatch_dist': mean_mismatch_dist,
            'delta_dist': delta_dist
        }
    }

    # Export decision times to CSV
    if save_path:
        csv_path = save_path.replace('.png', '_decision_times.csv').replace('.pdf', '_decision_times.csv')
        trials_df = pd.DataFrame(all_trials)
        trials_df.to_csv(csv_path, index=False)
        print(f"Decision times exported to: {csv_path}")
        
        # Export significance analysis
        sig_path = save_path.replace('.png', '_significance.csv').replace('.pdf', '_significance.csv')
        
        sig_data = {
            'Source': ['FS-Net (Normalized)', 'Human Data (Normalized)'],
            'Random_Seed': [random_seed if random_seed is not None else np.nan, np.nan],
            'Accuracy': [accuracy, np.nan],  # Only model has accuracy metric
            'Match_Mean_Normalized': [results['model_match']['mean'], results['human_match']['mean']],
            'Match_SD_Normalized': [results['model_match']['std'], results['human_match']['std']],
            'Mismatch_Mean_Normalized': [results['model_mismatch']['mean'], results['human_mismatch']['mean']],
            'Mismatch_SD_Normalized': [results['model_mismatch']['std'], results['human_mismatch']['std']],
            'T_Statistic': [model_ttest.statistic, human_ttest.statistic],
            'P_Value': [model_ttest.pvalue, human_ttest.pvalue],
            'Significant': [model_ttest.pvalue < 0.05, human_ttest.pvalue < 0.05],
            'Fast_Same_Effect': [model_fast_same, human_fast_same]
        }
        
        # Add millisecond-scale statistics if available
        if can_denormalize:
            sig_data['Match_Mean_MS'] = [np.mean(model_match_times_ms), np.mean(human_match_times_ms)]
            sig_data['Match_SD_MS'] = [np.std(model_match_times_ms), np.std(human_match_times_ms)]
            sig_data['Mismatch_Mean_MS'] = [np.mean(model_mismatch_times_ms), np.mean(human_mismatch_times_ms)]
            sig_data['Mismatch_SD_MS'] = [np.std(model_mismatch_times_ms), np.std(human_mismatch_times_ms)]
            sig_data['Effect_Size_MS'] = [
                np.mean(model_mismatch_times_ms) - np.mean(model_match_times_ms),
                np.mean(human_mismatch_times_ms) - np.mean(human_match_times_ms)
            ]
            sig_data['T_Statistic_MS'] = [model_ttest_ms.statistic, human_ttest_ms.statistic]
            sig_data['P_Value_MS'] = [model_ttest_ms.pvalue, human_ttest_ms.pvalue]
        
        sig_df = pd.DataFrame(sig_data)
        sig_df.to_csv(sig_path, index=False)
        print(f"Significance analysis exported to: {sig_path}")

    # Print results
    print("\n" + "="*60)
    print("Test Set Evaluation Results")
    print("="*60)
    print(f"\nModel Accuracy: {accuracy*100:.2f}%")
    print(f"RT Correlation: {correlation:.4f}")

    print("\n" + "-"*60)
    print("Decision Time Analysis (Normalized Scale)")
    print("-"*60)
    print(f"Match Trials (n={results['model_match']['n']}):")
    print(f"  Model DT: {results['model_match']['mean']:.4f} ± {results['model_match']['std']:.4f}")
    print(f"  Human RT: {results['human_match']['mean']:.4f} ± {results['human_match']['std']:.4f}")

    print(f"\nMismatch Trials (n={results['model_mismatch']['n']}):")
    print(f"  Model DT: {results['model_mismatch']['mean']:.4f} ± {results['model_mismatch']['std']:.4f}")
    print(f"  Human RT: {results['human_mismatch']['mean']:.4f} ± {results['human_mismatch']['std']:.4f}")
    
    # Print millisecond-scale statistics if available
    if can_denormalize:
        print("\n" + "-"*60)
        print("Decision Time Analysis (Original Scale - Milliseconds)")
        print("-"*60)
        print(f"Match Trials:")
        print(f"  Model DT: {np.mean(model_match_times_ms):.2f} ± {np.std(model_match_times_ms):.2f} ms")
        print(f"  Human RT: {np.mean(human_match_times_ms):.2f} ± {np.std(human_match_times_ms):.2f} ms")
        
        print(f"\nMismatch Trials:")
        print(f"  Model DT: {np.mean(model_mismatch_times_ms):.2f} ± {np.std(model_mismatch_times_ms):.2f} ms")
        print(f"  Human RT: {np.mean(human_mismatch_times_ms):.2f} ± {np.std(human_mismatch_times_ms):.2f} ms")
        
        print(f"\nEffect Sizes:")
        model_effect_ms = np.mean(model_mismatch_times_ms) - np.mean(model_match_times_ms)
        human_effect_ms = np.mean(human_mismatch_times_ms) - np.mean(human_match_times_ms)
        print(f"  Model: {model_effect_ms:.2f} ms (Mismatch - Match)")
        print(f"  Human: {human_effect_ms:.2f} ms (Mismatch - Match)")
    
    print("\n" + "-"*60)
    print("Statistical Significance (T-Tests)")
    print("-"*60)
    print(f"FS-Net: t={model_ttest.statistic:.4f}, p={model_ttest.pvalue:.4f} {'***' if model_ttest.pvalue < 0.001 else '**' if model_ttest.pvalue < 0.01 else '*' if model_ttest.pvalue < 0.05 else 'ns'}")
    print(f"Human:  t={human_ttest.statistic:.4f}, p={human_ttest.pvalue:.4f} {'***' if human_ttest.pvalue < 0.001 else '**' if human_ttest.pvalue < 0.01 else '*' if human_ttest.pvalue < 0.05 else 'ns'}")
    
    print("\n" + "-"*60)
    print("Fast-Same Effect Check")
    print("-"*60)
    print(f"FS-Net Fast-Same Effect: {'✓ YES' if model_fast_same else '✗ NO'}")
    print(f"Human Fast-Same Effect: {'✓ YES' if human_fast_same else '✗ NO'}")
    print(f"FS-Net Significant Fast-Same: {'✓ YES' if model_significant_fast_same else '✗ NO'}")

    # Diagnostic: Check data variance and unique values
    print("\n" + "-"*60)
    print("Data Variance Diagnostic")
    print("-"*60)
    print(f"Sample sizes (should be equal):")
    print(f"  Model Match: n={len(model_match_times)}")
    print(f"  Human Match: n={len(human_match_times)}")
    print(f"  Model Mismatch: n={len(model_mismatch_times)}")
    print(f"  Human Mismatch: n={len(human_mismatch_times)}")

    print(f"\nUnique values in model output:")
    print(f"  Model Match: {len(np.unique(np.round(model_match_times, 4)))} unique values")
    print(f"  Model Mismatch: {len(np.unique(np.round(model_mismatch_times, 4)))} unique values")
    print(f"  Human Match: {len(np.unique(np.round(human_match_times, 4)))} unique values")
    print(f"  Human Mismatch: {len(np.unique(np.round(human_mismatch_times, 4)))} unique values")

    print(f"\nVariance comparison:")
    print(f"  Model Match std: {np.std(model_match_times):.6f}")
    print(f"  Human Match std: {np.std(human_match_times):.6f}")
    print(f"  Model Mismatch std: {np.std(model_mismatch_times):.6f}")
    print(f"  Human Mismatch std: {np.std(human_mismatch_times):.6f}")

    # Check if model meets criteria for saving
    meets_criteria = (accuracy >= accuracy_threshold) and model_significant_fast_same
    
    if meets_criteria and model_save_path and full_model:
        print("\n" + "="*60)
        print("MODEL MEETS CRITERIA - SAVING!")
        print("="*60)
        print(f"✓ Accuracy {accuracy*100:.2f}% >= {accuracy_threshold*100:.2f}%")
        print(f"✓ Significant fast-same effect (p={model_ttest.pvalue:.4f})")

        # Save the complete model (only if save_model is True)
        if save_model:
            best_model_path = model_save_path.replace('.pth', '_BEST_fast_same.pth')
            torch.save({
                'model_state_dict': full_model.state_dict(),
                'accuracy': accuracy,
                'correlation': correlation,
                'ttest_statistic': model_ttest.statistic,
                'ttest_pvalue': model_ttest.pvalue,
                'match_mean': results['model_match']['mean'],
                'mismatch_mean': results['model_mismatch']['mean'],
                'results': results
            }, best_model_path)
        print(f"✓ Complete model saved to: {best_model_path}")
    else:
        print("\n" + "="*60)
        print("MODEL DOES NOT MEET CRITERIA")
        print("="*60)
        if accuracy < accuracy_threshold:
            print(f"✗ Accuracy {accuracy*100:.2f}% < {accuracy_threshold*100:.2f}% (threshold)")
        else:
            print(f"✓ Accuracy {accuracy*100:.2f}% >= {accuracy_threshold*100:.2f}%")
        
        if not model_significant_fast_same:
            if not model_fast_same:
                print(f"✗ No fast-same effect (match mean >= mismatch mean)")
            else:
                print(f"✗ Fast-same effect not significant (p={model_ttest.pvalue:.4f} >= 0.05)")
        else:
            print(f"✓ Significant fast-same effect")

    # Create beautiful APA-style half-raincloud plot
    # Determine if we should show detailed breakdown (correct/incorrect) for ablation studies
    is_ablation_study = (condition_name in ['baseline', 'cl_only', 'noise_only', 'rtify_unsupervised_noise'] and show_incorrect)

    # Use millisecond scale if available, otherwise normalized scale
    if can_denormalize:
        plot_data = {
            'model_match': model_match_times_ms,
            'model_mismatch': model_mismatch_times_ms,
            'human_match': human_match_times_ms,
            'human_mismatch': human_mismatch_times_ms
        }
        # Add correct/incorrect breakdown for ablation studies
        if is_ablation_study:
            plot_data.update({
                'model_match_correct': model_match_correct_times_ms,
                'model_match_incorrect': model_match_incorrect_times_ms,
                'model_mismatch_correct': model_mismatch_correct_times_ms,
                'model_mismatch_incorrect': model_mismatch_incorrect_times_ms
            })
        y_label = 'Reaction Time (ms)'
        print(f"\nGenerating raincloud plot with millisecond scale...")
    else:
        plot_data = {
            'model_match': model_match_times,
            'model_mismatch': model_mismatch_times,
            'human_match': human_match_times,
            'human_mismatch': human_mismatch_times
        }
        # Add correct/incorrect breakdown for ablation studies
        if is_ablation_study:
            plot_data.update({
                'model_match_correct': model_match_correct_times,
                'model_match_incorrect': model_match_incorrect_times,
                'model_mismatch_correct': model_mismatch_correct_times,
                'model_mismatch_incorrect': model_mismatch_incorrect_times
            })
        y_label = 'Normalized Time'
        print(f"\nGenerating raincloud plot with normalized scale...")

    if is_ablation_study:
        print(f"  Ablation study mode: showing correct/incorrect breakdown")

    # Choose figure size and layout based on whether it's ablation study
    if is_ablation_study:
        fig, ax = plt.subplots(1, 1, figsize=(22, 7))  # Wider for 6 model distributions
    else:
        fig, ax = plt.subplots(1, 1, figsize=(18, 7))  # Standard width
    fig.patch.set_facecolor('white')

    # Define positions based on visualization mode
    if is_ablation_study:
        # Ablation mode: show correct/incorrect breakdown (6 model + 2 human = 8 distributions)
        positions = {
            # Model predictions - Match (correct and incorrect)
            'model_match_correct': 2,
            'model_match_incorrect': 3.5,
            # Model predictions - Mismatch (correct and incorrect)
            'model_mismatch_correct': 6,
            'model_mismatch_incorrect': 7.5,
            # Test data (human)
            'test_match': 11,
            'test_mismatch': 12.5
        }
    else:
        # Standard mode: show aggregated model predictions
        positions = {
            # Training data (human)
            'train_match': 1,
            'train_mismatch': 2.5,
            # Model predictions (test)
            'model_match': 5.5,
            'model_mismatch': 7,
            # Test data (human)
            'test_match': 10,
            'test_mismatch': 11.5
        }

    # Create half-raincloud plots based on mode
    if is_ablation_study:
        # Ablation mode: show correct/incorrect breakdown
        # Define gray color for incorrect predictions
        gray_color = '#808080'

        # Model Match - Correct (use match color)
        if len(plot_data['model_match_correct']) > 0:
            create_half_raincloud_plot(ax, plot_data['model_match_correct'],
                                      positions['model_match_correct'],
                                      apa_colors['match'], side='right', width=0.6)

        # Model Match - Incorrect (use gray)
        if len(plot_data['model_match_incorrect']) > 0:
            create_half_raincloud_plot(ax, plot_data['model_match_incorrect'],
                                      positions['model_match_incorrect'],
                                      gray_color, side='right', width=0.6)

        # Model Mismatch - Correct (use mismatch color)
        if len(plot_data['model_mismatch_correct']) > 0:
            create_half_raincloud_plot(ax, plot_data['model_mismatch_correct'],
                                      positions['model_mismatch_correct'],
                                      apa_colors['mismatch'], side='right', width=0.6)

        # Model Mismatch - Incorrect (use gray)
        if len(plot_data['model_mismatch_incorrect']) > 0:
            create_half_raincloud_plot(ax, plot_data['model_mismatch_incorrect'],
                                      positions['model_mismatch_incorrect'],
                                      gray_color, side='right', width=0.6)

        # Human Test data
        create_half_raincloud_plot(ax, plot_data['human_match'], positions['test_match'],
                                  apa_colors['match'], side='right', width=0.6)
        create_half_raincloud_plot(ax, plot_data['human_mismatch'], positions['test_mismatch'],
                                  apa_colors['mismatch'], side='right', width=0.6)
    else:
        # Standard mode: show aggregated predictions
        # Plot training data (if available)
        if train_loader is not None and len(train_match_times_ms) > 0:
            create_half_raincloud_plot(ax, train_match_times_ms, positions['train_match'],
                                      apa_colors['match'], side='right', width=0.6)
            create_half_raincloud_plot(ax, train_mismatch_times_ms, positions['train_mismatch'],
                                      apa_colors['mismatch'], side='right', width=0.6)

        # Model predictions (test set)
        create_half_raincloud_plot(ax, plot_data['model_match'], positions['model_match'],
                                  apa_colors['match'], side='right', width=0.6)
        create_half_raincloud_plot(ax, plot_data['model_mismatch'], positions['model_mismatch'],
                                  apa_colors['mismatch'], side='right', width=0.6)

        # Test data (human)
        create_half_raincloud_plot(ax, plot_data['human_match'], positions['test_match'],
                                  apa_colors['match'], side='right', width=0.6)
        create_half_raincloud_plot(ax, plot_data['human_mismatch'], positions['test_mismatch'],
                                  apa_colors['mismatch'], side='right', width=0.6)

    # Add sample size annotations
    y_max = ax.get_ylim()[1]
    y_annotation = y_max * 1.05  # Position above the plot

    if is_ablation_study:
        # Ablation mode: annotate correct/incorrect counts
        if len(plot_data['model_match_correct']) > 0:
            ax.text(positions['model_match_correct'], y_annotation,
                   f"n={len(plot_data['model_match_correct'])}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        if len(plot_data['model_match_incorrect']) > 0:
            ax.text(positions['model_match_incorrect'], y_annotation,
                   f"n={len(plot_data['model_match_incorrect'])}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        if len(plot_data['model_mismatch_correct']) > 0:
            ax.text(positions['model_mismatch_correct'], y_annotation,
                   f"n={len(plot_data['model_mismatch_correct'])}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        if len(plot_data['model_mismatch_incorrect']) > 0:
            ax.text(positions['model_mismatch_incorrect'], y_annotation,
                   f"n={len(plot_data['model_mismatch_incorrect'])}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Human annotations
        ax.text(positions['test_match'], y_annotation, f"n={len(plot_data['human_match'])}",
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(positions['test_mismatch'], y_annotation, f"n={len(plot_data['human_mismatch'])}",
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        # Standard mode: annotate aggregated counts
        ax.text(positions['model_match'], y_annotation, f"n={len(plot_data['model_match'])}",
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(positions['model_mismatch'], y_annotation, f"n={len(plot_data['model_mismatch'])}",
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(positions['test_match'], y_annotation, f"n={len(plot_data['human_match'])}",
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(positions['test_mismatch'], y_annotation, f"n={len(plot_data['human_mismatch'])}",
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize axes with better styling
    # Map condition names to display labels
    condition_labels = {
        'baseline': 'RTify(supervised)',
        'cl_only': 'RTify(contrastive only)',
        'noise_only': 'RTify(noise only)',
        'rtify_unsupervised_noise': 'RTify(unsupervised, noise only)'
    }

    # Set x-axis labels based on mode
    if is_ablation_study:
        # Ablation mode: show model name with condition for correct/incorrect breakdown
        model_label = condition_labels.get(condition_name, "FS-Net")
        ax.set_xticks([2.75, 6.75, 11.75])  # Center positions for each group
        ax.set_xticklabels([f'{model_label} - Match',
                           f'{model_label} - Mismatch',
                           'Human(testing data)'],
                           fontsize=14, fontweight='bold', family='sans-serif')
        ax.set_xlim(0, 14)
    elif train_loader is not None and len(train_match_times_ms) > 0:
        # Standard mode with training data
        ax.set_xticks([1.75, 6.25, 10.75])
        ax.set_xticklabels(['Human(training data)',
                           f'{condition_labels.get(condition_name, "FS-Net")}',
                           'Human(testing data)'],
                           fontsize=14, fontweight='bold', family='sans-serif')
        ax.set_xlim(0, 13)
    else:
        # Standard mode without training data
        ax.set_xticks([6.25, 10.75])
        ax.set_xticklabels([f'{condition_labels.get(condition_name, "FS-Net")}',
                           'Human(testing data)'],
                           fontsize=14, fontweight='bold', family='sans-serif')
        ax.set_xlim(4.5, 13)

    ax.set_ylabel(y_label, fontsize=14, fontweight='bold', family='sans-serif')

    
    # APA format: clean background with no grid lines
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    
    # Add condition labels below each raincloud
    y_label_pos = ax.get_ylim()[0] - 0.08*(ax.get_ylim()[1]-ax.get_ylim()[0])

    if is_ablation_study:
        # Ablation mode: show correct/incorrect labels
        gray_color = '#808080'

        # Model Match labels
        if len(plot_data['model_match_correct']) > 0:
            ax.text(positions['model_match_correct'], y_label_pos, 'Correct',
                   ha='center', va='top', fontsize=11, style='italic', color=apa_colors['match'])
        if len(plot_data['model_match_incorrect']) > 0:
            ax.text(positions['model_match_incorrect'], y_label_pos, 'Incorrect',
                   ha='center', va='top', fontsize=11, style='italic', color=gray_color)

        # Model Mismatch labels
        if len(plot_data['model_mismatch_correct']) > 0:
            ax.text(positions['model_mismatch_correct'], y_label_pos, 'Correct',
                   ha='center', va='top', fontsize=11, style='italic', color=apa_colors['mismatch'])
        if len(plot_data['model_mismatch_incorrect']) > 0:
            ax.text(positions['model_mismatch_incorrect'], y_label_pos, 'Incorrect',
                   ha='center', va='top', fontsize=11, style='italic', color=gray_color)

        # Human Test data labels
        ax.text(positions['test_match'], y_label_pos, 'Match',
               ha='center', va='top', fontsize=11, style='italic', color=apa_colors['match'])
        ax.text(positions['test_mismatch'], y_label_pos, 'Mismatch',
               ha='center', va='top', fontsize=11, style='italic', color=apa_colors['mismatch'])
    else:
        # Standard mode: show match/mismatch labels
        # Training data labels (if available)
        if train_loader is not None and len(train_match_times_ms) > 0:
            ax.text(positions['train_match'], y_label_pos, 'Match',
                   ha='center', va='top', fontsize=11, style='italic', color=apa_colors['match'])
            ax.text(positions['train_mismatch'], y_label_pos, 'Mismatch',
                   ha='center', va='top', fontsize=11, style='italic', color=apa_colors['mismatch'])

        # Model labels
        ax.text(positions['model_match'], y_label_pos, 'Match',
               ha='center', va='top', fontsize=11, style='italic', color=apa_colors['match'])
        ax.text(positions['model_mismatch'], y_label_pos, 'Mismatch',
               ha='center', va='top', fontsize=11, style='italic', color=apa_colors['mismatch'])

        # Test data labels
        ax.text(positions['test_match'], y_label_pos, 'Match',
               ha='center', va='top', fontsize=11, style='italic', color=apa_colors['match'])
        ax.text(positions['test_mismatch'], y_label_pos, 'Mismatch',
               ha='center', va='top', fontsize=11, style='italic', color=apa_colors['mismatch'])
    
    # Add significance bars with stars (improved version)
    def add_significance_bar(ax, x1, x2, y, p_value, y_range, color='black'):
        """Add significance bar with stars"""
        # Bar height relative to y range
        bar_offset = y_range * 0.03
        text_offset = y_range * 0.05
        
        # Bar
        ax.plot([x1, x1, x2, x2], [y, y+bar_offset, y+bar_offset, y], 
               color=color, linewidth=1.5, solid_capstyle='round')
        
        # Determine significance level
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        # Display stars
        ax.text((x1+x2)/2, y+text_offset, stars, ha='center', va='bottom', 
               fontsize=13, fontweight='bold', color=color)
    
    # Calculate positions for significance bars
    all_data = (list(plot_data['model_match']) + list(plot_data['model_mismatch']) + 
                list(plot_data['human_match']) + list(plot_data['human_mismatch']))
    y_min, y_max = min(all_data), max(all_data)
    y_range = y_max - y_min
    
    # Use appropriate t-test results
    if can_denormalize:
        model_ttest_plot = model_ttest_ms
        human_ttest_plot = human_ttest_ms
    else:
        model_ttest_plot = model_ttest
        human_ttest_plot = human_ttest

    # Add significance bars at the top
    bar_y_position = y_max * 0.93

    # FS-Net significance bar
    add_significance_bar(ax, positions['model_match'], positions['model_mismatch'],
                        bar_y_position, model_ttest_plot.pvalue, y_range, color='dimgray')

    # Human Data significance bar
    add_significance_bar(ax, positions['test_match'], positions['test_mismatch'], 
                        bar_y_position, human_ttest_plot.pvalue, y_range, color='dimgray')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=apa_colors['match'], edgecolor='black', label='Match', alpha=0.6),
        Patch(facecolor=apa_colors['mismatch'], edgecolor='black', label='Mismatch', alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, fontsize=11)
    
    despine(ax)
    plt.tight_layout()

    if save_path and save_plots:
        # Save as PDF
        pdf_path = save_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"Raincloud plot saved to: {pdf_path}")

        # Also save as PNG for convenience
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Raincloud plot (PNG) saved to: {save_path}")

    plt.close()

    # ==================== CREATE NORMALIZED HISTOGRAM (Equal Area) ====================
    # This visualization makes it obvious that sample sizes are equal
    # because both distributions have the same total area (1.0)

    if save_path:
        fig_hist, axes_hist = plt.subplots(1, 2, figsize=(14, 6))
        fig_hist.patch.set_facecolor('white')

        # Left panel: Model vs Human (Match trials)
        ax_match = axes_hist[0]
        # Model data: filled bars with APA color
        ax_match.hist(plot_data['model_match'], bins=30, alpha=0.6,
                     color=apa_colors['match'], edgecolor='black', linewidth=1,
                     density=True, label=f'{condition_labels.get(condition_name, "FS-Net")} (n={len(plot_data["model_match"])})')
        # Human data: outline only (histtype='step') for contrast
        ax_match.hist(plot_data['human_match'], bins=30,
                     histtype='step', color='black', linewidth=2,
                     density=True, label=f'Human(testing data) (n={len(plot_data["human_match"])})')
        ax_match.set_xlabel(y_label, fontsize=14, fontweight='bold', family='sans-serif')
        ax_match.set_ylabel('Probability Density', fontsize=14, fontweight='bold', family='sans-serif')
        ax_match.set_title('Match Trials', fontsize=14, fontweight='bold', family='sans-serif')
        ax_match.legend(loc='upper right', frameon=True, fontsize=11, fancybox=True, shadow=True)
        # APA style: no grid, remove top and right spines
        ax_match.yaxis.grid(False)
        ax_match.xaxis.grid(False)
        despine(ax_match)

        # Right panel: Model vs Human (Mismatch trials)
        ax_mismatch = axes_hist[1]
        # Model data: filled bars with APA color
        ax_mismatch.hist(plot_data['model_mismatch'], bins=30, alpha=0.6,
                        color=apa_colors['mismatch'], edgecolor='black', linewidth=1,
                        density=True, label=f'{condition_labels.get(condition_name, "FS-Net")} (n={len(plot_data["model_mismatch"])})')
        # Human data: outline only (histtype='step') for contrast
        ax_mismatch.hist(plot_data['human_mismatch'], bins=30,
                        histtype='step', color='black', linewidth=2,
                        density=True, label=f'Human(testing data) (n={len(plot_data["human_mismatch"])})')
        ax_mismatch.set_xlabel(y_label, fontsize=14, fontweight='bold', family='sans-serif')
        ax_mismatch.set_ylabel('Probability Density', fontsize=14, fontweight='bold', family='sans-serif')
        ax_mismatch.set_title('Mismatch Trials', fontsize=14, fontweight='bold', family='sans-serif')
        ax_mismatch.legend(loc='upper right', frameon=True, fontsize=11, fancybox=True, shadow=True)
        # APA style: no grid, remove top and right spines
        ax_mismatch.yaxis.grid(False)
        ax_mismatch.xaxis.grid(False)
        despine(ax_mismatch)

        plt.tight_layout()

        # Save normalized histogram (only if save_plots is True)
        if save_plots:
            hist_path = save_path.replace('.png', '_normalized_histogram.png')
            hist_pdf_path = save_path.replace('.png', '_normalized_histogram.pdf')
            fig_hist.savefig(hist_pdf_path, dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none', format='pdf')
            fig_hist.savefig(hist_path, dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
            print(f"Normalized histogram saved to: {hist_pdf_path}")
            print(f"Normalized histogram (PNG) saved to: {hist_path}")

        plt.close(fig_hist)

    # Return full data if requested
    if return_full_data:
        results.update({
            'all_trials': all_trials,
            'model_match_times': model_match_times,
            'model_mismatch_times': model_mismatch_times,
            'human_match_times': human_match_times,
            'human_mismatch_times': human_mismatch_times,
            'model_match_times_ms': model_match_times_ms if can_denormalize else None,
            'model_mismatch_times_ms': model_mismatch_times_ms if can_denormalize else None,
            'human_match_times_ms': human_match_times_ms if can_denormalize else None,
            'human_mismatch_times_ms': human_mismatch_times_ms if can_denormalize else None
        })

    return results

# ==================== MAIN FUNCTION ====================
def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='Train Encoder-RNN Cognitive Model v3.1')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to human dataset CSV file',
                        default='Encoder_RNN_Cognitive-model/human_dataset.csv')
    parser.add_argument('--test_data_path', type=str, default=None,
                        help='Path to separate test dataset CSV file. If not provided, 80/20 split is used.')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save model and figures')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--use_rt_loss', action='store_true',
                        help='Use RT supervision during training')
    parser.add_argument('--speed_penalty', type=float, default=0.0,
                        help='Penalty for slow decisions (0=no penalty, 0.1-1.0=favor faster, higher=stronger)')
    parser.add_argument('--accuracy_threshold', type=float, default=0.7,
                        help='Minimum accuracy threshold for saving best model (default: 0.7)')
    parser.add_argument('--noise_position', type=str, default='evidence',
                        choices=['input', 'evidence', 'both', 'none'],
                        help='Where to inject noise')
    parser.add_argument('--evidence_noise_std', type=float, default=0.5,
                        help='Evidence Gaussian noise std')
    parser.add_argument('--evidence_mask_p', type=float, default=0.4,
                        help='Evidence dropout probability')
    parser.add_argument('--evidence_dropout_rescale', action='store_true',
                        help='Rescale after evidence dropout')
    parser.add_argument('--time_steps', type=int, default=20,
                        help='Number of time steps for decision')
    parser.add_argument('--threshold', type=float, default=6.0,
                        help='Evidence accumulation threshold for decision (default: 6.0, matches Tanh-bounded evidence [-1,1])')
    parser.add_argument('--evidence_scale', type=float, default=1.0,
                        help='Scale factor for evidence values (default: 1.0)')
    parser.add_argument('--pretrained_encoder_path', type=str, default=None,
                        help='Path to pretrained encoder weights from contrastive pre-training, use "None" or omit for training from scratch')
    parser.add_argument('--freeze_encoder', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether to freeze encoder weights during training (default: False)')
    parser.add_argument('--share_encoder_backbone', action='store_true',
                        help='Share encoder backbone for both images')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use (auto=automatic detection, cuda=NVIDIA GPU, mps=Apple Silicon, cpu=CPU)')
    parser.add_argument('--condition_name', type=str, default=None,
                        choices=['baseline', 'cl_only', 'noise_only', None],
                        help='Ablation study condition name for visualization labels (default: None = FS-Net)')
    parser.add_argument('--show_incorrect', action='store_true',
                        help='Show correct/incorrect breakdown in ablation study visualizations (default: False)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (0=single process)')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for faster GPU transfer')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--save_epoch', type=int, default=None,
                        help='Specific epoch to save outputs for (e.g., 50). If None, saves for all epochs meeting accuracy threshold.')
    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to save plot files (PNG/PDF). Set to False to save only CSV files (default: True)')
    parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to save model files (.pth). Set to False to save only CSV files (default: True)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device with better detection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu_id}')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                device = torch.device('cpu')
            else:
                device = torch.device(f'cuda:{args.gpu_id}')
        elif args.device == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available. Falling back to CPU.")
                device = torch.device('cpu')
            else:
                device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # Enable cuDNN benchmarking for better performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Set random seed for reproducibility
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("\n" + "="*60)
    print("Encoder-RNN Cognitive Model v3.1")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Path: {args.data_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Random Seed: {args.random_seed}")
    print(f"  Save Epoch: {args.save_epoch if args.save_epoch else 'All epochs meeting threshold'}")
    print(f"  RT Supervision: {args.use_rt_loss}")
    print(f"  Noise Position: {args.noise_position}")
    print(f"  Evidence Noise Std: {args.evidence_noise_std}")
    print(f"  Evidence Mask P: {args.evidence_mask_p}")
    print(f"  Evidence Dropout Rescale: {args.evidence_dropout_rescale}")
    print(f"  Time Steps: {args.time_steps}")
    print(f"  Device: {device}")

    # Create dataset
    print("\nCreating datasets...")
    generator = ShapeGenerator(square_side=22, matrix_size=64)

    # Training dataset from --data_path
    train_dataset = HumanAlignedDataset(args.data_path, generator)

    # Test dataset from --test_data_path (if provided), otherwise same as train
    if args.test_data_path:
        test_dataset = HumanAlignedDataset(args.test_data_path, generator)
        test_dataset_full = test_dataset  # Keep reference to full dataset for denormalization
        print(f"  Using separate test set: {args.test_data_path}")
    else:
        # Fallback: split train set (original behavior)
        full_dataset = train_dataset  # Keep reference before splitting
        total_len = len(train_dataset)
        train_size = int(0.8 * total_len)
        test_size = total_len - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        test_dataset_full = full_dataset  # Use full dataset for denormalization
        print(f"  Splitting train set: {train_size} train, {test_size} test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == 'cuda'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == 'cuda'
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    
    print("\nCreating model...")
    pretrained_path = args.pretrained_encoder_path
    if pretrained_path in [None, "None", "none", ""]:
        pretrained_path = None

    # Create model
    print("\nCreating model...")
    if pretrained_path:
        print(f"  Loading pretrained encoder from: {pretrained_path}")
        print(f"  Freeze encoder: {args.freeze_encoder}")
    else:
        print("  Training encoder from scratch")

    noise_pos = None if args.noise_position == 'none' else args.noise_position
    model = EncoderRTifyModel(
        pretrained_encoder_path= pretrained_path,
        freeze_encoder=args.freeze_encoder,
        share_encoder_backbone=args.share_encoder_backbone,
        noise_position=noise_pos,
        mask_p=0.0,
        gaussian_std=0.0,
        evidence_noise_std=args.evidence_noise_std,
        evidence_mask_p=args.evidence_mask_p,
        evidence_dropout_rescale=args.evidence_dropout_rescale,
        time_steps=args.time_steps,
        threshold=args.threshold,
        evidence_scale=args.evidence_scale
    )

    # Generate filename with hyperparameters
    rt_sup = "rt_sup" if args.use_rt_loss else "no_rt_sup"
    noise_pos = args.noise_position if args.noise_position != 'none' else 'no_noise'
    rescale = "rescale" if args.evidence_dropout_rescale else "no_rescale"
    encoder_type = "pretrained_frozen" if args.pretrained_encoder_path and args.freeze_encoder else \
                   "pretrained_finetune" if args.pretrained_encoder_path else "scratch"
    speed_str = f"_speed{args.speed_penalty}" if args.speed_penalty > 0 else ""
    filename = (f"{encoder_type}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_"
                f"{noise_pos}_std{args.evidence_noise_std}_"
                f"mask{args.evidence_mask_p}_{rescale}_"
                f"t{args.time_steps}_{rt_sup}{speed_str}")

    # Train model
    model_save_path = os.path.join(args.output_dir, f'{filename}.pth')
    history, best_epoch_info, all_best_epochs, accuracy_threshold_epochs, all_threshold_results = train_model(
        model, train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_rt_loss=args.use_rt_loss,
        speed_penalty=args.speed_penalty,
        save_path=model_save_path,
        accuracy_threshold=args.accuracy_threshold,
        save_best=True,
        test_dataset=test_dataset_full,  # ← 添加这个参数!传入test_dataset以支持denormalization
        output_dir=args.output_dir,
        filename=filename,
        time_steps=args.time_steps,
        condition_name=args.condition_name,
        show_incorrect=args.show_incorrect,
        save_epoch=args.save_epoch,
        random_seed=args.random_seed,
        save_plots=args.save_plots,
        save_model=args.save_model
    )

    # Skip final filtering when save_epoch is specified
    # User only wants the specified epoch's output, no additional processing needed
    if args.save_epoch is not None:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Outputs saved for epoch {args.save_epoch}")
        print(f"Skipping final filtering (save_epoch specified)")
        print("="*60)
        return

    # Final filtering: Find the best epoch among all that met accuracy threshold
    # Use the collected results instead of re-evaluating
    print("\n" + "="*60)
    print("FINAL FILTERING: BEST EPOCH SELECTION")
    print("="*60)
    print(f"Found {len(all_threshold_results)} epochs that met accuracy threshold")
    
    if all_threshold_results:
        # Filter for significant fast-same effect
        significant_epochs = [ep for ep in all_threshold_results if ep['has_significant_fast_same']]
        
        if significant_epochs:
            print(f"Found {len(significant_epochs)} epochs with significant fast-same effect")
            
            # Find the one with largest effect size
            best_significant_epoch = max(significant_epochs, key=lambda x: x['fast_same_effect_size'])
            
            print(f"\n★ BEST EPOCH (Significant + Largest Effect Size):")
            print(f"  Epoch: {best_significant_epoch['epoch']}")
            print(f"  Accuracy: {best_significant_epoch['accuracy']*100:.2f}%")
            print(f"  Effect Size: {best_significant_epoch['fast_same_effect_size']:.4f}")
            print(f"  P-value: {best_significant_epoch['p_value']:.4f}")
            print(f"  Files: {best_significant_epoch['filename']}")
            
            # Save the best model from significant epochs
            best_significant_path = model_save_path.replace('.pth', '_BEST_SIGNIFICANT_ep{}.pth'.format(best_significant_epoch['epoch']))
            torch.save(best_significant_epoch['model_state'], best_significant_path)
            print(f"  Model saved: {best_significant_path}")

            # Create a summary CSV of all significant epochs (only if save_epoch is None)
            # When save_epoch is specified, user only wants that specific epoch's CSV
            if args.save_epoch is None:
                summary_data = []
                for ep in significant_epochs:
                    summary_data.append({
                        'Epoch': ep['epoch'],
                        'Accuracy': ep['accuracy'],
                        'Effect_Size': ep['fast_same_effect_size'],
                        'P_Value': ep['p_value'],
                        'Match_Mean': ep['match_mean'],
                        'Mismatch_Mean': ep['mismatch_mean'],
                        'Correlation': ep['correlation'],
                        'Filename': ep['filename']
                    })

                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(args.output_dir, 'significant_epochs_summary.csv')
                summary_df.to_csv(summary_path, index=False)
                print(f"  Summary saved: {summary_path}")
            else:
                print(f"  Skipping summary CSV (save_epoch={args.save_epoch} specified)")
            
            # Generate visualizations for all significant epochs using saved data
            print(f"\n" + "="*60)
            print(f"Generating Final Visualizations for {len(significant_epochs)} SIGNIFICANT Epochs")
            print("(Using pre-generated data, no re-evaluation needed)")
            print("="*60)
            
            for idx, epoch_snap in enumerate(significant_epochs):
                ep_num = epoch_snap['epoch']
                print(f"\n[{idx+1}/{len(significant_epochs)}] Processing Epoch {ep_num}...")
                print(f"  Effect: {epoch_snap['fast_same_effect_size']:.4f}, p={epoch_snap['p_value']:.4f} ✓ SIGNIFICANT, Acc={epoch_snap['accuracy']*100:.2f}%")
                print(f"  Files already generated: {epoch_snap['filename']}")
                
        else:
            print(f"No epochs with significant fast-same effect found")
            if all_threshold_results:
                best_non_significant = max(all_threshold_results, key=lambda x: x['fast_same_effect_size'])
                print(f"Best non-significant epoch: {best_non_significant['epoch']} (effect: {best_non_significant['fast_same_effect_size']:.4f})")
    
    else:
        print(f"\nNo epochs met the accuracy threshold of {args.accuracy_threshold*100:.1f}%")
    
    # Also generate visualizations for original best epochs (if any)
    if len(all_best_epochs) > 0:
        print("\n" + "="*60)
        print(f"Additional Visualizations for {len(all_best_epochs)} BEST Epochs")
        print("(Epochs with significant fast-same effect during training)")
        print("="*60)
        
        for idx, epoch_snap in enumerate(all_best_epochs):
            ep_num = epoch_snap['epoch']
            print(f"\n[{idx+1}/{len(all_best_epochs)}] Processing Epoch {ep_num}...")
            print(f"  Reason: {epoch_snap['reason']}")
            print(f"  Effect: {epoch_snap['fast_same_effect_size']:.4f}, p={epoch_snap['p_value']:.4f} ✓ SIGNIFICANT, Acc={epoch_snap['accuracy']*100:.2f}%")
            
            # Create epoch-specific filenames
            epoch_prefix = f"{filename}_ep{ep_num:02d}_BEST"
            epoch_fig_path = os.path.join(args.output_dir, f"{epoch_prefix}.png")
            epoch_model_path = os.path.join(args.output_dir, f"{epoch_prefix}.pth")
            
            # Save model
            torch.save({
                'model_state_dict': epoch_snap['model_state'],
                'epoch': ep_num,
                'accuracy': epoch_snap['accuracy'],
                'fast_same_effect_size': epoch_snap['fast_same_effect_size'],
                'p_value': epoch_snap['p_value'],
                'match_mean': epoch_snap['match_mean'],
                'mismatch_mean': epoch_snap['mismatch_mean'],
                'has_significant_fast_same': epoch_snap['has_significant_fast_same'],
                'meets_accuracy': epoch_snap['meets_accuracy'],
                'ttest_statistic': epoch_snap['ttest_statistic']
            }, epoch_model_path)
            print(f"  ✓ Model saved: {os.path.basename(epoch_model_path)}")
            
            # Load model and generate visualization
            model.load_state_dict(epoch_snap['model_state'])
            model.eval()
            
            # Generate visualization
            epoch_results = analyze_and_plot_decision_time(
                model, test_loader,
                test_dataset=test_dataset_full,
                device=device,
                save_path=epoch_fig_path,
                time_steps=args.time_steps,
                accuracy_threshold=args.accuracy_threshold,
                model_save_path=epoch_model_path,
                full_model=model,
                condition_name=args.condition_name,
                show_incorrect=args.show_incorrect
            )
            
            print(f"  ✓ Visualization and CSV exported")
        
        print("\n" + "="*60)
        print(f"Completed! Generated outputs for {len(all_best_epochs)} epochs")
        print("="*60)
    
    # Also evaluate final model for reference (optional)
    print("\n" + "="*60)
    print("Final Epoch Model Evaluation (for reference)")
    print("="*60)
    figure_save_path = os.path.join(args.output_dir, f'{filename}_final.png')
    
    # Load final model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    results = analyze_and_plot_decision_time(
        model, test_loader,
        train_loader=train_loader,  # Add training data
        train_dataset=train_dataset,  # Add training dataset
        test_dataset=test_dataset_full,  # Pass full dataset for denormalization
        device=device,
        save_path=figure_save_path,
        time_steps=args.time_steps,
        accuracy_threshold=args.accuracy_threshold,
        model_save_path=model_save_path,
        condition_name=args.condition_name,  # Add condition name
        full_model=model,
        show_incorrect=args.show_incorrect  # Show correct/incorrect breakdown if requested
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_save_path}")
    print(f"Figure saved to: {figure_save_path}")
    print(f"\nFinal Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Final RT Correlation: {results['correlation']:.4f}")


if __name__ == '__main__':
    main()