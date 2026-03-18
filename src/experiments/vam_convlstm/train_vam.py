"""
ConvLSTM Training Script for VAM Lost in Migration Task

Task: Flanker task variant - identify the direction of the center bird
Stimuli: Birds pointing in 4 directions (L/R/U/D) with flanker birds
Behavioral Data: Human responses and reaction times

Model Features:
- ConvLSTM-based architecture with differentiable decision function
- Evidence accumulation with learnable threshold
- Learnable noise parameters for better fitting to human RT distribution
- Learning from human responses (including errors)

Usage:
    python train_vam.py --data_dir <path> --epochs 50 --batch_size 64 --use_rt_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.data.preprocess_vam_data import VAMDataset

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass
sns.set_palette("husl")


def add_noise(x, mask_p=0.0, std=0.0, rescale_after_dropout=True):
    if mask_p == 0 and std == 0:
        return x

    x_noisy = x.clone()

    if mask_p > 0:
        mask = torch.bernoulli(
            torch.ones_like(x) * (1 - mask_p)
        )
        x_noisy = x_noisy * mask

        if rescale_after_dropout:
            x_noisy = x_noisy / (1 - mask_p + 1e-8)

    if std > 0:
        noise = torch.randn_like(x) * std
        x_noisy = x_noisy + noise

    return x_noisy


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


class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, kernel_size, stride=1, padding=1):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=input_channel + num_filter,
            out_channels=num_filter * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.Wci = nn.Parameter(torch.zeros(1, num_filter, 1, 1))
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, 1, 1))
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, 1, 1))

        self._input_channel = input_channel
        self._num_filter = num_filter

    def forward(self, inputs=None, states=None, seq_len=20):
        device = inputs.device
        B, _, H, W = inputs[0].shape

        if states is None:
            c = torch.zeros(
                (B, self._num_filter, H, W),
                dtype=torch.float, device=device
            )
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

        outputs = torch.stack(outputs, dim=0)
        return outputs, (h, c)


class RTify_ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_filter: int,
        kernel_size: int,
        output_size: int,
        time_steps: int = 20,
        sigma: float = 2.0,
        noise_position: str = 'evidence',
        evidence_noise_std: float = 0.0,
        evidence_mask_p: float = 0.0,
        evidence_dropout_rescale: bool = False,
        learnable_noise: bool = True,
        adapt_input_channel: int = None
    ):
        super(RTify_ConvLSTM, self).__init__()
        self.time_steps = time_steps
        self.noise_position = noise_position
        self.evidence_dropout_rescale = evidence_dropout_rescale
        self.learnable_noise = learnable_noise
        self.adapt_input_channel = adapt_input_channel
        
        if adapt_input_channel is not None and adapt_input_channel != input_channel:
            self.input_adapter = nn.Conv2d(adapt_input_channel, input_channel, kernel_size=1)
            self.effective_input_channel = adapt_input_channel
            print(f"Added input adapter: {adapt_input_channel} -> {input_channel} channels")
        else:
            self.input_adapter = None
            self.effective_input_channel = input_channel
        
        self.convlstm = ConvLSTM(
            input_channel=input_channel,
            num_filter=num_filter,
            kernel_size=kernel_size
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filter, output_size)

        self.evidence = nn.Sequential(
            nn.Linear(num_filter, num_filter),
            nn.ReLU(),
            nn.Linear(num_filter, 1),
            nn.Tanh()
        )

        self.threshold = torch.nn.Parameter(torch.tensor(6.0))
        self.sigma = sigma
        
        if learnable_noise:
            self._noise_std_raw = nn.Parameter(torch.tensor(0.1).log())
            self._mask_p_raw = nn.Parameter(torch.tensor(0.3).logit())
        else:
            self._noise_std_raw = None
            self._mask_p_raw = None
            self._fixed_noise_std = evidence_noise_std
            self._fixed_mask_p = evidence_mask_p
    
    @property
    def noise_std(self):
        if self.learnable_noise:
            return torch.nn.functional.softplus(self._noise_std_raw)
        return self._fixed_noise_std
    
    @property
    def mask_p(self):
        if self.learnable_noise:
            return torch.sigmoid(self._mask_p_raw)
        return self._fixed_mask_p
        
    def forward(self, x):
        device = x.device
        B, C, H, W = x.shape
        
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        hidden_states, (h, c) = self.convlstm(x_seq, seq_len=self.time_steps)

        time_steps, B, num_filter, H, W = hidden_states.shape
        hidden_2d = hidden_states.view(time_steps * B, num_filter, H, W)
        pooled_2d = self.pool(hidden_2d).squeeze()
        hidden_states = pooled_2d.view(time_steps, B, num_filter)

        logit_trajectory = self.fc(hidden_states).squeeze().permute(1, 0, 2)
        
        s_traj = self.evidence(hidden_states).squeeze(-1).permute(1, 0)
        
        if self.noise_position in ['evidence', 'both']:
            s_traj = add_noise(
                s_traj,
                self.mask_p,
                self.noise_std,
                rescale_after_dropout=self.evidence_dropout_rescale
            )
        
        s_accumulated = torch.cumsum(s_traj, dim=1)
        dsdt_trajectory = torch.diff(s_accumulated, dim=1)
        dsdt_trajectory = torch.cat((dsdt_trajectory[:, 0].unsqueeze(1), dsdt_trajectory), dim=1)
        decision_time = DiffDecision.apply(s_accumulated - self.threshold, dsdt_trajectory)
        
        soft_index = torch.exp(-0.5 * (decision_time.unsqueeze(1) - torch.arange(self.time_steps, device=device)) ** 2 / self.sigma ** 2)
        soft_index = soft_index / soft_index.sum(dim=-1, keepdim=True)
        decision_logits = (logit_trajectory * soft_index.unsqueeze(-1)).sum(dim=1)
                
        return decision_logits, (decision_time+1) / self.time_steps


def train_model(model, train_loader, num_epochs=5, lr=1e-3, device='cpu',
                use_rt_loss=False, rt_loss_weight=2.0, speed_penalty=0.0, output_dir='./output', filename='model',
                learn_human_response=True, pretrained_path=None, freeze_encoder=False):
    
    label_criterion = nn.CrossEntropyLoss()
    rt_criterion = nn.MSELoss()
    
    if pretrained_path is not None:
        print(f"\nLoading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        pretrained_dict_filtered = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if 'fc' in k and 'weight' in k:
                    print(f"  Skipping {k} (decision output layer)")
                    continue
                if v.shape == model_dict[k].shape:
                    pretrained_dict_filtered[k] = v
                else:
                    print(f"  Skipping {k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
            else:
                print(f"  Skipping {k} (not in model)")
        
        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict_filtered)}/{len(pretrained_dict)} parameters from pretrained model")
        
        if freeze_encoder:
            print("Freezing encoder layers (ConvLSTM)")
            for name, param in model.named_parameters():
                if 'convlstm' in name:
                    param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    rt_loss_list, label_loss_list, acc_list, corr_list = [], [], [], []

    model.to(device)

    print(f"\nStarting Training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"RT Supervision: {'Yes' if use_rt_loss else 'No'}")
    print(f"RT Loss Weight: {rt_loss_weight}")
    print(f"Speed Penalty: {speed_penalty}")
    print(f"Learn Human Response: {'Yes' if learn_human_response else 'No (learn correct label)'}")
    print(f"Pretrained: {'Yes' if pretrained_path else 'No'}")
    print(f"Freeze Encoder: {'Yes' if freeze_encoder else 'No'}")
    print("="*60)

    for epoch in range(num_epochs):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            response = batch['response'].to(device)
            rt = batch['rt_normalized'].to(device)
            correct = batch['correct'].to(device)

            target = response if learn_human_response else labels

            optimizer.zero_grad()
            decision_logits, rt_pred = model(images)

            rt_loss = rt_criterion(rt_pred, rt)
            label_loss = label_criterion(decision_logits, target)

            if speed_penalty > 0:
                speed_loss = speed_penalty * rt_pred.mean()
            else:
                speed_loss = 0.0

            if use_rt_loss:
                total_loss = label_loss + rt_loss_weight * rt_loss + speed_loss
            else:
                total_loss = label_loss + speed_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            rt_loss_list.append(rt_loss.item())
            label_loss_list.append(label_loss.item())
            acc_correct = (decision_logits.argmax(-1) == labels).float().mean().item()
            acc_response = (decision_logits.argmax(-1) == response).float().mean().item()
            acc_list.append(acc_correct)

            rt_pred_np = rt_pred.detach().cpu().numpy().flatten()
            rt_np = rt.cpu().numpy().flatten()
            corr_temp = np.corrcoef(rt_pred_np, rt_np)[0, 1] if len(rt_pred_np) > 1 else 0.0
            corr_list.append(np.nan_to_num(corr_temp))

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'rt_w': f'{rt_loss_weight:.1f}',
                'acc_correct': f'{acc_correct:.3f}',
                'acc_response': f'{acc_response:.3f}',
                'corr': f'{np.nan_to_num(corr_temp):.3f}'
            })

    print("\nTraining complete!")
    
    return rt_loss_list, label_loss_list, acc_list, corr_list


def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)
    
    all_rt_pred = []
    all_rt_human = []
    all_labels = []
    all_preds = []
    all_correct = []
    all_responses = []
    all_congruency = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            responses = batch['response'].to(device)
            rt_human = batch['rt_normalized'].to(device)
            correct = batch['correct'].to(device)
            congruency = batch['congruency'].to(device)

            decision_logits, decision_time = model(images)
            pred_labels = decision_logits.argmax(dim=-1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = pred_labels[i].item()
                human_response = responses[i].item()
                model_dt = decision_time[i].item()
                human_rt = rt_human[i].item()
                is_correct = correct[i].item()
                is_congruent = congruency[i].item()

                all_rt_pred.append(model_dt)
                all_rt_human.append(human_rt)
                all_labels.append(true_label)
                all_preds.append(pred_label)
                all_correct.append(is_correct)
                all_responses.append(human_response)
                all_congruency.append(is_congruent)
    
    all_rt_pred = np.array(all_rt_pred)
    all_rt_human = np.array(all_rt_human)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_correct = np.array(all_correct)
    all_responses = np.array(all_responses)
    all_congruency = np.array(all_congruency)
    
    accuracy_correct = np.mean(all_preds == all_labels)
    accuracy_response = np.mean(all_preds == all_responses)
    correlation = np.corrcoef(all_rt_pred, all_rt_human)[0, 1]
    
    correct_rt = all_rt_pred[all_correct == 1]
    incorrect_rt = all_rt_pred[all_correct == 0]
    
    congruent_rt = all_rt_pred[all_congruency == 0]
    incongruent_rt = all_rt_pred[all_congruency == 1]
    
    return {
        'accuracy_correct': accuracy_correct,
        'accuracy_response': accuracy_response,
        'correlation': correlation,
        'rt_pred': all_rt_pred,
        'rt_human': all_rt_human,
        'correct_rt': correct_rt,
        'incorrect_rt': incorrect_rt,
        'congruent_rt': congruent_rt,
        'incongruent_rt': incongruent_rt,
        'labels': all_labels,
        'preds': all_preds,
        'correct': all_correct,
        'responses': all_responses,
        'congruency': all_congruency
    }


def plot_training_curves(rt_loss, label_loss, acc, corr, save_path=None):
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax = ax.flatten()
    
    ax[0].plot(rt_loss, '-k')
    ax[0].set_title('RT loss')
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('loss')
    
    ax[1].plot(label_loss, '-k')
    ax[1].set_title('Label loss')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('loss')
    
    ax[2].plot(acc, '-k')
    ax[2].set_title('Accuracy')
    ax[2].set_xlabel('iteration')
    ax[2].set_ylabel('accuracy')
    
    ax[3].plot(corr, '-k')
    ax[3].set_ylim(-1, 1)
    ax[3].set_title('RT correlation')
    ax[3].set_xlabel('iteration')
    ax[3].set_ylabel('correlation')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.close()


def plot_rt_distribution(results, save_path=None):
    all_rt_pred = results['rt_pred']
    all_rt_human = results['rt_human']
    correct_rt = results['correct_rt']
    incorrect_rt = results['incorrect_rt']
    congruent_rt = results['congruent_rt']
    incongruent_rt = results['incongruent_rt']
    correlation = results['correlation']
    all_labels = results['labels']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].scatter(all_rt_human, all_rt_pred, alpha=0.3, s=10)
    axes[0, 0].set_xlabel('Human RT (normalized)')
    axes[0, 0].set_ylabel('Model RT (normalized)')
    axes[0, 0].set_title(f'Model vs Human RT (r={correlation:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    z = np.polyfit(all_rt_human, all_rt_pred, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(all_rt_human, p(all_rt_human), "r--", alpha=0.8, label='fit line')
    axes[0, 0].legend()
    
    axes[0, 1].hist(correct_rt, bins=30, alpha=0.6, label=f'Correct (n={len(correct_rt)})', density=True)
    if len(incorrect_rt) > 0:
        axes[0, 1].hist(incorrect_rt, bins=30, alpha=0.6, label=f'Incorrect (n={len(incorrect_rt)})', density=True)
    axes[0, 1].set_xlabel('RT (normalized)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('RT Distribution by Correctness')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(congruent_rt, bins=30, alpha=0.6, label=f'Congruent (n={len(congruent_rt)})', density=True)
    axes[1, 0].hist(incongruent_rt, bins=30, alpha=0.6, label=f'Incongruent (n={len(incongruent_rt)})', density=True)
    axes[1, 0].set_xlabel('RT (normalized)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('RT Distribution by Congruency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    direction_names = ['Left', 'Right', 'Up', 'Down']
    rt_by_direction = [all_rt_pred[all_labels == i] for i in range(4)]
    
    axes[1, 1].boxplot(rt_by_direction, labels=direction_names)
    axes[1, 1].set_xlabel('Direction')
    axes[1, 1].set_ylabel('RT (normalized)')
    axes[1, 1].set_title('RT by Target Direction')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"RT distribution plot saved to: {save_path}")
    
    plt.close()


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
    
    parser = argparse.ArgumentParser(description='Train ConvLSTM Model for VAM Lost in Migration')
    parser.add_argument('--data_dir', type=str, 
                        default='VAM_Lost-in-Migration',
                        help='Path to VAM data directory')
    parser.add_argument('--output_dir', type=str, default='./output_vam',
                        help='Directory to save model and figures')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--use_rt_loss', action='store_true',
                        help='Use RT supervision during training')
    parser.add_argument('--rt_loss_weight', type=float, default=2.0,
                        help='Weight for RT loss (default: 2.0)')
    parser.add_argument('--speed_penalty', type=float, default=0.0,
                        help='Penalty for slow decisions')
    parser.add_argument('--noise_position', type=str, default='evidence',
                        choices=['evidence', 'none'],
                        help='Where to inject noise')
    parser.add_argument('--evidence_noise_std', type=float, default=0.5,
                        help='Evidence Gaussian noise std')
    parser.add_argument('--evidence_mask_p', type=float, default=0.4,
                        help='Evidence dropout probability')
    parser.add_argument('--evidence_dropout_rescale', action='store_true',
                        help='Rescale after evidence dropout')
    parser.add_argument('--time_steps', type=int, default=20,
                        help='Number of time steps for decision')
    parser.add_argument('--num_filter', type=int, default=16,
                        help='Number of filters in ConvLSTM')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for ConvLSTM')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Sigma for soft decision weighting')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for faster GPU transfer')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of users for training')
    parser.add_argument('--max_trials_per_user', type=int, default=25000,
                        help='Maximum trials per user')
    parser.add_argument('--fixed_noise', action='store_true',
                        help='Use fixed noise parameters instead of learnable ones')
    parser.add_argument('--learn_correct_label', action='store_true',
                        help='Learn correct label instead of human response')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained model for fine-tuning')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder layers when fine-tuning')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size for stimuli')
    parser.add_argument('--users', type=int, nargs='*', default=None,
                        help='List of user IDs to use (default: all users)')
    parser.add_argument('--use_log_norm', action='store_true', default=True,
                        help='Use log normalization for RT (default: True)')
    parser.add_argument('--use_linear_norm', action='store_true',
                        help='Use linear normalization instead of log')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("\n" + "="*60)
    print("ConvLSTM Model for VAM Lost in Migration")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Dir: {args.data_dir}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Random Seed: {args.random_seed}")
    print(f"  RT Supervision: {args.use_rt_loss}")
    print(f"  Log Normalization: {not args.use_linear_norm}")
    print(f"  Num Filter: {args.num_filter}")
    print(f"  Kernel Size: {args.kernel_size}")
    print(f"  Time Steps: {args.time_steps}")
    print(f"  Sigma: {args.sigma}")
    print(f"  Fixed Noise: {args.fixed_noise}")
    print(f"  Device: {device}")
    print(f"  Pretrained Model: {args.pretrained_model}")
    print(f"  Freeze Encoder: {args.freeze_encoder}")

    print("\nCreating datasets...")
    use_log_norm = not args.use_linear_norm
    print(f"  Using log normalization: {use_log_norm}")
    
    train_dataset = VAMDataset(
        args.data_dir,
        users=args.users,
        max_trials_per_user=args.max_trials_per_user,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        split='train',
        use_log_norm=use_log_norm
    )
    
    test_dataset = VAMDataset(
        args.data_dir,
        users=args.users,
        max_trials_per_user=args.max_trials_per_user,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        split='test',
        use_log_norm=use_log_norm
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == 'cuda',
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == 'cuda',
        drop_last=True
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    print("\nCreating model...")
    noise_pos = None if args.noise_position == 'none' else args.noise_position
    learnable_noise = not args.fixed_noise
    
    if args.pretrained_model:
        print(f"\nWill fine-tune from: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location='cpu', weights_only=False)
        pretrained_config = checkpoint.get('config', {})
        pretrained_input_channel = pretrained_config.get('input_channel', 1)
        
        print(f"  Pretrained model input channels: {pretrained_input_channel}")
        print(f"  VAM input channels: 3")
        
        model = RTify_ConvLSTM(
            input_channel=pretrained_input_channel,
            num_filter=args.num_filter,
            kernel_size=args.kernel_size,
            output_size=4,
            time_steps=args.time_steps,
            sigma=args.sigma,
            noise_position=noise_pos,
            evidence_noise_std=args.evidence_noise_std,
            evidence_mask_p=args.evidence_mask_p,
            evidence_dropout_rescale=args.evidence_dropout_rescale,
            learnable_noise=learnable_noise,
            adapt_input_channel=3
        )
    else:
        model = RTify_ConvLSTM(
            input_channel=3,
            num_filter=args.num_filter,
            kernel_size=args.kernel_size,
            output_size=4,
            time_steps=args.time_steps,
            sigma=args.sigma,
            noise_position=noise_pos,
            evidence_noise_std=args.evidence_noise_std,
            evidence_mask_p=args.evidence_mask_p,
            evidence_dropout_rescale=args.evidence_dropout_rescale,
            learnable_noise=learnable_noise
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Initial threshold: {model.threshold.item():.4f}")
    print(f"  Learnable noise: {learnable_noise}")
    if learnable_noise:
        print(f"  Initial noise_std: {model.noise_std.item():.4f}")
        print(f"  Initial mask_p: {model.mask_p.item():.4f}")
    else:
        print(f"  Fixed noise_std: {model.noise_std:.4f}")
        print(f"  Fixed mask_p: {model.mask_p:.4f}")

    rt_sup = "rt_sup" if args.use_rt_loss else "no_rt_sup"
    learn_human_response = not args.learn_correct_label
    resp_mode = "human_resp" if learn_human_response else "correct_label"
    ft_mode = "finetune" if args.pretrained_model else "scratch"
    filename = f"convlstm_nf{args.num_filter}_ks{args.kernel_size}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_t{args.time_steps}_{rt_sup}_{resp_mode}_{ft_mode}"

    rt_loss, label_loss, acc, corr = train_model(
        model, train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_rt_loss=args.use_rt_loss,
        rt_loss_weight=args.rt_loss_weight,
        speed_penalty=args.speed_penalty,
        output_dir=args.output_dir,
        filename=filename,
        learn_human_response=learn_human_response,
        pretrained_path=args.pretrained_model,
        freeze_encoder=args.freeze_encoder
    )

    training_curve_path = os.path.join(args.output_dir, f'{filename}_training_curves.png')
    plot_training_curves(rt_loss, label_loss, acc, corr, training_curve_path)

    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    results = evaluate_model(model, test_loader, device)

    print(f"\nAccuracy (vs correct label): {results['accuracy_correct']*100:.2f}%")
    print(f"Accuracy (vs human response): {results['accuracy_response']*100:.2f}%")
    print(f"RT Correlation: {results['correlation']:.4f}")
    print(f"Learned Threshold: {model.threshold.item():.4f}")
    
    if learnable_noise:
        print(f"Learned noise_std: {model.noise_std.item():.4f}")
        print(f"Learned mask_p: {model.mask_p.item():.4f}")
    
    print(f"\nRT by Correctness (normalized):")
    print(f"  Correct trials: {results['correct_rt'].mean():.4f} +/- {results['correct_rt'].std():.4f} (n={len(results['correct_rt'])})")
    if len(results['incorrect_rt']) > 0:
        print(f"  Incorrect trials: {results['incorrect_rt'].mean():.4f} +/- {results['incorrect_rt'].std():.4f} (n={len(results['incorrect_rt'])})")
    
    print(f"\nRT by Congruency (normalized):")
    print(f"  Congruent trials: {results['congruent_rt'].mean():.4f} +/- {results['congruent_rt'].std():.4f} (n={len(results['congruent_rt'])})")
    print(f"  Incongruent trials: {results['incongruent_rt'].mean():.4f} +/- {results['incongruent_rt'].std():.4f} (n={len(results['incongruent_rt'])})")

    rt_dist_path = os.path.join(args.output_dir, f'{filename}_rt_distribution.png')
    plot_rt_distribution(results, rt_dist_path)

    model_path = os.path.join(args.output_dir, f'{filename}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_channel': 3,
            'num_filter': args.num_filter,
            'kernel_size': args.kernel_size,
            'output_size': 4,
            'time_steps': args.time_steps,
            'sigma': args.sigma,
            'noise_position': noise_pos,
            'evidence_noise_std': args.evidence_noise_std,
            'evidence_mask_p': args.evidence_mask_p,
            'evidence_dropout_rescale': args.evidence_dropout_rescale,
            'learnable_noise': learnable_noise,
            'learn_human_response': learn_human_response
        },
        'final_accuracy_correct': results['accuracy_correct'],
        'final_accuracy_response': results['accuracy_response'],
        'final_correlation': results['correlation'],
        'final_threshold': model.threshold.item(),
        'final_noise_std': model.noise_std.item() if learnable_noise else model.noise_std,
        'final_mask_p': model.mask_p.item() if learnable_noise else model.mask_p
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    results_df = pd.DataFrame({
        'true_label': results['labels'],
        'pred_label': results['preds'],
        'human_response': results['responses'],
        'correct': results['correct'],
        'congruency': results['congruency'],
        'rt_pred_normalized': results['rt_pred'],
        'rt_human_normalized': results['rt_human'],
        'rt_pred_seconds': test_dataset.denormalize_rt(results['rt_pred']),
        'rt_human_seconds': test_dataset.denormalize_rt(results['rt_human'])
    })
    results_path = os.path.join(args.output_dir, f'{filename}_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
