"""
改进版训练脚本 - 包含多种RT分布匹配策略
"""
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
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.data.preprocess_mnist_behavioral import MNISTBehavioralDataset

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Calibri'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

APA_COLORS = {
    'blue': '#4472C4',
    'orange': '#ED7D31',
    'gray': '#A5A5A5',
    'gold': '#FFC000',
    'light_blue': '#5B9BD5',
    'green': '#70AD47',
    'red': '#C00000',
    'purple': '#7030A0',
}

def add_noise(x, mask_p=0.0, std=0.0, rescale_after_dropout=True):
    if mask_p == 0 and std == 0:
        return x
    x_noisy = x.clone()
    if mask_p > 0:
        mask = torch.bernoulli(torch.ones_like(x) * (1 - mask_p))
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
    def __init__(self, input_channels=1, num_filters=16, kernel_size=3, 
                 output_size=8, time_steps=20, sigma=2.0,
                 noise_position='evidence', mask_p=0.0, gaussian_std=0.0,
                 evidence_noise_std=0.0, evidence_mask_p=0.0,
                 evidence_dropout_rescale=False,
                 threshold=6.0, learnable_noise=True,
                 min_noise_std=0.0):
        super().__init__()
        self.evidence_dropout_rescale = evidence_dropout_rescale
        self.time_steps = time_steps
        self.noise_position = noise_position
        self.learnable_noise = learnable_noise
        self.min_noise_std = min_noise_std
        
        self.conv_lstm = ConvLSTM(
            input_channel=input_channels,
            num_filter=num_filters,
            kernel_size=kernel_size
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters, output_size)
        self.evidence = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1),
            nn.Tanh()
        )
        
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.sigma = sigma
        
        if learnable_noise:
            self._noise_std_raw = nn.Parameter(torch.tensor(0.1).log())
            self._mask_p_raw = nn.Parameter(torch.tensor(0.3).logit())
        else:
            self._fixed_noise_std = evidence_noise_std
            self._fixed_mask_p = evidence_mask_p
    
    @property
    def noise_std(self):
        if self.learnable_noise:
            return torch.clamp(F.softplus(self._noise_std_raw), min=self.min_noise_std)
        return self._fixed_noise_std
    
    @property
    def mask_p(self):
        if self.learnable_noise:
            return torch.sigmoid(self._mask_p_raw)
        return self._fixed_mask_p
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        
        hidden_states, _ = self.conv_lstm(x_seq)
        
        time_steps, B, num_filter, H, W = hidden_states.shape
        hidden_2d = hidden_states.view(time_steps * B, num_filter, H, W)
        pooled_2d = self.pool(hidden_2d).squeeze()
        hidden_states = pooled_2d.view(time_steps, B, num_filter)
        
        logit_trajectory = self.fc(hidden_states).squeeze().permute(1, 0, 2)
        s_traj = self.evidence(hidden_states).squeeze(-1).permute(1, 0)
        
        if self.noise_position in ['evidence', 'both']:
            s_traj = add_noise(
                s_traj, self.mask_p, self.noise_std,
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


def rt_distribution_loss(rt_pred, rt_human, n_bins=20):
    """KL散度损失，使模型RT分布接近人类RT分布"""
    pred_hist = torch.histc(rt_pred, bins=n_bins, min=0, max=1)
    human_hist = torch.histc(rt_human, bins=n_bins, min=0, max=1)
    
    pred_hist = pred_hist / (pred_hist.sum() + 1e-8) + 1e-8
    human_hist = human_hist / (human_hist.sum() + 1e-8) + 1e-8
    
    kl_div = (human_hist * (human_hist.log() - pred_hist.log())).sum()
    return kl_div


def skewness_penalty(rt_pred, target_skewness=1.0):
    """鼓励RT分布具有正偏度（右偏）"""
    mean = rt_pred.mean()
    std = rt_pred.std() + 1e-8
    skewness = ((rt_pred - mean) ** 3).mean() / (std ** 3)
    return (skewness - target_skewness) ** 2


def compute_skewness(tensor):
    """计算张量的偏度"""
    mean = tensor.mean()
    std = tensor.std() + 1e-8
    return ((tensor - mean) ** 3).mean() / (std ** 3)


def train_model(model, train_loader, test_loader, num_epochs, lr, device, 
                use_rt_loss=True, rt_dist_weight=0.0, skewness_weight=0.0,
                output_dir='.', learn_human_response=True):
    
    label_criterion = nn.CrossEntropyLoss()
    rt_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_rt_corr': [],
        'test_loss': [], 'test_acc': [], 'test_rt_corr': [],
        'noise_std': [], 'mask_p': [], 'threshold': [],
        'rt_skewness': []
    }
    
    model.to(device)
    
    print(f"\nStarting Training...")
    print(f"Device: {device}, Epochs: {num_epochs}")
    print(f"RT Distribution Loss Weight: {rt_dist_weight}")
    print(f"Skewness Penalty Weight: {skewness_weight}")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_rt_corr = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            response = batch['response'].to(device)
            rt = batch['rt_normalized'].to(device)
            
            target = response if learn_human_response else labels
            
            optimizer.zero_grad()
            decision_logits, rt_pred = model(images)
            
            label_loss = label_criterion(decision_logits, target)
            rt_loss = rt_criterion(rt_pred, rt) if use_rt_loss else torch.tensor(0.0)
            
            total_loss = label_loss + rt_loss
            
            if rt_dist_weight > 0:
                dist_loss = rt_distribution_loss(rt_pred, rt)
                total_loss = total_loss + rt_dist_weight * dist_loss
            
            if skewness_weight > 0:
                skew_loss = skewness_penalty(rt_pred, target_skewness=1.0)
                total_loss = total_loss + skewness_weight * skew_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            acc = (decision_logits.argmax(-1) == target).float().mean().item()
            epoch_acc += acc
            
            rt_pred_np = rt_pred.detach().cpu().numpy().flatten()
            rt_np = rt.cpu().numpy().flatten()
            if len(rt_pred_np) > 1:
                corr = np.corrcoef(rt_pred_np, rt_np)[0, 1]
                epoch_rt_corr += np.nan_to_num(corr)
            
            n_batches += 1
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'acc': f'{acc:.3f}',
                'rt_corr': f'{epoch_rt_corr/n_batches:.3f}'
            })
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['train_acc'].append(epoch_acc / n_batches)
        history['train_rt_corr'].append(epoch_rt_corr / n_batches)
        
        if model.learnable_noise:
            history['noise_std'].append(model.noise_std.item())
            history['mask_p'].append(model.mask_p.item())
        history['threshold'].append(model.threshold.item())
        
        model.eval()
        all_rt_pred, all_rt_human = [], []
        test_loss, test_acc, test_rt_corr = 0, 0, 0
        n_test = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                response = batch['response'].to(device)
                rt = batch['rt_normalized'].to(device)
                
                target = response if learn_human_response else labels
                
                decision_logits, rt_pred = model(images)
                
                loss = label_criterion(decision_logits, target)
                test_loss += loss.item()
                
                acc = (decision_logits.argmax(-1) == target).float().mean().item()
                test_acc += acc
                
                all_rt_pred.extend(rt_pred.cpu().numpy())
                all_rt_human.extend(rt.cpu().numpy())
                
                n_test += 1
        
        all_rt_pred = np.array(all_rt_pred)
        all_rt_human = np.array(all_rt_human)
        
        if len(all_rt_pred) > 1:
            test_rt_corr = np.corrcoef(all_rt_pred.flatten(), all_rt_human.flatten())[0, 1]
            history['rt_skewness'].append(compute_skewness(torch.from_numpy(all_rt_pred)).item())
        
        history['test_loss'].append(test_loss / n_test)
        history['test_acc'].append(test_acc / n_test)
        history['test_rt_corr'].append(np.nan_to_num(test_rt_corr))
        
        print(f"Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, "
              f"Train Acc={history['train_acc'][-1]:.3f}, "
              f"Test Acc={history['test_acc'][-1]:.3f}, "
              f"RT Corr={history['test_rt_corr'][-1]:.3f}, "
              f"RT Skew={history['rt_skewness'][-1]:.3f}")
    
    return model, history


def plot_results(history, results_df, output_dir, experiment_name):
    """生成APA风格可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], color=APA_COLORS['blue'], linewidth=1.5, label='Train')
    ax1.plot(history['test_loss'], color=APA_COLORS['orange'], linewidth=1.5, label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('A. Training Loss', fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = axes[0, 1]
    ax2.plot(history['train_acc'], color=APA_COLORS['green'], linewidth=1.5, label='Train')
    ax2.plot(history['test_acc'], color=APA_COLORS['red'], linewidth=1.5, label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('B. Accuracy', fontweight='bold', loc='left')
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax3 = axes[1, 0]
    ax3.plot(history['test_rt_corr'], color=APA_COLORS['purple'], linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('RT Correlation')
    ax3.set_title('C. RT Correlation', fontweight='bold', loc='left')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    ax4 = axes[1, 1]
    ax4.plot(history['rt_skewness'], color=APA_COLORS['light_blue'], linewidth=1.5)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.axhline(y=1.0, color=APA_COLORS['green'], linestyle=':', linewidth=0.5, alpha=0.5, label='Target (Human)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('RT Skewness')
    ax4.set_title('D. RT Distribution Skewness', fontweight='bold', loc='left')
    ax4.legend(frameon=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_training_curves.png'), dpi=300)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    ax1 = axes[0]
    rt_pred = results_df['rt_pred_normalized'].values
    rt_human = results_df['rt_human_normalized'].values
    
    ax1.scatter(rt_human, rt_pred, alpha=0.3, s=8, color=APA_COLORS['blue'], edgecolors='none')
    
    mask = ~np.isnan(rt_human) & ~np.isnan(rt_pred)
    if mask.sum() > 1:
        z = np.polyfit(rt_human[mask], rt_pred[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(rt_human[mask].min(), rt_human[mask].max(), 100)
        ax1.plot(x_line, p(x_line), color=APA_COLORS['red'], linewidth=2)
        
        corr = np.corrcoef(rt_human[mask], rt_pred[mask])[0, 1]
        ax1.set_title(f'A. Model vs. Human RT\nr = {corr:.3f}', fontweight='bold', loc='left')
    
    ax1.set_xlabel('Human RT (normalized)')
    ax1.set_ylabel('Model RT (normalized)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
    
    ax2 = axes[1]
    correct_rt = results_df[results_df['correct'] == True]['rt_pred_normalized'].values
    incorrect_rt = results_df[results_df['correct'] == False]['rt_pred_normalized'].values
    
    bins = np.linspace(0, 1, 25)
    ax2.hist(correct_rt, bins=bins, alpha=0.7, label=f'Correct (n={len(correct_rt):,})', 
             color=APA_COLORS['green'], density=True, edgecolor='white', linewidth=0.5)
    if len(incorrect_rt) > 0:
        ax2.hist(incorrect_rt, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect_rt):,})', 
                 color=APA_COLORS['red'], density=True, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('RT (normalized)')
    ax2.set_ylabel('Density')
    ax2.set_title('B. RT Distribution by Accuracy', fontweight='bold', loc='left')
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_rt_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Figures saved to {output_dir}")


def evaluate_model(model, test_loader, device, dataset):
    """评估模型并返回结果DataFrame"""
    model.eval()
    model.to(device)
    
    all_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            response = batch['response'].to(device)
            correct = batch['correct'].to(device)
            rt_human = batch['rt_normalized'].to(device)
            
            decision_logits, rt_pred = model(images)
            pred_labels = decision_logits.argmax(dim=-1)
            
            for i in range(len(labels)):
                all_results.append({
                    'true_label': labels[i].item(),
                    'pred_label': pred_labels[i].item(),
                    'human_response': response[i].item(),
                    'correct': correct[i].item(),
                    'rt_pred_normalized': rt_pred[i].item(),
                    'rt_human_normalized': rt_human[i].item(),
                })
    
    df = pd.DataFrame(all_results)
    return df


def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM with RT distribution matching')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_rt_loss', action='store_true')
    parser.add_argument('--rt_dist_weight', type=float, default=0.0, help='Weight for RT distribution matching loss')
    parser.add_argument('--skewness_weight', type=float, default=0.0, help='Weight for skewness penalty')
    parser.add_argument('--fixed_noise', action='store_true', help='Use fixed noise parameters')
    parser.add_argument('--min_noise_std', type=float, default=0.0, help='Minimum noise std for learnable noise')
    parser.add_argument('--initial_threshold', type=float, default=6.0, help='Initial threshold value')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("="*60)
    print("ConvLSTM RT Distribution Matching Experiment")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"RT Distribution Loss Weight: {args.rt_dist_weight}")
    print(f"Skewness Penalty Weight: {args.skewness_weight}")
    print(f"Fixed Noise: {args.fixed_noise}")
    print(f"Min Noise Std: {args.min_noise_std}")
    print(f"Initial Threshold: {args.initial_threshold}")
    print("="*60)
    
    print("\nLoading dataset...")
    full_dataset = MNISTBehavioralDataset(args.data_path, mnist_root='data/mnist-data')
    
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    test_size = total_len - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print("\nCreating model...")
    model = RTify_ConvLSTM(
        input_channels=1,
        num_filters=16,
        kernel_size=3,
        output_size=8,
        time_steps=20,
        sigma=2.0,
        noise_position='evidence',
        evidence_noise_std=0.5,
        evidence_mask_p=0.4,
        threshold=args.initial_threshold,
        learnable_noise=not args.fixed_noise,
        min_noise_std=args.min_noise_std
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Learnable noise: {not args.fixed_noise}")
    
    model, history = train_model(
        model, train_loader, test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_rt_loss=args.use_rt_loss,
        rt_dist_weight=args.rt_dist_weight,
        skewness_weight=args.skewness_weight,
        output_dir=args.output_dir,
        learn_human_response=True
    )
    
    print("\nEvaluating...")
    results_df = evaluate_model(model, test_loader, device, full_dataset)
    
    accuracy = (results_df['pred_label'] == results_df['true_label']).mean()
    rt_corr = np.corrcoef(results_df['rt_pred_normalized'], results_df['rt_human_normalized'])[0, 1]
    rt_skew = compute_skewness(torch.from_numpy(results_df['rt_pred_normalized'].values)).item()
    
    print(f"\nFinal Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  RT Correlation: {rt_corr:.4f}")
    print(f"  RT Skewness: {rt_skew:.4f}")
    print(f"  Threshold: {model.threshold.item():.4f}")
    if model.learnable_noise:
        print(f"  Noise Std: {model.noise_std.item():.4f}")
        print(f"  Mask P: {model.mask_p.item():.4f}")
    
    results_df.to_csv(os.path.join(args.output_dir, f'{args.experiment_name}_results.csv'), index=False)
    
    plot_results(history, results_df, args.output_dir, args.experiment_name)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_accuracy': accuracy,
        'final_rt_corr': rt_corr,
        'final_rt_skewness': rt_skew,
    }, os.path.join(args.output_dir, f'{args.experiment_name}_model.pth'))
    
    print(f"\nResults saved to {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
