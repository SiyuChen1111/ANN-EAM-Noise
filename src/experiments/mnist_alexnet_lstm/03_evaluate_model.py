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
import sys
from torchvision import models

sys.path.insert(0, 'scripts')
from preprocess_mnist_behavioral import MNISTBehavioralDataset

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

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

class PretrainedAlexNet(nn.Module):
    def __init__(self, feature_dim=4096, freeze_features=True):
        super().__init__()
        
        alexnet = models.alexnet(pretrained=True)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

class RTify_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size,
                 time_steps=20, sigma=2.0,
                 noise_position='evidence',
                 mask_p=0.0, gaussian_std=0.0,
                 evidence_noise_std=0.0, evidence_mask_p=0.0,
                 evidence_dropout_rescale=False,
                 threshold=6.0,
                 num_lstm_layers=1):

        super().__init__()
        self.evidence_dropout_rescale = evidence_dropout_rescale
        self.time_steps = time_steps
        self.noise_position = noise_position
        self.mask_p = mask_p
        self.gaussian_std = gaussian_std
        self.evidence_noise_std = evidence_noise_std
        self.evidence_mask_p = evidence_mask_p
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=False
        )

        self.fc = nn.Linear(hidden_dim, output_size)

        self.evidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.threshold = torch.nn.Parameter(torch.tensor(threshold))
        self.sigma = sigma

    def forward(self, x):
        B, input_dim = x.shape
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1)

        if self.noise_position in ['input', 'both']:
            x_seq = add_noise(x_seq, self.mask_p, self.gaussian_std)

        hidden_states, _ = self.lstm(x_seq)

        logit_trajectory = self.fc(hidden_states).permute(1, 0, 2)
        s_traj = self.evidence(hidden_states).squeeze(-1).permute(1, 0)

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

class AlexNetRTifyModel(nn.Module):
    def __init__(self, feature_dim=4096, hidden_dim=512, output_size=8,
                 time_steps=20, sigma=2.0, freeze_encoder=True,
                 noise_position='evidence',
                 mask_p=0.0, gaussian_std=0.0,
                 evidence_noise_std=0.0, evidence_mask_p=0.0,
                 evidence_dropout_rescale=False,
                 threshold=6.0, num_lstm_layers=1):
        super().__init__()
        self.freeze_encoder = freeze_encoder

        self.encoder = PretrainedAlexNet(feature_dim=feature_dim, freeze_features=freeze_encoder)

        self.rtify = RTify_LSTM(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_size=output_size,
            time_steps=time_steps,
            sigma=sigma,
            noise_position=noise_position,
            mask_p=mask_p,
            gaussian_std=gaussian_std,
            evidence_noise_std=evidence_noise_std,
            evidence_mask_p=evidence_mask_p,
            evidence_dropout_rescale=evidence_dropout_rescale,
            threshold=threshold,
            num_lstm_layers=num_lstm_layers
        )

    def forward(self, image):
        z = self.encoder(image)
        decision_logits, decision_time = self.rtify(z)
        return decision_logits, decision_time, z

def train_model(model, train_loader, num_epochs=50, lr=1e-4, device='cpu', use_rt_loss=True):
    label_criterion = nn.CrossEntropyLoss()
    rt_criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    rt_loss_list, label_loss_list, acc_list, corr_list = [], [], [], []
    model.to(device)

    print(f"\nStarting Training...")
    print(f"Device: {device}, Epochs: {num_epochs}, RT Supervision: {use_rt_loss}")
    print("="*60)

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)

            optimizer.zero_grad()
            decision_logits, rt_pred, _ = model(images)

            rt_loss = rt_criterion(rt_pred, rt)
            label_loss = label_criterion(decision_logits, labels)

            total_loss = rt_loss + label_loss if use_rt_loss else label_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            rt_loss_list.append(rt_loss.item())
            label_loss_list.append(label_loss.item())
            acc = (decision_logits.argmax(-1) == labels).float().mean().item()
            acc_list.append(acc)

            rt_pred_np = rt_pred.detach().cpu().numpy().flatten()
            rt_np = rt.cpu().numpy().flatten()
            corr_temp = np.corrcoef(rt_pred_np, rt_np)[0, 1] if len(rt_pred_np) > 1 else 0.0
            corr_list.append(np.nan_to_num(corr_temp))

            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'acc': f'{acc:.3f}'})

    print("\nTraining complete!")
    return rt_loss_list, label_loss_list, acc_list, corr_list

def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)
    
    all_rt_pred, all_rt_human, all_labels, all_preds, all_correct = [], [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt_human = batch['rt_normalized'].to(device)
            correct = batch['correct'].to(device)

            decision_logits, decision_time, _ = model(images)
            pred_labels = decision_logits.argmax(dim=-1)

            all_rt_pred.extend(decision_time.cpu().numpy())
            all_rt_human.extend(rt_human.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_labels.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
    
    return {
        'rt_pred': np.array(all_rt_pred),
        'rt_human': np.array(all_rt_human),
        'labels': np.array(all_labels),
        'preds': np.array(all_preds),
        'correct': np.array(all_correct)
    }

def plot_training_curves_apa(rt_loss, label_loss, acc, corr, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].plot(rt_loss, color='#1f77b4', linewidth=1.5)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('RT Loss (MSE)')
    axes[0, 0].set_title('A. RT Loss')
    
    axes[0, 1].plot(label_loss, color='#ff7f0e', linewidth=1.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Label Loss (Cross-Entropy)')
    axes[0, 1].set_title('B. Label Loss')
    
    axes[1, 0].plot(acc, color='#2ca02c', linewidth=1.5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('C. Accuracy')
    axes[1, 0].set_ylim(0, 1.05)
    
    axes[1, 1].plot(corr, color='#d62728', linewidth=1.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('RT Correlation')
    axes[1, 1].set_title('D. RT Correlation')
    axes[1, 1].set_ylim(-1, 1)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

def plot_rt_distribution_apa(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    rt_pred = results['rt_pred']
    rt_human = results['rt_human']
    labels = results['labels']
    correct = results['correct']
    correlation = np.corrcoef(rt_pred, rt_human)[0, 1]
    
    axes[0].scatter(rt_human, rt_pred, alpha=0.3, s=8, color='#1f77b4')
    z = np.polyfit(rt_human, rt_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(rt_human.min(), rt_human.max(), 100)
    axes[0].plot(x_line, p(x_line), color='#d62728', linewidth=2, label=f'Linear fit')
    axes[0].set_xlabel('Human RT (normalized)')
    axes[0].set_ylabel('Model RT (normalized)')
    axes[0].set_title(f'A. Model vs. Human RT\nr = {correlation:.3f}')
    axes[0].legend(frameon=False)
    
    correct_rt = rt_pred[correct == 1]
    incorrect_rt = rt_pred[correct == 0]
    
    bins = np.linspace(0, 1, 25)
    axes[1].hist(correct_rt, bins=bins, alpha=0.7, label=f'Correct (n={len(correct_rt)})', 
                 color='#2ca02c', density=True, edgecolor='white', linewidth=0.5)
    if len(incorrect_rt) > 0:
        axes[1].hist(incorrect_rt, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect_rt)})', 
                     color='#d62728', density=True, edgecolor='white', linewidth=0.5)
    axes[1].set_xlabel('RT (normalized)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('B. RT Distribution by Accuracy')
    axes[1].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RT distribution saved to: {save_path}")

def plot_rt_by_stimulus_apa(results, save_path):
    rt_pred = results['rt_pred']
    rt_human = results['rt_human']
    labels = results['labels']
    
    unique_labels = np.unique(labels)
    n_stim = len(unique_labels)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    for i, stim in enumerate(unique_labels):
        mask = labels == stim
        stim_rt_pred = rt_pred[mask]
        stim_rt_human = rt_human[mask]
        
        if len(stim_rt_pred) > 1:
            corr = np.corrcoef(stim_rt_pred, stim_rt_human)[0, 1]
        else:
            corr = 0
        
        axes[i].scatter(stim_rt_human, stim_rt_pred, alpha=0.3, s=6, color='#1f77b4')
        axes[i].set_xlabel('Human RT')
        axes[i].set_ylabel('Model RT')
        axes[i].set_title(f'Stimulus {stim+1} (n={len(stim_rt_pred)})\nr = {corr:.3f}')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    for i in range(n_stim, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RT by stimulus saved to: {save_path}")

def main():
    DATA_PATH = 'data/RTNet_Dataset/behavioral data.csv'
    OUTPUT_DIR = './output_mnist_alexnet'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print("\nLoading dataset...")
    full_dataset = MNISTBehavioralDataset(DATA_PATH, mnist_root='data/mnist-data', image_size=227)
    
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    test_size = total_len - train_size
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print("\nCreating model...")
    model = AlexNetRTifyModel(
        feature_dim=4096,
        hidden_dim=512,
        output_size=8,
        time_steps=20,
        sigma=2.0,
        freeze_encoder=True,
        noise_position='evidence',
        evidence_noise_std=0.1,
        evidence_mask_p=0.3,
        evidence_dropout_rescale=False,
        threshold=6.0,
        num_lstm_layers=1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Initial threshold: {model.rtify.threshold.item():.4f}")
    
    rt_loss, label_loss, acc, corr = train_model(
        model, train_loader, num_epochs=50, lr=1e-4, device=device, use_rt_loss=True
    )
    
    training_curve_path = os.path.join(OUTPUT_DIR, 'training_curves_apa.png')
    plot_training_curves_apa(rt_loss, label_loss, acc, corr, training_curve_path)
    
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    results = evaluate_model(model, test_loader, device)
    
    accuracy = np.mean(results['preds'] == results['labels'])
    correlation = np.corrcoef(results['rt_pred'], results['rt_human'])[0, 1]
    
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"RT Correlation: {correlation:.4f}")
    print(f"Learned Threshold: {model.rtify.threshold.item():.4f}")
    
    correct_rt = results['rt_pred'][results['correct'] == 1]
    incorrect_rt = results['rt_pred'][results['correct'] == 0]
    print(f"\nRT by Correctness (normalized):")
    print(f"  Correct: {correct_rt.mean():.4f} +/- {correct_rt.std():.4f} (n={len(correct_rt)})")
    if len(incorrect_rt) > 0:
        print(f"  Incorrect: {incorrect_rt.mean():.4f} +/- {incorrect_rt.std():.4f} (n={len(incorrect_rt)})")
    
    rt_dist_path = os.path.join(OUTPUT_DIR, 'rt_distribution_apa.png')
    plot_rt_distribution_apa(results, rt_dist_path)
    
    rt_stim_path = os.path.join(OUTPUT_DIR, 'rt_by_stimulus_apa.png')
    plot_rt_by_stimulus_apa(results, rt_stim_path)
    
    model_path = os.path.join(OUTPUT_DIR, 'alexnet_lstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_accuracy': accuracy,
        'final_correlation': correlation,
        'final_threshold': model.rtify.threshold.item()
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    results_df = pd.DataFrame({
        'true_label': results['labels'],
        'pred_label': results['preds'],
        'correct': results['correct'],
        'rt_pred_normalized': results['rt_pred'],
        'rt_human_normalized': results['rt_human'],
        'rt_pred_seconds': full_dataset.denormalize_rt(results['rt_pred']),
        'rt_human_seconds': full_dataset.denormalize_rt(results['rt_human'])
    })
    results_path = os.path.join(OUTPUT_DIR, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
