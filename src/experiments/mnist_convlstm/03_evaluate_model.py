import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, 'scripts')
from preprocess_mnist_behavioral import MNISTBehavioralDataset
from train_mnist_convlstm import ConvLSTM, RTify_ConvLSTM, DiffDecision, add_noise

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

def train_model(model, train_loader, num_epochs=50, lr=1e-3, device='cpu', use_rt_loss=True):
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
            decision_logits, rt_pred = model(images)

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

            decision_logits, decision_time = model(images)
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
    OUTPUT_DIR = './output_mnist_convlstm'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print("\nLoading dataset...")
    full_dataset = MNISTBehavioralDataset(DATA_PATH, mnist_root='data/mnist-data', image_size=28)
    
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    test_size = total_len - train_size
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print("\nCreating model...")
    model = RTify_ConvLSTM(
        input_channel=1,
        num_filter=16,
        kernel_size=3,
        output_size=8,
        time_steps=20,
        sigma=2.0,
        noise_position='evidence',
        evidence_noise_std=0.1,
        evidence_mask_p=0.3,
        evidence_dropout_rescale=False
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial threshold: {model.threshold.item():.4f}")
    
    rt_loss, label_loss, acc, corr = train_model(
        model, train_loader, num_epochs=50, lr=1e-3, device=device, use_rt_loss=True
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
    print(f"Learned Threshold: {model.threshold.item():.4f}")
    
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
    
    model_path = os.path.join(OUTPUT_DIR, 'convlstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_accuracy': accuracy,
        'final_correlation': correlation,
        'final_threshold': model.threshold.item()
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
