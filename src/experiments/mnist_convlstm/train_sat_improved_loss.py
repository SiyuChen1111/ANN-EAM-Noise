"""
SAT Model Training with Improved Loss Design

Key improvements:
1. Condition-specific loss weights
2. Different speed penalty for Speed/Accuracy conditions
3. Threshold differentiation regularization
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.models.convlstm_sat import RTify_ConvLSTM_SAT
from src.data.preprocess_mnist_behavioral_log import MNISTBehavioralDatasetLog
from torch.utils.data import DataLoader


# ============ Hyperparameters ============
W_LABEL_SPEED = 1.0
W_LABEL_ACC = 2.0
W_RT_SPEED = 3.0
W_RT_ACC = 1.0

SP_SPEED = 0.2
SP_ACC = -0.05

TH_DIFF_TARGET = 2.0
TH_DIFF_WEIGHT = 0.1


def compute_condition_specific_loss(model, images, labels, rt, response, sat_conditions, device):
    """
    Compute condition-specific loss.
    
    Args:
        model: RTify_ConvLSTM_SAT model
        images: input images [B, C, H, W]
        labels: correct labels [B]
        rt: normalized RT [B]
        response: human response [B]
        sat_conditions: list of SAT condition strings
        device: torch device
    
    Returns:
        total_loss: combined loss
        loss_dict: dictionary of individual losses
    """
    B = len(labels)
    
    decision_logits, rt_pred, confidence = model(images, sat_condition=sat_conditions)
    
    speed_mask = torch.tensor(['speed' in str(s).lower() for s in sat_conditions], device=device)
    acc_mask = ~speed_mask
    
    n_speed = speed_mask.sum().item()
    n_acc = acc_mask.sum().item()
    
    label_criterion = nn.CrossEntropyLoss(reduction='none')
    label_loss_all = label_criterion(decision_logits, response)
    
    label_loss_speed = label_loss_all[speed_mask].mean() if n_speed > 0 else torch.tensor(0.0, device=device)
    label_loss_acc = label_loss_all[acc_mask].mean() if n_acc > 0 else torch.tensor(0.0, device=device)
    
    label_loss = W_LABEL_SPEED * label_loss_speed + W_LABEL_ACC * label_loss_acc
    
    rt_criterion = nn.MSELoss(reduction='none')
    rt_loss_all = rt_criterion(rt_pred.squeeze(), rt)
    
    rt_loss_speed = rt_loss_all[speed_mask].mean() if n_speed > 0 else torch.tensor(0.0, device=device)
    rt_loss_acc = rt_loss_all[acc_mask].mean() if n_acc > 0 else torch.tensor(0.0, device=device)
    
    rt_loss = W_RT_SPEED * rt_loss_speed + W_RT_ACC * rt_loss_acc
    
    speed_penalties = []
    for i, sat in enumerate(sat_conditions):
        if 'accuracy' in str(sat).lower():
            speed_penalties.append(SP_ACC * rt_pred[i])
        else:
            speed_penalties.append(SP_SPEED * rt_pred[i])
    speed_penalty_loss = torch.stack(speed_penalties).mean()
    
    th_diff = model.threshold_accuracy - model.threshold_speed
    th_diff_loss = torch.relu(TH_DIFF_TARGET - th_diff)
    
    total_loss = (
        label_loss + 
        rt_loss + 
        speed_penalty_loss + 
        TH_DIFF_WEIGHT * th_diff_loss
    )
    
    return total_loss, {
        'label_loss_speed': label_loss_speed.item(),
        'label_loss_acc': label_loss_acc.item(),
        'rt_loss_speed': rt_loss_speed.item(),
        'rt_loss_acc': rt_loss_acc.item(),
        'speed_penalty_loss': speed_penalty_loss.item(),
        'th_diff_loss': th_diff_loss.item(),
        'th_speed': model.threshold_speed.item(),
        'th_acc': model.threshold_accuracy.item(),
    }


def evaluate_model(model, test_loader, device, dataset):
    """
    Evaluate model performance with condition-specific metrics.
    """
    model.eval()
    
    results = {
        'speed': {'rt_pred': [], 'rt_human': [], 'correct': [], 'label': [], 'response': []},
        'accuracy': {'rt_pred': [], 'rt_human': [], 'correct': [], 'label': [], 'response': []},
    }
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)
            response = batch['response'].to(device)
            sat_conditions = batch.get('sat', None)
            
            if sat_conditions is None:
                sat_conditions = ['speed'] * len(labels)
            elif isinstance(sat_conditions, torch.Tensor):
                sat_conditions = sat_conditions.tolist()
            elif isinstance(sat_conditions, np.ndarray):
                sat_conditions = sat_conditions.tolist()
            
            decision_logits, rt_pred, confidence = model(images, sat_condition=sat_conditions)
            
            pred_labels = decision_logits.argmax(dim=-1)
            
            rt_pred_out = dataset.denormalize_rt(rt_pred.cpu())
            rt_pred_denorm = rt_pred_out.numpy() if hasattr(rt_pred_out, 'numpy') else rt_pred_out
            rt_human_out = dataset.denormalize_rt(rt)
            rt_human_denorm = rt_human_out.numpy() if hasattr(rt_human_out, 'numpy') else rt_human_out
            
            for i in range(len(labels)):
                sat = sat_conditions[i]
                sat_key = 'speed' if 'speed' in str(sat).lower() else 'accuracy'
                
                results[sat_key]['rt_pred'].append(rt_pred_denorm[i] if hasattr(rt_pred_denorm, '__getitem__') else rt_pred_denorm)
                results[sat_key]['rt_human'].append(rt_human_denorm[i] if hasattr(rt_human_denorm, '__getitem__') else rt_human_denorm)
                results[sat_key]['correct'].append((pred_labels[i] == response[i]).item())
                results[sat_key]['label'].append(labels[i].item())
                results[sat_key]['response'].append(response[i].item())
    
    metrics = {}
    
    for sat_key in ['speed', 'accuracy']:
        r = results[sat_key]
        if len(r['correct']) > 0:
            acc = np.mean(r['correct'])
            rt_pred_arr = np.array(r['rt_pred']).flatten()
            rt_human_arr = np.array(r['rt_human']).flatten()
            if len(rt_pred_arr) > 1 and np.std(rt_pred_arr) > 0 and np.std(rt_human_arr) > 0:
                corr = np.corrcoef(rt_pred_arr, rt_human_arr)[0, 1]
            else:
                corr = 0.0
            rt_pred_mean = np.mean(rt_pred_arr)
            rt_human_mean = np.mean(rt_human_arr)
        else:
            acc, corr, rt_pred_mean, rt_human_mean = 0, 0, 0, 0
        
        metrics[f'{sat_key}_acc'] = acc
        metrics[f'{sat_key}_rt_corr'] = corr if not np.isnan(corr) else 0.0
        metrics[f'{sat_key}_rt_pred'] = rt_pred_mean
        metrics[f'{sat_key}_rt_human'] = rt_human_mean
    
    metrics['rt_diff_pred'] = metrics['accuracy_rt_pred'] - metrics['speed_rt_pred']
    metrics['rt_diff_human'] = metrics['accuracy_rt_human'] - metrics['speed_rt_human']
    
    return metrics


def train_model(model, train_loader, test_loader, dataset, num_epochs, lr, device, output_dir):
    """
    Train model with improved loss design.
    """
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, 'training_log.txt'), 'w')
    
    print("\n" + "="*80)
    print("SAT Model Training with Improved Loss Design")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Initial Th_Speed: {model.threshold_speed.item():.4f}")
    print(f"Initial Th_Acc: {model.threshold_accuracy.item():.4f}")
    print("="*80, file=log_file)
    print(f"Device: {device}", file=log_file)
    print(f"Epochs: {num_epochs}", file=log_file)
    print(f"Learning rate: {lr}", file=log_file)
    print(f"W_LABEL_SPEED: {W_LABEL_SPEED}", file=log_file)
    print(f"W_LABEL_ACC: {W_LABEL_ACC}", file=log_file)
    print(f"W_RT_SPEED: {W_RT_SPEED}", file=log_file)
    print(f"W_RT_ACC: {W_RT_ACC}", file=log_file)
    print(f"SP_SPEED: {SP_SPEED}", file=log_file)
    print(f"SP_ACC: {SP_ACC}", file=log_file)
    print(f"TH_DIFF_TARGET: {TH_DIFF_TARGET}", file=log_file)
    print(f"TH_DIFF_WEIGHT: {TH_DIFF_WEIGHT}", file=log_file)
    print("="*80, file=log_file)
    
    header = f"{'Epoch':>6} | {'Sp_Acc':>7} | {'Acc_Acc':>7} | {'Sp_Corr':>8} | {'Acc_Corr':>8} | {'RT_Diff':>7} | {'Th_Sp':>6} | {'Th_Acc':>6}"
    print(header)
    print(header, file=log_file)
    print("-"*80)
    print("-"*80, file=log_file)
    
    best_corr = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_losses = {
            'label_loss_speed': [], 'label_loss_acc': [],
            'rt_loss_speed': [], 'rt_loss_acc': [],
            'speed_penalty_loss': [], 'th_diff_loss': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)
            response = batch['response'].to(device)
            sat_conditions = batch.get('sat', None)
            
            if sat_conditions is None:
                sat_conditions = ['speed'] * len(labels)
            elif isinstance(sat_conditions, torch.Tensor):
                sat_conditions = sat_conditions.tolist()
            elif isinstance(sat_conditions, np.ndarray):
                sat_conditions = sat_conditions.tolist()
            
            optimizer.zero_grad()
            
            total_loss, loss_dict = compute_condition_specific_loss(
                model, images, labels, rt, response, sat_conditions, device
            )
            
            total_loss.backward()
            optimizer.step()
            
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k].append(v)
        
        metrics = evaluate_model(model, test_loader, device, dataset)
        
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        epoch_line = (
            f"{epoch+1:>6} | "
            f"{metrics['speed_acc']*100:>6.1f}% | "
            f"{metrics['accuracy_acc']*100:>6.1f}% | "
            f"{metrics['speed_rt_corr']:>8.4f} | "
            f"{metrics['accuracy_rt_corr']:>8.4f} | "
            f"{metrics['rt_diff_pred']:>7.3f} | "
            f"{loss_dict['th_speed']:>6.2f} | "
            f"{loss_dict['th_acc']:>6.2f}"
        )
        print(epoch_line, flush=True)
        print(epoch_line, file=log_file)
        log_file.flush()
        
        avg_corr = (metrics['speed_rt_corr'] + metrics['accuracy_rt_corr']) / 2
        if avg_corr > best_corr:
            best_corr = avg_corr
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'losses': avg_losses,
            }, os.path.join(output_dir, 'best_model.pth'))
    
    print("-"*80)
    print(f"Best RT Corr: {best_corr:.4f} at Epoch {best_epoch}")
    print("-"*80, file=log_file)
    print(f"Best RT Corr: {best_corr:.4f} at Epoch {best_epoch}", file=log_file)
    
    log_file.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='SAT Model Training with Improved Loss')
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/raw/rtnet/behavioral data.csv')
    parser.add_argument('--output_dir', type=str, 
                        default='outputs/experiments/mnist_convlstm/exp_sat_improved_loss')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    dataset = MNISTBehavioralDatasetLog(
        args.data_path,
        mnist_root='/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/data/mnist-data',
        image_size=28,
        add_input_noise=True
    )
    
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    test_size = total_len - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = RTify_ConvLSTM_SAT(
        input_channel=config.get('input_channel', 1),
        num_filter=config.get('num_filter', 16),
        kernel_size=config.get('kernel_size', 3),
        output_size=config.get('output_size', 8),
        time_steps=config.get('time_steps', 40),
        sigma=config.get('sigma', 1.0),
        noise_position=config.get('noise_position', 'evidence'),
        evidence_noise_std=config.get('evidence_noise_std', 0.5),
        evidence_mask_p=config.get('evidence_mask_p', 0.4),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.threshold_speed.data.fill_(4.28)
    model.threshold_accuracy.data.fill_(4.28)
    
    print(f"Loaded from: {args.pretrained_path}")
    print(f"Initial Th_Speed: {model.threshold_speed.item():.4f}")
    print(f"Initial Th_Acc: {model.threshold_accuracy.item():.4f}")
    
    model = train_model(
        model, train_loader, test_loader, dataset,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
