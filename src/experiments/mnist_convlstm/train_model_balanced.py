"""
ConvLSTM Training Script for RTNet Task - Balanced Version

Balances accuracy and RT prediction with adjustable weights.
"""

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
import argparse
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.data.preprocess_mnist_behavioral_log import MNISTBehavioralDatasetLog

import importlib.util
spec = importlib.util.spec_from_file_location("train_module", 
    os.path.join(project_root, "src/experiments/mnist_convlstm/02_train_model.py"))
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

RTify_ConvLSTM = train_module.RTify_ConvLSTM
evaluate_model = train_module.evaluate_model
plot_training_curves = train_module.plot_training_curves
plot_rt_distribution = train_module.plot_rt_distribution

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass
sns.set_palette("husl")


def train_model_balanced(model, train_loader, num_epochs, lr, device, 
                         use_rt_loss=True, rt_loss_weight=2.0, speed_penalty=0.1,
                         output_dir='./output', filename='model',
                         learn_human_response=True):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    rt_loss_list = []
    label_loss_list = []
    acc_list = []
    corr_list = []
    
    rt_criterion = nn.MSELoss()
    label_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_rt_loss = []
        epoch_label_loss = []
        epoch_acc = []
        epoch_corr = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)
            response = batch['response'].to(device)
            
            optimizer.zero_grad()
            
            decision_logits, decision_time = model(images)
            
            if learn_human_response:
                target_label = response
            else:
                target_label = labels
            
            label_loss = label_criterion(decision_logits, target_label)
            
            rt_pred = decision_time
            rt_loss = rt_criterion(rt_pred, rt)
            
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
            
            epoch_rt_loss.append(rt_loss.item())
            epoch_label_loss.append(label_loss.item())
            
            acc_correct = (decision_logits.argmax(-1) == labels).float().mean().item()
            epoch_acc.append(acc_correct)
            
            rt_pred_np = rt_pred.detach().cpu().numpy().flatten()
            rt_np = rt.cpu().numpy().flatten()
            corr_temp = np.corrcoef(rt_pred_np, rt_np)[0, 1] if len(rt_pred_np) > 1 else 0.0
            epoch_corr.append(np.nan_to_num(corr_temp))
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'rt_w': f'{rt_loss_weight:.1f}',
                'acc_correct': f'{acc_correct:.3f}',
                'corr': f'{np.nan_to_num(corr_temp):.3f}'
            })
        
        rt_loss_list.append(np.mean(epoch_rt_loss))
        label_loss_list.append(np.mean(epoch_label_loss))
        acc_list.append(np.mean(epoch_acc))
        corr_list.append(np.mean(epoch_corr))
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'RT Loss = {np.mean(epoch_rt_loss):.4f}, '
              f'Label Loss = {np.mean(epoch_label_loss):.4f}, '
              f'Acc = {np.mean(epoch_acc)*100:.1f}%, '
              f'Corr = {np.mean(epoch_corr):.4f}, '
              f'Threshold = {model.threshold.item():.4f}')
    
    print("\nTraining complete!")
    
    return rt_loss_list, label_loss_list, acc_list, corr_list


def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM Model with Balanced Loss')
    parser.add_argument('--data_path', type=str, 
                        default='data/raw/rtnet/behavioral data.csv',
                        help='Path to behavioral data CSV file')
    parser.add_argument('--output_dir', type=str, default='./output_mnist_convlstm_balanced',
                        help='Directory to save model and figures')
    parser.add_argument('--epochs', type=int, default=70,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--use_rt_loss', action='store_true',
                        help='Use RT supervision during training')
    parser.add_argument('--rt_loss_weight', type=float, default=2.0,
                        help='Weight for RT loss (default: 2.0)')
    parser.add_argument('--speed_penalty', type=float, default=0.1,
                        help='Penalty for slow decisions (default: 0.1)')
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
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--fixed_noise', action='store_true',
                        help='Use fixed noise parameters instead of learnable ones')
    parser.add_argument('--learn_correct_label', action='store_true',
                        help='Learn correct label instead of human response')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    print("\n" + "="*60)
    print("ConvLSTM Model for MNIST RT Prediction (Balanced Loss)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Path: {args.data_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Time Steps: {args.time_steps}")
    print(f"  RT Supervision: {args.use_rt_loss}")
    print(f"  RT Loss Weight: {args.rt_loss_weight}")
    print(f"  Speed Penalty: {args.speed_penalty}")
    print(f"  Device: {device}")

    print("\nCreating datasets...")
    full_dataset = MNISTBehavioralDatasetLog(
        args.data_path, 
        mnist_root='data/mnist-data', 
        image_size=28
    )
    
    total_len = len(full_dataset)
    train_size = int((1 - args.test_split) * total_len)
    test_size = total_len - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.random_seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    print("\nCreating model...")
    learnable_noise = not args.fixed_noise
    
    model = RTify_ConvLSTM(
        input_channel=1,
        num_filter=args.num_filter,
        kernel_size=args.kernel_size,
        output_size=8,
        time_steps=args.time_steps,
        sigma=args.sigma,
        noise_position='evidence',
        evidence_noise_std=0.5,
        evidence_mask_p=0.4,
        evidence_dropout_rescale=False,
        learnable_noise=learnable_noise
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Initial threshold: {model.threshold.item():.4f}")

    rt_sup = "rt_sup_balanced" if args.use_rt_loss else "no_rt_sup"
    learn_human_response = not args.learn_correct_label
    resp_mode = "human_resp" if learn_human_response else "correct_label"
    filename = f"convlstm_balanced_rt{args.rt_loss_weight}_sp{args.speed_penalty}_ep{args.epochs}"

    rt_loss, label_loss, acc, corr = train_model_balanced(
        model, train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_rt_loss=args.use_rt_loss,
        rt_loss_weight=args.rt_loss_weight,
        speed_penalty=args.speed_penalty,
        output_dir=args.output_dir,
        filename=filename,
        learn_human_response=learn_human_response
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
    
    print(f"\nRT by Correctness (normalized):")
    print(f"  Correct trials: {results['correct_rt'].mean():.4f} +/- {results['correct_rt'].std():.4f}")
    if len(results['incorrect_rt']) > 0:
        print(f"  Incorrect trials: {results['incorrect_rt'].mean():.4f} +/- {results['incorrect_rt'].std():.4f}")

    rt_dist_path = os.path.join(args.output_dir, f'{filename}_rt_distribution.png')
    plot_rt_distribution(results, full_dataset, rt_dist_path)

    model_path = os.path.join(args.output_dir, f'{filename}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_channel': 1,
            'num_filter': args.num_filter,
            'kernel_size': args.kernel_size,
            'output_size': 8,
            'time_steps': args.time_steps,
            'sigma': args.sigma,
            'noise_position': 'evidence',
            'evidence_noise_std': 0.5,
            'evidence_mask_p': 0.4,
            'evidence_dropout_rescale': False,
            'learnable_noise': learnable_noise,
            'learn_human_response': learn_human_response,
            'log_normalization': True,
            'rt_loss_weight': args.rt_loss_weight,
            'speed_penalty': args.speed_penalty
        },
        'final_accuracy_correct': results['accuracy_correct'],
        'final_accuracy_response': results['accuracy_response'],
        'final_correlation': results['correlation'],
        'final_threshold': model.threshold.item(),
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    results_df = pd.DataFrame({
        'true_label': results['labels'],
        'pred_label': results['preds'],
        'human_response': results['responses'],
        'correct': results['correct'],
        'rt_pred_normalized': results['rt_pred'],
        'rt_human_normalized': results['rt_human'],
        'rt_pred_seconds': full_dataset.denormalize_rt(results['rt_pred']),
        'rt_human_seconds': full_dataset.denormalize_rt(results['rt_human'])
    })
    results_path = os.path.join(args.output_dir, f'{filename}_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
