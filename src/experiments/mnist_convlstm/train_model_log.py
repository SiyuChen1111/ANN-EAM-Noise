"""
ConvLSTM Training Script for RTNet Task - Log Normalization Version

Uses log-scale normalized RT values for better matching human RT distribution.
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
train_model = train_module.train_model
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


def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM Model with Log-Normalized RT')
    parser.add_argument('--data_path', type=str, 
                        default='data/raw/rtnet/behavioral data.csv',
                        help='Path to behavioral data CSV file')
    parser.add_argument('--output_dir', type=str, default='./output_mnist_convlstm_log',
                        help='Directory to save model and figures')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--use_rt_loss', action='store_true',
                        help='Use RT supervision during training')
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
    print("ConvLSTM Model for MNIST RT Prediction (Log Normalization)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Path: {args.data_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Time Steps: {args.time_steps}")
    print(f"  RT Supervision: {args.use_rt_loss}")
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

    rt_sup = "rt_sup_log" if args.use_rt_loss else "no_rt_sup"
    learn_human_response = not args.learn_correct_label
    resp_mode = "human_resp" if learn_human_response else "correct_label"
    filename = f"convlstm_log_t{args.time_steps}_{rt_sup}_{resp_mode}"

    rt_loss, label_loss, acc, corr = train_model(
        model, train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_rt_loss=args.use_rt_loss,
        speed_penalty=0.0,
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
            'log_normalization': True
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
