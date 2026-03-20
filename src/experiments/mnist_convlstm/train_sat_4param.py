"""
Stage 2 Fine-tuning with 4-parameter SAT model.

4-parameter SAT model:
- threshold_speed: learnable, starts from 4.28
- threshold_accuracy: learnable, starts from 4.28
- speed_penalty_speed: fixed (0.3) - encourages fast decisions in speed condition
- speed_penalty_accuracy: fixed (0.08) - allows more deliberation in accuracy condition

This script fine-tunes the network weights while keeping thresholds learnable
and speed_penalty values fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys

from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.models.convlstm_sat import RTify_ConvLSTM_SAT
from src.data.preprocess_mnist_behavioral_log import MNISTBehavioralDatasetLog


def train_4param_sat(model, train_loader, num_epochs, lr, device,
                     rt_loss_weight=2.0,
                     speed_penalty_speed=0.3,
                     speed_penalty_accuracy=0.08,
                     output_dir='./outputs'):
    """
    Train 4-param SAT model with fixed speed_penalty and learnable thresholds.

    Args:
        model: 4-param SAT model
        train_loader: Training data loader
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        rt_loss_weight: Weight for RT loss
        speed_penalty_speed: Fixed speed_penalty for speed condition
        speed_penalty_accuracy: Fixed speed_penalty for accuracy condition
        output_dir: Output directory for results
    """
    model.to(device)
    model.train()

    label_criterion = nn.CrossEntropyLoss()
    rt_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'rt_loss': [],
        'label_loss': [],
        'acc': [],
        'corr': [],
        'threshold_speed': [],
        'threshold_accuracy': [],
        'speed_penalty_speed': [],
        'speed_penalty_accuracy': []
    }

    log_file = open(output_dir + '/training_progress.log', 'w')

    print("\n" + "="*60)
    print("STAGE 2: 4-param SAT Fine-tuning")
    print("="*60, file=log_file)
    print(f"Device: {device}", file=log_file)
    print(f"Epochs: {num_epochs}", file=log_file)
    print(f"Learning rate: {lr}", file=log_file)
    print(f"RT loss weight: {rt_loss_weight}", file=log_file)
    print(f"speed_penalty_speed: {speed_penalty_speed} (fixed)", file=log_file)
    print(f"speed_penalty_accuracy: {speed_penalty_accuracy} (fixed)", file=log_file)
    print(f"threshold_speed: {model.threshold_speed.item():.4f} (learnable)", file=log_file)
    print(f"threshold_accuracy: {model.threshold_accuracy.item():.4f} (learnable)", file=log_file)
    print("="*60, file=log_file)
    print("| Epoch   | Acc     | Corr   | RT Loss | Th_Speed | Th_Acc |", file=log_file)
    print("|---------|---------|--------|---------|----------|--------|", file=log_file)

    for epoch in range(num_epochs):
        model.train()

        epoch_rt_loss = []
        epoch_label_loss = []
        epoch_acc = []
        epoch_corr = []

        total_batches = len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - {total_batches} batches")

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
            elif isinstance(sat_conditions, list):
                pass
            else:
                sat_conditions = [str(sat_conditions)] * len(labels)

            optimizer.zero_grad()

            decision_logits, rt_pred, confidence = model(images, sat_condition=sat_conditions)

            label_loss = label_criterion(decision_logits, response)
            rt_loss = rt_criterion(rt_pred, rt)

            # Apply SAT-conditional speed_penalty
            speed_losses = []
            for i, sat in enumerate(sat_conditions):
                if 'accuracy' in sat.lower():
                    speed_losses.append(speed_penalty_accuracy * rt_pred[i])
                else:
                    speed_losses.append(speed_penalty_speed * rt_pred[i])
            speed_loss = torch.stack(speed_losses).mean()

            total_loss = label_loss + rt_loss_weight * rt_loss + speed_loss

            total_loss.backward()

            # Zero gradients for threshold parameters (they should remain learnable but controlled)
            # Note: We don't zero threshold gradients, we let them learn
            # But we ensure no unexpected gradient issues

            optimizer.step()

            epoch_rt_loss.append(rt_loss.item())
            epoch_label_loss.append(label_loss.item())

            pred_labels = decision_logits.argmax(dim=-1)
            correct = (pred_labels == labels).float().mean().item()
            epoch_acc.append(correct)

            if len(rt_pred) > 1:
                corr = np.corrcoef(rt_pred.detach().cpu().numpy(),
                                   rt.detach().cpu().numpy())[0, 1]
                epoch_corr.append(corr)

        history['rt_loss'].append(np.mean(epoch_rt_loss))
        history['label_loss'].append(np.mean(epoch_label_loss))
        history['acc'].append(np.mean(epoch_acc))
        history['corr'].append(np.nan_to_num(np.mean(epoch_corr)))
        history['threshold_speed'].append(model.threshold_speed.item())
        history['threshold_accuracy'].append(model.threshold_accuracy.item())
        history['speed_penalty_speed'].append(speed_penalty_speed)
        history['speed_penalty_accuracy'].append(speed_penalty_accuracy)

        th_spd = model.threshold_speed.item()
        th_acc = model.threshold_accuracy.item()
        acc_pct = np.mean(epoch_acc) * 100
        corr_val = np.nan_to_num(np.mean(epoch_corr))
        rt_loss_val = np.mean(epoch_rt_loss)

        epoch_line = f"| Epoch {epoch+1:3d}/{num_epochs} | Acc: {acc_pct:5.1f}% | Corr: {corr_val:6.4f} | RT Loss: {rt_loss_val:.4f} | Th_Speed: {th_spd:.4f} | Th_Acc: {th_acc:.4f} |"
        print(epoch_line, flush=True)
        print(epoch_line, file=log_file)
        log_file.flush()

    
    log_file.close()
    return history


def evaluate_model_with_difficulty(model, test_loader, device, dataset):
    """
    Evaluate model performance with difficulty analysis.
    """
    model.eval()

    results = {
        'all': {'rt_pred': [], 'rt_human': [], 'correct': []},
        'easy': {'rt_pred': [], 'rt_human': [], 'correct': []},
        'difficult': {'rt_pred': [], 'rt_human': [], 'correct': []},
        'speed': {'rt_pred': [], 'rt_human': [], 'correct': []},
        'accuracy': {'rt_pred': [], 'rt_human': [], 'correct': []}
    }

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)
            response = batch['response'].to(device)
            sat_conditions = batch.get('sat', None)
            difficulty = batch['difficulty']

            if sat_conditions is None:
                sat_conditions = ['speed'] * len(labels)
            elif isinstance(sat_conditions, torch.Tensor):
                sat_conditions = sat_conditions.tolist()
            elif isinstance(sat_conditions, np.ndarray):
                sat_conditions = sat_conditions.tolist()
            elif isinstance(sat_conditions, list):
                pass
            else:
                sat_conditions = [str(sat_conditions)] * len(labels)

            decision_logits, rt_pred, confidence = model(images, sat_condition=sat_conditions)

            pred_labels = decision_logits.argmax(dim=-1)
            correct = (pred_labels == labels).float().cpu().numpy()

            rt_pred_denorm = dataset.denormalize_rt(rt_pred.cpu()).numpy()
            rt_human_denorm = dataset.denormalize_rt(rt).cpu().numpy()

            for i in range(len(labels)):
                results['all']['rt_pred'].append(rt_pred_denorm[i])
                results['all']['rt_human'].append(rt_human_denorm[i])
                results['all']['correct'].append(correct[i])

                diff = difficulty[i] if isinstance(difficulty, list) else difficulty[i].item()
                if diff in results:
                    results[diff]['rt_pred'].append(rt_pred_denorm[i])
                    results[diff]['rt_human'].append(rt_human_denorm[i])
                    results[diff]['correct'].append(correct[i])

                sat = sat_conditions[i]
                sat_key = 'speed' if 'speed' in sat.lower() else 'accuracy'
                if sat_key in results:
                    results[sat_key]['rt_pred'].append(rt_pred_denorm[i])
                    results[sat_key]['rt_human'].append(rt_human_denorm[i])
                    results[sat_key]['correct'].append(correct[i])

    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 2: 4-param SAT Fine-tuning')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to 4-param SAT model')
    parser.add_argument('--data_path', type=str,
                        default='data/raw/rtnet/behavioral data.csv',
                        help='Path to behavioral data CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/experiments/mnist_convlstm/exp_sat_4param',
                        help='Directory to save model and results')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--rt_loss_weight', type=float, default=2.0,
                        help='Weight for RT loss')
    parser.add_argument('--speed_penalty_speed', type=float, default=0.3,
                        help='Fixed speed_penalty for speed condition')
    parser.add_argument('--speed_penalty_accuracy', type=float, default=0.08,
                        help='Fixed speed_penalty for accuracy condition')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda/mps)')

    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("LOADING DATASET (with input noise)")
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
    print("LOADING 4-PARAM SAT MODEL")
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
        evidence_dropout_rescale=config.get('evidence_dropout_rescale', False),
        learnable_noise=config.get('learnable_noise', False)
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Set speed_penalty values from checkpoint
    speed_penalty_speed = checkpoint.get('speed_penalty_speed', args.speed_penalty_speed)
    speed_penalty_accuracy = checkpoint.get('speed_penalty_accuracy', args.speed_penalty_accuracy)
    model.speed_penalty_speed = speed_penalty_speed
    model.speed_penalty_accuracy = speed_penalty_accuracy

    print(f"Loaded from: {args.pretrained_path}")
    print(f"Original threshold: {checkpoint.get('original_threshold', 'N/A')}")
    print(f"Current Threshold Speed: {model.threshold_speed.item():.4f}")
    print(f"Current Threshold Accuracy: {model.threshold_accuracy.item():.4f}")
    print(f"speed_penalty_speed: {model.speed_penalty_speed} (fixed)")
    print(f"speed_penalty_accuracy: {model.speed_penalty_accuracy} (fixed)")

    history = train_4param_sat(
        model, train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        rt_loss_weight=args.rt_loss_weight,
        speed_penalty_speed=speed_penalty_speed,
        speed_penalty_accuracy=speed_penalty_accuracy,
        output_dir=args.output_dir
    )

    print("\n" + "="*60)
    print("EVALUATION: SAT Conditions")
    print("="*60)

    results = evaluate_model_with_difficulty(model, test_loader, device, dataset)

    print("\n--- Overall Performance ---")
    overall_acc = np.mean(results['all']['correct'])
    overall_corr = np.corrcoef(results['all']['rt_pred'], results['all']['rt_human'])[0, 1]
    print(f"Accuracy: {overall_acc*100:.2f}%")
    print(f"RT Correlation: {overall_corr:.4f}")

    print("\n--- Speed vs Accuracy ---")
    for sat_key in ['speed', 'accuracy']:
        acc = np.mean(results[sat_key]['correct']) * 100
        corr = np.corrcoef(results[sat_key]['rt_pred'], results[sat_key]['rt_human'])[0, 1]
        rt_pred_mean = np.mean(results[sat_key]['rt_pred'])
        rt_human_mean = np.mean(results[sat_key]['rt_human'])
        print(f"{sat_key}: Acc={acc:.1f}%, Corr={corr:.4f}, RT_pred={rt_pred_mean:.3f}s, RT_human={rt_human_mean:.3f}s")

    print("\n--- Easy vs Difficult ---")
    for diff_key in ['easy', 'difficult']:
        acc = np.mean(results[diff_key]['correct']) * 100
        corr = np.corrcoef(results[diff_key]['rt_pred'], results[diff_key]['rt_human'])[0, 1]
        rt_pred_mean = np.mean(results[diff_key]['rt_pred'])
        rt_human_mean = np.mean(results[diff_key]['rt_human'])
        print(f"{diff_key}: Acc={acc:.1f}%, Corr={corr:.4f}, RT_pred={rt_pred_mean:.3f}s, RT_human={rt_human_mean:.3f}s")

    print(f"\nFinal Threshold Speed: {model.threshold_speed.item():.4f}")
    print(f"Final Threshold Accuracy: {model.threshold_accuracy.item():.4f}")
    print(f"speed_penalty_speed: {model.speed_penalty_speed} (fixed)")
    print(f"speed_penalty_accuracy: {model.speed_penalty_accuracy} (fixed)")

    # Save model
    filename = f"convlstm_4param_sat_ep{args.epochs}_spd{args.speed_penalty_speed}_acc{args.speed_penalty_accuracy}"

    model_path = os.path.join(args.output_dir, f'{filename}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'stage1_path': args.pretrained_path,
        'final_accuracy': overall_acc,
        'final_correlation': overall_corr,
        'final_threshold_speed': model.threshold_speed.item(),
        'final_threshold_accuracy': model.threshold_accuracy.item(),
        'speed_penalty_speed': model.speed_penalty_speed,
        'speed_penalty_accuracy': model.speed_penalty_accuracy,
        'history': history,
        'transfer_type': '4param_sat'
    }, model_path)

    print(f"\nSaved model to: {model_path}")


if __name__ == '__main__':
    main()
