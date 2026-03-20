"""
Training Script for ConvLSTM with SAT Thresholds

Based on exp11 parameters:
- No input noise (only internal evidence noise)
- SAT-conditioned thresholds (speed/accuracy)
- Confidence output

Key features:
- Separate thresholds for Speed/Accuracy conditions
- Internal evidence noise (our innovation)
- Confidence = max_prob - second_max_prob
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.data.preprocess_mnist_behavioral_log import MNISTBehavioralDatasetLog
from src.models.convlstm_sat import RTify_ConvLSTM_SAT


def evaluate_model(model, test_loader, device, dataset):
    """Evaluate model and return detailed results."""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_responses = []
    all_correct = []
    all_rt_pred = []
    all_rt_human = []
    all_confidence = []
    all_sat = []
    all_difficulty = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)
            response = batch['response'].to(device)
            sat_conditions = batch.get('sat', ['speed'] * len(labels))
            
            decision_logits, rt_pred, confidence = model(images, sat_condition=sat_conditions)
            
            pred_labels = decision_logits.argmax(dim=-1)
            correct = (pred_labels == labels).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_labels.cpu().numpy())
            all_responses.extend(response.cpu().numpy())
            all_correct.extend(correct)
            all_rt_pred.extend(rt_pred.cpu().numpy())
            all_rt_human.extend(rt.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())
            all_sat.extend(sat_conditions if isinstance(sat_conditions, list) else list(sat_conditions))
            all_difficulty.extend(batch.get('difficulty', ['unknown'] * len(labels)))
    
    results = {
        'labels': np.array(all_labels),
        'preds': np.array(all_preds),
        'responses': np.array(all_responses),
        'correct': np.array(all_correct),
        'rt_pred': np.array(all_rt_pred),
        'rt_human': np.array(all_rt_human),
        'confidence': np.array(all_confidence),
        'sat': all_sat,
        'difficulty': all_difficulty,
        'accuracy_correct': np.mean(all_labels == np.array(all_preds)),
        'accuracy_response': np.mean(all_responses == np.array(all_preds)),
        'correlation': np.corrcoef(all_rt_pred, all_rt_human)[0, 1] if len(all_rt_pred) > 1 else 0
    }
    
    return results


def train_model(model, train_loader, num_epochs, lr, device, use_rt_loss=True, 
                rt_loss_weight=2.0, speed_penalty=0.1, output_dir='./output', 
                filename='model', learn_human_response=True):
    """Train the model."""
    
    label_criterion = nn.CrossEntropyLoss()
    rt_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    rt_loss_list = []
    label_loss_list = []
    acc_list = []
    corr_list = []
    threshold_speed_list = []
    threshold_accuracy_list = []
    
    model.to(device)
    
    print(f"\nStarting Training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"RT Supervision: {'Yes' if use_rt_loss else 'No'}")
    print(f"Speed Penalty: {speed_penalty}")
    print(f"Learn Human Response: {'Yes' if learn_human_response else 'No (learn correct label)'}")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_rt_loss = []
        epoch_label_loss = []
        epoch_acc = []
        epoch_corr = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)
            response = batch['response'].to(device)
            sat_conditions = batch.get('sat', ['speed'] * len(labels))
            
            target = response if learn_human_response else labels
            
            optimizer.zero_grad()
            
            decision_logits, rt_pred, confidence = model(images, sat_condition=sat_conditions)
            
            label_loss = label_criterion(decision_logits, target)
            
            if use_rt_loss:
                rt_loss = rt_criterion(rt_pred, rt)
                speed_loss = speed_penalty * rt_pred.mean()
                total_loss = label_loss + rt_loss_weight * rt_loss + speed_loss
            else:
                total_loss = label_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_label_loss.append(label_loss.item())
            if use_rt_loss:
                epoch_rt_loss.append(rt_loss.item())
            
            pred_labels = decision_logits.argmax(dim=-1)
            correct = (pred_labels == labels).float().mean().item()
            epoch_acc.append(correct)
            
            if len(rt_pred) > 1:
                corr = np.corrcoef(rt_pred.detach().cpu().numpy(), 
                                   rt.detach().cpu().numpy())[0, 1]
                epoch_corr.append(corr)
            
            pbar.set_postfix({
                'label_loss': f'{label_loss.item():.4f}',
                'acc': f'{correct*100:.1f}%'
            })
        
        rt_loss_list.append(np.mean(epoch_rt_loss) if use_rt_loss else 0)
        label_loss_list.append(np.mean(epoch_label_loss))
        acc_list.append(np.mean(epoch_acc))
        corr_list.append(np.nan_to_num(np.mean(epoch_corr)))
        threshold_speed_list.append(model.threshold_speed.item())
        threshold_accuracy_list.append(model.threshold_accuracy.item())
        
        if use_rt_loss:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"RT Loss = {np.mean(epoch_rt_loss):.4f}, "
                  f"Label Loss = {np.mean(epoch_label_loss):.4f}, "
                  f"Acc = {np.mean(epoch_acc)*100:.1f}%, "
                  f"Corr = {np.nan_to_num(np.mean(epoch_corr)):.4f}, "
                  f"Th_Speed = {model.threshold_speed.item():.4f}, "
                  f"Th_Acc = {model.threshold_accuracy.item():.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Label Loss = {np.mean(epoch_label_loss):.4f}, "
                  f"Acc = {np.mean(epoch_acc)*100:.1f}%")
    
    return rt_loss_list, label_loss_list, acc_list, corr_list, threshold_speed_list, threshold_accuracy_list


def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM with SAT Thresholds')
    parser.add_argument('--data_path', type=str, 
                        default='data/raw/rtnet/behavioral data.csv',
                        help='Path to behavioral data CSV file')
    parser.add_argument('--output_dir', type=str, 
                        default='./outputs/experiments/mnist_convlstm/exp_sat',
                        help='Directory to save model and results')
    parser.add_argument('--epochs', type=int, default=70,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_filter', type=int, default=16,
                        help='Number of filters in ConvLSTM')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for ConvLSTM')
    parser.add_argument('--time_steps', type=int, default=40,
                        help='Number of time steps')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Sigma for soft indexing')
    parser.add_argument('--rt_loss_weight', type=float, default=2.0,
                        help='Weight for RT loss')
    parser.add_argument('--speed_penalty', type=float, default=0.1,
                        help='Speed penalty coefficient')
    parser.add_argument('--use_rt_loss', action='store_true', default=True,
                        help='Use RT supervision')
    parser.add_argument('--learn_correct_label', action='store_true',
                        help='Learn correct labels instead of human responses')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    dataset = MNISTBehavioralDatasetLog(
        args.data_path,
        mnist_root='/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/data/mnist-data',
        image_size=28,
        add_input_noise=False
    )
    
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    test_size = total_len - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    model = RTify_ConvLSTM_SAT(
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
        learnable_noise=False
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Initial threshold_speed: {model.threshold_speed.item():.4f}")
    print(f"  Initial threshold_accuracy: {model.threshold_accuracy.item():.4f}")
    
    rt_sup = "rt_sup" if args.use_rt_loss else "no_rt_sup"
    learn_human_response = not args.learn_correct_label
    resp_mode = "human_resp" if learn_human_response else "correct_label"
    filename = f"convlstm_sat_rt{args.rt_loss_weight}_sp{args.speed_penalty}_ep{args.epochs}"
    
    rt_loss, label_loss, acc, corr, th_speed, th_acc = train_model(
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
    
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    results = evaluate_model(model, test_loader, device, dataset)
    
    print(f"\nAccuracy (vs correct label): {results['accuracy_correct']*100:.2f}%")
    print(f"Accuracy (vs human response): {results['accuracy_response']*100:.2f}%")
    print(f"RT Correlation: {results['correlation']:.4f}")
    print(f"Learned Threshold Speed: {model.threshold_speed.item():.4f}")
    print(f"Learned Threshold Accuracy: {model.threshold_accuracy.item():.4f}")
    
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
            'learnable_noise': False,
            'learn_human_response': learn_human_response,
            'log_normalization': True,
            'rt_loss_weight': args.rt_loss_weight,
            'speed_penalty': args.speed_penalty
        },
        'final_accuracy_correct': results['accuracy_correct'],
        'final_accuracy_response': results['accuracy_response'],
        'final_correlation': results['correlation'],
        'final_threshold_speed': model.threshold_speed.item(),
        'final_threshold_accuracy': model.threshold_accuracy.item(),
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    results_df = pd.DataFrame({
        'true_label': results['labels'],
        'pred_label': results['preds'],
        'human_response': results['responses'],
        'correct': results['correct'],
        'rt_pred_normalized': results['rt_pred'],
        'rt_human_normalized': results['rt_human'],
        'rt_pred_seconds': dataset.denormalize_rt(results['rt_pred']),
        'rt_human_seconds': dataset.denormalize_rt(results['rt_human']),
        'confidence': results['confidence'],
        'sat': results['sat'],
        'difficulty': results['difficulty']
    })
    results_path = os.path.join(args.output_dir, f'{filename}_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
