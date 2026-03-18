"""
Re-evaluate model and save results with difficulty information.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

project_root = '/Users/siyu/Documents/GitHub/ANN-EAM-Nosie'
sys.path.insert(0, project_root)

from src.data.preprocess_mnist_behavioral_log import MNISTBehavioralDatasetLog

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    from src.experiments.mnist_convlstm.train_model_balanced import RTify_ConvLSTM
    
    model = RTify_ConvLSTM(
        input_channel=config['input_channel'],
        num_filter=config['num_filter'],
        kernel_size=config['kernel_size'],
        output_size=config['output_size'],
        time_steps=config['time_steps'],
        sigma=config['sigma'],
        noise_position=config['noise_position'],
        evidence_noise_std=config['evidence_noise_std'],
        evidence_mask_p=config['evidence_mask_p'],
        evidence_dropout_rescale=config['evidence_dropout_rescale'],
        learnable_noise=config['learnable_noise']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, checkpoint

def evaluate_with_difficulty(model, test_loader, device, dataset):
    model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            response = batch['response'].to(device)
            correct = batch['correct'].to(device)
            rt_human = batch['rt_normalized'].to(device)
            difficulty = batch['difficulty']
            
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
                    'rt_pred_seconds': dataset.denormalize_rt(rt_pred[i].item()),
                    'rt_human_seconds': dataset.denormalize_rt(rt_human[i].item()),
                    'difficulty': difficulty[i]
                })
    
    return pd.DataFrame(all_results)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MNISTBehavioralDatasetLog(
        '/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/data/raw/rtnet/behavioral data.csv',
        mnist_root='/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/data/mnist-data',
        image_size=28
    )
    
    # Split dataset
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    test_size = total_len - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    model_path = '/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm/exp11_t40/convlstm_balanced_rt2.0_sp0.1_ep70.pth'
    print(f"\nLoading model from: {model_path}")
    model, config, checkpoint = load_model(model_path, device)
    
    print(f"Model loaded. Final accuracy: {checkpoint['final_accuracy_correct']*100:.2f}%")
    print(f"Final RT ratio: {checkpoint['final_threshold']:.4f}")
    
    # Evaluate
    print("\nEvaluating model with difficulty information...")
    results_df = evaluate_with_difficulty(model, test_loader, device, dataset)
    
    # Save results
    output_path = '/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm/exp11_t40/convlstm_balanced_rt2.0_sp0.1_ep70_results_with_difficulty.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Analyze by difficulty
    print("\n" + "="*60)
    print("Performance by Difficulty")
    print("="*60)
    
    for diff in ['easy', 'difficult']:
        diff_df = results_df[results_df['difficulty'] == diff]
        if len(diff_df) > 0:
            model_acc = (diff_df['pred_label'] == diff_df['true_label']).mean() * 100
            human_acc = diff_df['correct'].mean() * 100
            model_rt = diff_df['rt_pred_seconds'].mean()
            human_rt = diff_df['rt_human_seconds'].mean()
            rt_ratio = model_rt / human_rt
            
            print(f"\n{diff.upper()} ({len(diff_df)} trials):")
            print(f"  Model Accuracy: {model_acc:.2f}%")
            print(f"  Human Accuracy: {human_acc:.2f}%")
            print(f"  Model RT: {model_rt:.3f}s")
            print(f"  Human RT: {human_rt:.3f}s")
            print(f"  RT Ratio: {rt_ratio:.2f}x")

if __name__ == '__main__':
    main()
