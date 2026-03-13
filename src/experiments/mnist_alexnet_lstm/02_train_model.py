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
from torchvision import models

from preprocess_mnist_behavioral import MNISTBehavioralDataset

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
                 noise_position='input',
                 mask_p=0.0,
                 gaussian_std=0.0,
                 evidence_noise_std=0.0,
                 evidence_mask_p=0.0,
                 evidence_dropout_rescale=False,
                 evidence_scale=1.0,
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
        self.evidence_scale = evidence_scale
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(input_dim)

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
        self.register_buffer('threshold', torch.tensor(threshold))
        self.sigma = sigma

    def forward(self, x):
        B, input_dim = x.shape

        x_normed = self.input_norm(x)

        x_seq = x_normed.unsqueeze(0).repeat(self.time_steps, 1, 1)

        if self.noise_position in ['input', 'both']:
            x_seq = add_noise(x_seq, self.mask_p, self.gaussian_std)

        hidden_states, _ = self.lstm(x_seq)

        logit_trajectory = self.fc(hidden_states).permute(1, 0, 2)
        s_traj = self.evidence(hidden_states).squeeze(-1).permute(1, 0) * self.evidence_scale

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
    def __init__(self, feature_dim=4096, hidden_dim=512, output_size=10,
                 time_steps=20, sigma=2.0, freeze_encoder=True,
                 noise_position='evidence',
                 mask_p=0.0, gaussian_std=0.0,
                 evidence_noise_std=0.0, evidence_mask_p=0.0,
                 evidence_dropout_rescale=False, evidence_scale=1.0,
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
            evidence_scale=evidence_scale,
            threshold=threshold,
            num_lstm_layers=num_lstm_layers
        )

    def forward(self, image):
        z = self.encoder(image)
        decision_logits, decision_time = self.rtify(z)
        return decision_logits, decision_time, z

    def forward_with_diagnostics(self, image):
        z = self.encoder(image)
        decision_logits, decision_time = self.rtify(z)
        return decision_logits, decision_time, z


def train_model(model, train_loader, test_loader=None, num_epochs=5, lr=1e-3, device='cpu',
                use_rt_loss=False, speed_penalty=0.0, save_path=None,
                accuracy_threshold=0.7, save_best=True, test_dataset=None,
                output_dir='./output', filename='model', time_steps=20,
                condition_name=None, random_seed=None):
    from scipy import stats
    
    label_criterion = nn.CrossEntropyLoss()
    rt_criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {
        'rt_loss': [], 'label_loss': [], 'total_loss': [], 'acc': [], 'corr': []
    }
    
    best_model_state = None
    best_epoch_info = {
        'epoch': -1,
        'accuracy': 0.0,
        'correlation': 0.0,
        'meets_accuracy': False
    }

    model.to(device)

    log_path = os.path.join(output_dir, f'{filename}_training_log.txt')
    with open(log_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING LOG - MNIST AlexNet-LSTM Model\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {filename}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"RT Supervision: {'Yes' if use_rt_loss else 'No'}\n")
        f.write(f"Speed Penalty: {speed_penalty}\n")
        f.write("="*70 + "\n\n")

    print(f"\nStarting Training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"RT Supervision: {'Yes' if use_rt_loss else 'No'}")
    print(f"Speed Penalty: {speed_penalty}")
    print(f"Training log: {log_path}")
    print("="*60)

    for epoch in range(num_epochs):
        model.train()
        epoch_metrics = {'rt_loss': [], 'label_loss': [], 'acc': [], 'corr': []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt = batch['rt_normalized'].to(device)

            optimizer.zero_grad()
            decision_logits, rt_pred, _ = model(images)

            rt_loss = rt_criterion(rt_pred, rt)
            label_loss = label_criterion(decision_logits, labels)

            if speed_penalty > 0:
                speed_loss = speed_penalty * rt_pred.mean()
            else:
                speed_loss = 0.0

            if use_rt_loss:
                total_loss = rt_loss + label_loss + speed_loss
            else:
                total_loss = label_loss + speed_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_metrics['rt_loss'].append(rt_loss.item())
            epoch_metrics['label_loss'].append(label_loss.item())
            acc = (decision_logits.argmax(-1) == labels).float().mean().item()
            epoch_metrics['acc'].append(acc)

            rt_pred_np = rt_pred.detach().cpu().numpy().flatten()
            rt_np = rt.cpu().numpy().flatten()
            corr_temp = np.corrcoef(rt_pred_np, rt_np)[0, 1] if len(rt_pred_np) > 1 else 0.0
            epoch_metrics['corr'].append(np.nan_to_num(corr_temp))

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'acc': f'{acc:.3f}',
                'corr': f'{np.nan_to_num(corr_temp):.3f}'
            })

        for key in epoch_metrics:
            history[key].append(np.mean(epoch_metrics[key]))

        if use_rt_loss:
            history['total_loss'].append(history['rt_loss'][-1] + history['label_loss'][-1])
        else:
            history['total_loss'].append(history['label_loss'][-1])

        print(f"\nEpoch {epoch+1}/{num_epochs} Training Summary:")
        print(f"  RT Loss: {history['rt_loss'][-1]:.4f}")
        print(f"  Label Loss: {history['label_loss'][-1]:.4f}")
        print(f"  Accuracy: {history['acc'][-1]*100:.2f}%")
        print(f"  RT Correlation: {history['corr'][-1]:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"  RT Loss: {history['rt_loss'][-1]:.4f}\n")
            f.write(f"  Label Loss: {history['label_loss'][-1]:.4f}\n")
            f.write(f"  Accuracy: {history['acc'][-1]*100:.2f}%\n")
            f.write(f"  RT Correlation: {history['corr'][-1]:.4f}\n")
            f.write("-"*70 + "\n")
        
        if test_loader is not None:
            print(f"\n  Evaluating on test set...")
            model.eval()
            
            all_rt_pred = []
            all_rt_human = []
            all_labels = []
            all_preds = []
            correct_predictions = 0
            total_predictions = 0
            
            can_denormalize = (test_dataset is not None and 
                              hasattr(test_dataset, 'denormalize_rt'))
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    rt_human = batch['rt_normalized'].to(device)

                    decision_logits, decision_time, _ = model(images)
                    pred_labels = decision_logits.argmax(dim=-1)

                    for i in range(len(labels)):
                        true_label = labels[i].item()
                        pred_label = pred_labels[i].item()
                        model_dt = decision_time[i].item()
                        human_rt = rt_human[i].item()

                        if pred_label == true_label:
                            correct_predictions += 1
                        total_predictions += 1

                        all_rt_pred.append(model_dt)
                        all_rt_human.append(human_rt)
                        all_labels.append(true_label)
                        all_preds.append(pred_label)
            
            test_accuracy = correct_predictions / total_predictions
            
            all_rt_pred = np.array(all_rt_pred)
            all_rt_human = np.array(all_rt_human)
            all_labels = np.array(all_labels)
            
            correlation = np.corrcoef(all_rt_pred, all_rt_human)[0, 1]
            
            print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
            print(f"  RT Correlation: {correlation:.4f}")
            
            meets_accuracy = test_accuracy >= accuracy_threshold
            
            if save_best:
                is_better = test_accuracy > best_epoch_info['accuracy']
                
                if is_better:
                    best_epoch_info = {
                        'epoch': epoch + 1,
                        'accuracy': test_accuracy,
                        'correlation': correlation,
                        'meets_accuracy': meets_accuracy
                    }
                    best_model_state = model.state_dict().copy()
                    print(f"  ★ NEW BEST: Higher accuracy ({test_accuracy*100:.2f}%)")
        
        print("-"*60)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"\nFinal model saved to: {save_path}")

    if save_best and best_model_state is not None and save_path:
        best_model_path = save_path.replace('.pth', '_BEST.pth')
        
        torch.save({
            'model_state_dict': best_model_state,
            'epoch': best_epoch_info['epoch'],
            'accuracy': best_epoch_info['accuracy'],
            'correlation': best_epoch_info['correlation'],
            'history': history
        }, best_model_path)
        
        print(f"\n" + "="*60)
        print("BEST MODEL SUMMARY")
        print("="*60)
        print(f"Best Epoch: {best_epoch_info['epoch']}/{num_epochs}")
        print(f"Accuracy: {best_epoch_info['accuracy']*100:.2f}%")
        print(f"RT Correlation: {best_epoch_info['correlation']:.4f}")
        print(f"Saved to: {best_model_path}")
        print("="*60)

    return history, best_epoch_info


def analyze_results(model, test_loader, test_dataset, device, save_path=None):
    from scipy import stats
    
    model.eval()
    model.to(device)
    
    all_rt_pred = []
    all_rt_human = []
    all_labels = []
    all_preds = []
    all_correct = []
    
    can_denormalize = hasattr(test_dataset, 'denormalize_rt')
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Analyzing"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            rt_human = batch['rt_normalized'].to(device)
            correct = batch['correct'].to(device)

            decision_logits, decision_time, _ = model(images)
            pred_labels = decision_logits.argmax(dim=-1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = pred_labels[i].item()
                model_dt = decision_time[i].item()
                human_rt = rt_human[i].item()
                is_correct = correct[i].item()

                all_rt_pred.append(model_dt)
                all_rt_human.append(human_rt)
                all_labels.append(true_label)
                all_preds.append(pred_label)
                all_correct.append(is_correct)
    
    all_rt_pred = np.array(all_rt_pred)
    all_rt_human = np.array(all_rt_human)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_correct = np.array(all_correct)
    
    accuracy = np.mean(all_preds == all_labels)
    correlation = np.corrcoef(all_rt_pred, all_rt_human)[0, 1]
    
    correct_rt = all_rt_pred[all_correct]
    incorrect_rt = all_rt_pred[~all_correct]
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"RT Correlation: {correlation:.4f}")
    print(f"\nRT by Correctness:")
    print(f"  Correct trials: {correct_rt.mean():.4f} ± {correct_rt.std():.4f} (n={len(correct_rt)})")
    print(f"  Incorrect trials: {incorrect_rt.mean():.4f} ± {incorrect_rt.std():.4f} (n={len(incorrect_rt)})")
    
    if can_denormalize:
        correct_rt_ms = test_dataset.denormalize_rt(correct_rt) * 1000
        incorrect_rt_ms = test_dataset.denormalize_rt(incorrect_rt) * 1000
        print(f"\nRT in milliseconds:")
        print(f"  Correct trials: {correct_rt_ms.mean():.1f} ± {correct_rt_ms.std():.1f} ms")
        print(f"  Incorrect trials: {incorrect_rt_ms.mean():.1f} ± {incorrect_rt_ms.std():.1f} ms")
    
    if save_path:
        results_df = pd.DataFrame({
            'true_label': all_labels,
            'pred_label': all_preds,
            'correct': all_correct,
            'rt_pred_normalized': all_rt_pred,
            'rt_human_normalized': all_rt_human
        })
        
        if can_denormalize:
            results_df['rt_pred_seconds'] = test_dataset.denormalize_rt(all_rt_pred)
            results_df['rt_human_seconds'] = test_dataset.denormalize_rt(all_rt_human)
        
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
    
    return {
        'accuracy': accuracy,
        'correlation': correlation,
        'correct_rt_mean': correct_rt.mean(),
        'incorrect_rt_mean': incorrect_rt.mean()
    }


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
    
    parser = argparse.ArgumentParser(description='Train AlexNet-LSTM Model for MNIST with RT prediction')
    parser.add_argument('--data_path', type=str, 
                        default='RTNet_Dataset/behavioral data.csv',
                        help='Path to behavioral data CSV file')
    parser.add_argument('--output_dir', type=str, default='./output_mnist',
                        help='Directory to save model and figures')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--use_rt_loss', action='store_true',
                        help='Use RT supervision during training')
    parser.add_argument('--speed_penalty', type=float, default=0.0,
                        help='Penalty for slow decisions')
    parser.add_argument('--accuracy_threshold', type=float, default=0.7,
                        help='Minimum accuracy threshold for saving best model')
    parser.add_argument('--noise_position', type=str, default='evidence',
                        choices=['input', 'evidence', 'both', 'none'],
                        help='Where to inject noise')
    parser.add_argument('--evidence_noise_std', type=float, default=0.5,
                        help='Evidence Gaussian noise std')
    parser.add_argument('--evidence_mask_p', type=float, default=0.4,
                        help='Evidence dropout probability')
    parser.add_argument('--evidence_dropout_rescale', action='store_true',
                        help='Rescale after evidence dropout')
    parser.add_argument('--time_steps', type=int, default=20,
                        help='Number of time steps for decision')
    parser.add_argument('--threshold', type=float, default=6.0,
                        help='Evidence accumulation threshold for decision')
    parser.add_argument('--evidence_scale', type=float, default=1.0,
                        help='Scale factor for evidence values')
    parser.add_argument('--freeze_encoder', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to freeze encoder weights during training')
    parser.add_argument('--feature_dim', type=int, default=4096,
                        help='Feature dimension from AlexNet')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for LSTM')
    parser.add_argument('--num_lstm_layers', type=int, default=1,
                        help='Number of LSTM layers')
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
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for testing')

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
    print("AlexNet-LSTM Model for MNIST RT Prediction")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Path: {args.data_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Random Seed: {args.random_seed}")
    print(f"  RT Supervision: {args.use_rt_loss}")
    print(f"  Freeze Encoder: {args.freeze_encoder}")
    print(f"  Feature Dim: {args.feature_dim}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Time Steps: {args.time_steps}")
    print(f"  Device: {device}")

    print("\nCreating datasets...")
    full_dataset = MNISTBehavioralDataset(args.data_path)
    
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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == 'cuda'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == 'cuda'
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    print("\nCreating model...")
    noise_pos = None if args.noise_position == 'none' else args.noise_position
    model = AlexNetRTifyModel(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        output_size=10,
        time_steps=args.time_steps,
        freeze_encoder=args.freeze_encoder,
        noise_position=noise_pos,
        mask_p=0.0,
        gaussian_std=0.0,
        evidence_noise_std=args.evidence_noise_std,
        evidence_mask_p=args.evidence_mask_p,
        evidence_dropout_rescale=args.evidence_dropout_rescale,
        evidence_scale=args.evidence_scale,
        threshold=args.threshold,
        num_lstm_layers=args.num_lstm_layers
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    rt_sup = "rt_sup" if args.use_rt_loss else "no_rt_sup"
    freeze_str = "frozen" if args.freeze_encoder else "finetune"
    filename = (f"alexnet_lstm_{freeze_str}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_"
                f"h{args.hidden_dim}_t{args.time_steps}_{rt_sup}")

    model_save_path = os.path.join(args.output_dir, f'{filename}.pth')
    history, best_epoch_info = train_model(
        model, train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_rt_loss=args.use_rt_loss,
        speed_penalty=args.speed_penalty,
        save_path=model_save_path,
        accuracy_threshold=args.accuracy_threshold,
        save_best=True,
        test_dataset=full_dataset,
        output_dir=args.output_dir,
        filename=filename,
        time_steps=args.time_steps,
        random_seed=args.random_seed
    )

    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    results_path = os.path.join(args.output_dir, f'{filename}_results.csv')
    results = analyze_results(model, test_loader, full_dataset, device, results_path)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_save_path}")
    print(f"Results saved to: {results_path}")
    print(f"\nFinal Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Final RT Correlation: {results['correlation']:.4f}")


if __name__ == '__main__':
    main()
