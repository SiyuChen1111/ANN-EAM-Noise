"""
Transfer weights from exp11 single-threshold model to 4-parameter SAT model.

4-parameter SAT model:
- threshold_speed: learnable, initialized from exp11 threshold (4.28)
- threshold_accuracy: learnable, initialized from exp11 threshold (4.28)
- speed_penalty_speed: fixed (0.3) - data-driven from human RT difference
- speed_penalty_accuracy: fixed (0.08) - data-driven from human RT difference

Human data analysis:
- speed focus: RT=0.855s, Acc=69.2%
- accuracy focus: RT=1.045s, Acc=71.2%
- RT difference: 0.189s (speed is faster)
- This justifies: speed_penalty_speed > speed_penalty_accuracy
"""

import torch
import torch.nn as nn
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.convlstm_sat import RTify_ConvLSTM_SAT
from src.experiments.mnist_convlstm.train_model_balanced import RTify_ConvLSTM


def transfer_exp11_to_4param_sat(
    exp11_path,
    output_path,
    threshold_speed=3.0,
    threshold_accuracy=4.28,
    speed_penalty_speed=0.3,
    speed_penalty_accuracy=0.08,
    device='cpu'
):
    """
    Transfer weights from exp11 single-threshold model to 4-param SAT model.

    Args:
        exp11_path: Path to exp11 model checkpoint
        output_path: Path to save 4-param SAT model
        threshold_speed: Initial value for threshold_speed (will be set to 4.28 from exp11)
        threshold_accuracy: Initial value for threshold_accuracy (will be set to 4.28 from exp11)
        speed_penalty_speed: Fixed speed_penalty for speed condition
        speed_penalty_accuracy: Fixed speed_penalty for accuracy condition
        device: Device to load model on
    """
    print("="*60)
    print("Weight Transfer: exp11 -> 4-param SAT model")
    print("="*60)

    # Load exp11 checkpoint
    checkpoint = torch.load(exp11_path, map_location=device, weights_only=False)
    exp11_config = checkpoint.get('config', {})
    original_threshold = checkpoint.get('final_threshold_speed', 4.28)

    print(f"\nLoaded exp11 model from: {exp11_path}")
    print(f"Original threshold: {original_threshold:.4f}")
    print(f"Stage 1 accuracy: {checkpoint.get('final_accuracy_correct', 'N/A')}")

    # Create 4-param SAT model with same config as exp11
    model = RTify_ConvLSTM_SAT(
        input_channel=exp11_config.get('input_channel', 1),
        num_filter=exp11_config.get('num_filter', 16),
        kernel_size=exp11_config.get('kernel_size', 3),
        output_size=exp11_config.get('output_size', 8),
        time_steps=exp11_config.get('time_steps', 40),
        sigma=exp11_config.get('sigma', 1.0),
        noise_position=exp11_config.get('noise_position', 'evidence'),
        evidence_noise_std=exp11_config.get('evidence_noise_std', 0.5),
        evidence_mask_p=exp11_config.get('evidence_mask_p', 0.4),
        evidence_dropout_rescale=exp11_config.get('evidence_dropout_rescale', False),
        learnable_noise=exp11_config.get('learnable_noise', False)
    )

    # Load exp11 state dict
    exp11_state_dict = checkpoint['model_state_dict']

    # Filter out 'threshold' from exp11 and load the rest
    sat_state_dict = {}
    for key, value in exp11_state_dict.items():
        if 'threshold' not in key:
            sat_state_dict[key] = value

    # Load filtered state dict
    model.load_state_dict(sat_state_dict, strict=False)

    # Set threshold values from exp11 (both start at 4.28)
    model.threshold_speed.data.fill_(original_threshold)
    model.threshold_accuracy.data.fill_(original_threshold)

    # Make thresholds learnable
    model.threshold_speed.requires_grad = True
    model.threshold_accuracy.requires_grad = True

    # Store fixed speed_penalty values (not model parameters, just config)
    model.speed_penalty_speed = speed_penalty_speed
    model.speed_penalty_accuracy = speed_penalty_accuracy

    print(f"\n4-param SAT model created:")
    print(f"  threshold_speed: {model.threshold_speed.item():.4f} (learnable)")
    print(f"  threshold_accuracy: {model.threshold_accuracy.item():.4f} (learnable)")
    print(f"  speed_penalty_speed: {model.speed_penalty_speed} (fixed)")
    print(f"  speed_penalty_accuracy: {model.speed_penalty_accuracy} (fixed)")

    # Verify thresholds are different
    print(f"\nThreshold verification:")
    print(f"  threshold_speed.requires_grad: {model.threshold_speed.requires_grad}")
    print(f"  threshold_accuracy.requires_grad: {model.threshold_accuracy.requires_grad}")

    # Save 4-param SAT model
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': exp11_config,
        'stage1_path': exp11_path,
        'original_threshold': original_threshold,
        'final_threshold_speed': model.threshold_speed.item(),
        'final_threshold_accuracy': model.threshold_accuracy.item(),
        'speed_penalty_speed': model.speed_penalty_speed,
        'speed_penalty_accuracy': model.speed_penalty_accuracy,
        'transfer_type': 'exp11_to_4param_sat',
        'threshold_fixed': False,  # thresholds are learnable
        'speed_penalty_fixed': True  # speed_penalty is fixed
    }

    torch.save(save_dict, output_path)
    print(f"\nSaved 4-param SAT model to: {output_path}")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transfer exp11 to 4-param SAT model')
    parser.add_argument('--exp11_path', type=str,
                        default='outputs/experiments/mnist_convlstm/exp11_t40/convlstm_balanced_rt2.0_sp0.1_ep70.pth',
                        help='Path to exp11 model')
    parser.add_argument('--output_path', type=str,
                        default='outputs/experiments/mnist_convlstm/exp11_t40/convlstm_4param_sat.pth',
                        help='Path to save 4-param SAT model')
    parser.add_argument('--speed_penalty_speed', type=float, default=0.3,
                        help='Fixed speed_penalty for speed condition')
    parser.add_argument('--speed_penalty_accuracy', type=float, default=0.08,
                        help='Fixed speed_penalty for accuracy condition')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda/mps)')

    args = parser.parse_args()

    transfer_exp11_to_4param_sat(
        exp11_path=args.exp11_path,
        output_path=args.output_path,
        speed_penalty_speed=args.speed_penalty_speed,
        speed_penalty_accuracy=args.speed_penalty_accuracy,
        device=args.device
    )
