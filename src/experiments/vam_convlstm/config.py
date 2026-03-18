"""
VAM ConvLSTM Experiment Configuration

This module contains configuration and utility functions for VAM experiments.
"""

import os

VAM_DATA_DIR = 'VAM_Lost-in-Migration'
OUTPUT_BASE_DIR = 'outputs/experiments/vam_convlstm'

DEFAULT_CONFIG = {
    'data': {
        'max_trials_per_user': 25000,
        'min_rt': 250,
        'image_size': 128,
        'train_ratio': 0.8,
        'use_log_norm': True,
    },
    'model': {
        'input_channel': 3,
        'num_filter': 16,
        'kernel_size': 3,
        'output_size': 4,
        'time_steps': 20,
        'sigma': 2.0,
        'noise_position': 'evidence',
        'evidence_noise_std': 0.5,
        'evidence_mask_p': 0.4,
        'learnable_noise': False,
    },
    'training': {
        'epochs': 100,
        'batch_size': 64,
        'lr': 1e-3,
        'use_rt_loss': True,
        'speed_penalty': 0.0,
        'learn_human_response': True,
    }
}

EXPERIMENTS = {
    'exp01_log_norm': {
        'description': 'Train with log normalization and fixed noise (consistent with MNIST)',
        'pretrained': None,
        'freeze_encoder': False,
        'epochs': 100,
        'use_log_norm': True,
        'learnable_noise': False,
    },
    'exp02_finetune_mnist': {
        'description': 'Fine-tune from MNIST pretrained model',
        'pretrained': 'outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100/convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp.pth',
        'freeze_encoder': False,
        'epochs': 50,
        'use_log_norm': True,
        'learnable_noise': False,
    },
}


def get_experiment_dir(experiment_name):
    """Get the output directory for an experiment."""
    return os.path.join(OUTPUT_BASE_DIR, experiment_name)


def get_experiment_config(experiment_name):
    """Get the configuration for an experiment."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    config = DEFAULT_CONFIG.copy()
    config['experiment'] = EXPERIMENTS[experiment_name].copy()
    return config
