"""
ConvLSTM with SAT-Conditioned Threshold

Based on the theoretical framework from Alós-Ferrer and Garagnani (2026):
"Speed-accuracy tradeoffs can often be captured by assuming that the drift rate 
of a single process remains unchanged, but the thresholds become lower (e.g., 
in response to time pressure, cognitive load, or a focus on speed rather than accuracy)"

This model learns separate thresholds for Speed and Accuracy conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.experiments.mnist_convlstm.train_model_balanced import RTify_ConvLSTM


class DiffDecision(torch.autograd.Function):
    """Differentiable decision time computation."""
    
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


class RTify_ConvLSTM_SAT(RTify_ConvLSTM):
    """
    ConvLSTM model with SAT-conditioned threshold.
    
    Learns separate thresholds for:
    - Speed focus: Lower threshold -> faster decisions
    - Accuracy focus: Higher threshold -> more accurate decisions
    
    Based on RTNet's design:
    - threshold_levels = [3, 5] for speed/accuracy
    - noise_levels = [2.1, 2.9] for easy/difficult
    """
    
    def __init__(self, input_channel=1, num_filter=16, kernel_size=3, 
                 output_size=8, time_steps=20, sigma=1.0,
                 noise_position='evidence', evidence_noise_std=0.5,
                 evidence_mask_p=0.4, evidence_dropout_rescale=True,
                 learnable_noise=False):
        
        super().__init__(
            input_channel=input_channel,
            num_filter=num_filter,
            kernel_size=kernel_size,
            output_size=output_size,
            time_steps=time_steps,
            sigma=sigma,
            noise_position=noise_position,
            evidence_noise_std=evidence_noise_std,
            evidence_mask_p=evidence_mask_p,
            evidence_dropout_rescale=evidence_dropout_rescale,
            learnable_noise=learnable_noise
        )
        
        # Replace single threshold with SAT-specific thresholds
        # Following RTNet's design: speed=3, accuracy=5
        del self.threshold
        self.threshold_speed = nn.Parameter(torch.tensor(3.0))
        self.threshold_accuracy = nn.Parameter(torch.tensor(5.0))
        
        # SAT condition mapping
        self.sat_mapping = {
            'speed focus': 0,
            'accuracy focus': 1,
            'speed': 0,
            'accuracy': 1,
            'unknown': 0
        }
    
    def _get_threshold_batch(self, sat_conditions, batch_size, device):
        """
        Get threshold tensor for a batch based on SAT conditions.
        
        Args:
            sat_conditions: list of SAT condition strings
            batch_size: batch size
            device: torch device
        
        Returns:
            threshold tensor of shape [batch_size]
        """
        thresholds = []
        for sat in sat_conditions:
            if isinstance(sat, str):
                sat_lower = sat.lower()
                if sat_lower in ['speed focus', 'speed']:
                    thresholds.append(self.threshold_speed)
                elif sat_lower in ['accuracy focus', 'accuracy']:
                    thresholds.append(self.threshold_accuracy)
                else:
                    thresholds.append(self.threshold_speed)  # Default
            else:
                thresholds.append(self.threshold_speed)  # Default
        
        return torch.stack(thresholds)
    
    def forward(self, x, sat_condition=None):
        """
        Forward pass with SAT-conditioned threshold.
        
        Args:
            x: input images [B, C, H, W]
            sat_condition: SAT condition(s) - list of strings
        
        Returns:
            decision_logits: [B, output_size]
            rt_normalized: [B]
        """
        device = x.device
        B, C, H, W = x.shape

        # Process through ConvLSTM
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        hidden_states, (h, c) = self.convlstm(x_seq, seq_len=self.time_steps)

        # Flatten and pool
        time_steps, B, num_filter, H, W = hidden_states.shape
        hidden_2d = hidden_states.view(time_steps * B, num_filter, H, W)
        pooled_2d = self.pool(hidden_2d).squeeze()
        hidden_states = pooled_2d.view(time_steps, B, num_filter)

        # Get logits trajectory
        logit_trajectory = self.fc(hidden_states).squeeze().permute(1, 0, 2)
        
        # Compute evidence
        s_traj = self.evidence(hidden_states).squeeze(-1).permute(1, 0)
        
        # Add noise if needed
        if self.noise_position in ['evidence', 'both']:
            s_traj = self._add_noise_to_evidence(s_traj)
        
        # Accumulate evidence
        s_accumulated = torch.cumsum(s_traj, dim=1)
        dsdt_trajectory = torch.diff(s_accumulated, dim=1)
        dsdt_trajectory = torch.cat((dsdt_trajectory[:, 0].unsqueeze(1), dsdt_trajectory), dim=1)
        
        # Get SAT-conditioned threshold
        if sat_condition is not None:
            threshold_batch = self._get_threshold_batch(sat_condition, B, device)
        else:
            threshold_batch = self.threshold_speed.expand(B)
        
        # Compute decision time: subtract threshold from accumulated evidence
        # s_accumulated: [B, time_steps], threshold_batch: [B]
        decision_time = DiffDecision.apply(
            s_accumulated - threshold_batch.unsqueeze(1), 
            dsdt_trajectory
        )
        
        # Soft indexing for decision logits
        soft_index = torch.exp(-0.5 * (decision_time.unsqueeze(1) - torch.arange(self.time_steps, device=device)) ** 2 / self.sigma ** 2)
        soft_index = soft_index / soft_index.sum(dim=-1, keepdim=True)
        decision_logits = (logit_trajectory * soft_index.unsqueeze(-1)).sum(dim=1)
        
        # Compute confidence: max probability - second max probability
        probs = F.softmax(decision_logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
                
        return decision_logits, (decision_time + 1) / self.time_steps, confidence
    
    def _add_noise_to_evidence(self, s_traj):
        """Add noise to evidence trajectory."""
        if self.learnable_noise:
            noise_std = self.noise_std
            mask_p = self.mask_p
        else:
            noise_std = self._fixed_noise_std
            mask_p = self._fixed_mask_p
        
        # Add Gaussian noise
        noise = torch.randn_like(s_traj) * noise_std
        s_traj = s_traj + noise
        
        # Apply dropout mask
        if mask_p > 0:
            mask = torch.bernoulli(torch.ones_like(s_traj) * (1 - mask_p))
            s_traj = s_traj * mask
            if self.evidence_dropout_rescale:
                s_traj = s_traj / (1 - mask_p)
        
        return s_traj
    
    def get_threshold_values(self):
        """Return current threshold values for logging."""
        return {
            'threshold_speed': self.threshold_speed.item(),
            'threshold_accuracy': self.threshold_accuracy.item()
        }
