================================================================================
                    MNIST ConvLSTM Experiment Directory Guide
================================================================================

Last Updated: 2026-03-19

Directory Structure:
--------------------
src/experiments/mnist_convlstm/
├── Core Training Scripts
│   ├── 02_train_model.py              # Base model definition (RTify_ConvLSTM)
│   ├── train_model_balanced.py        # Balanced loss training version
│   └── train_model_log.py             # Log-normalized RT training version
│
├── SAT Model Training Scripts
│   ├── train_sat.py                   # SAT training (with confidence output)
│   ├── train_sat_4param.py            # 4-parameter SAT model training ★ Primary
│   ├── train_sat_stage2.py            # Stage 2 fine-tuning
│   └── train_model_sat.py             # Early SAT version
│
├── Weight Transfer Scripts
│   └── transfer_to_4param_sat.py      # Transfer weights to 4-param SAT ★ Primary
│
├── Experimental Scripts
│   └── train_alexnet_lstm.py          # AlexNet+LSTM architecture experiment
│
├── Evaluation & Visualization
│   ├── 03_evaluate_model.py           # Model evaluation script
│   └── 03_visualize_results_apa.py    # APA format result visualization
│
└── archive/                           # Archived experimental scripts
    ├── 04_train_rt_distribution.py    # RT distribution matching experiment
    ├── train_two_stage.py             # Two-stage training strategy experiment
    └── train_model_with_noise.py      # Input noise experiment

================================================================================
                            File Descriptions
================================================================================

【Core Training Scripts】
------------------------

1. 02_train_model.py
   - Function: Define base ConvLSTM model (RTify_ConvLSTM)
   - Contains: ConvLSTM, RTify_ConvLSTM, DiffDecision, add_noise core components
   - Usage: Base dependency for other training scripts
   - Key Parameters:
     * time_steps: Decision time steps (default: 20)
     * sigma: Soft decision weighting parameter (default: 2.0)
     * learnable_noise: Whether to use learnable noise parameters

2. train_model_balanced.py
   - Function: Balanced loss training (classification + RT loss)
   - Features: Adjustable RT loss weight and speed penalty
   - Usage: Train model for simultaneous classification and RT prediction
   - Key Parameters:
     * rt_loss_weight: RT loss weight (default: 2.0)
     * speed_penalty: Speed penalty coefficient (default: 0.1)

3. train_model_log.py
   - Function: Training with log-normalized RT values
   - Features: Better matching to human RT distribution
   - Dependencies: Imports model definition from 02_train_model.py

【SAT Model Training Scripts】
-----------------------------

4. train_sat.py
   - Function: SAT-conditioned threshold training
   - Features:
     * Separate speed/accuracy thresholds
     * Outputs confidence (max_prob - second_max_prob)
   - Usage: Train model that distinguishes speed/accuracy conditions

5. train_sat_4param.py ★ (Currently Primary)
   - Function: 4-parameter SAT model training
   - Features:
     * threshold_speed: Learnable threshold (speed condition)
     * threshold_accuracy: Learnable threshold (accuracy condition)
     * speed_penalty_speed: Fixed penalty coefficient (0.3)
     * speed_penalty_accuracy: Fixed penalty coefficient (0.08)
   - Usage Example:
     python -m src.experiments.mnist_convlstm.train_sat_4param \
         --pretrained_path outputs/convlstm_4param_sat.pth \
         --output_dir outputs/ \
         --epochs 40 --batch_size 64 --lr 0.001 \
         --rt_loss_weight 2.0 \
         --speed_penalty_speed 0.3 \
         --speed_penalty_accuracy 0.08

6. train_sat_stage2.py
   - Function: Stage 2 fine-tuning training
   - Features: Fine-tune on noisy MNIST with difficulty levels
   - Usage: Evaluate model performance under Easy/Difficult conditions

7. train_model_sat.py
   - Function: Early SAT training version
   - Status: Retained for reference

【Weight Transfer Scripts】
--------------------------

8. transfer_to_4param_sat.py ★ (Currently Primary)
   - Function: Transfer single-threshold model weights to 4-param SAT model
   - Process:
     1. Load pretrained single-threshold model
     2. Create 4-parameter SAT model
     3. Transfer network weights (exclude old threshold)
     4. Initialize new thresholds (speed=accuracy=original)
     5. Set fixed penalty coefficients
   - Usage Example:
     python -m src.experiments.mnist_convlstm.transfer_to_4param_sat \
         --exp11_path outputs/convlstm_balanced.pth \
         --output_path outputs/convlstm_4param_sat.pth \
         --speed_penalty_speed 0.3 \
         --speed_penalty_accuracy 0.08

【Experimental Scripts】
-----------------------

9. train_alexnet_lstm.py
   - Function: AlexNet + LSTM architecture experiment
   - Features: Different architecture comparison experiment
   - Status: Retained for architecture comparison reference

【Evaluation & Visualization】
-----------------------------

10. 03_evaluate_model.py
    - Function: Model evaluation script
    - Outputs: Accuracy, RT correlation, RT distribution, etc.

11. 03_visualize_results_apa.py
    - Function: APA format result visualization
    - Features: Compliant with academic publication standards
    - Outputs: Training curves, RT distribution plots, etc.

================================================================================
                            Model Architecture
================================================================================

【Base Model: RTify_ConvLSTM】
-----------------------------
Input: MNIST image [B, 1, 28, 28]
      ↓
ConvLSTM (temporal processing) → hidden_states [T, B, F, H, W]
      ↓
AdaptivePool + FC → logit_trajectory [B, T, output_size]
      ↓
Evidence Module (MLP) → s_traj [B, T]
      ↓
Noise (Gaussian + Dropout) → s_traj_noisy
      ↓
Evidence Accumulation (cumsum) → s_accumulated
      ↓
DiffDecision (threshold) → decision_time [B]
      ↓
Output: decision_logits [B, 8], rt_normalized [B]

【SAT Model: RTify_ConvLSTM_SAT】
--------------------------------
Inherits from RTify_ConvLSTM, core improvements:
- Single threshold → SAT-conditioned thresholds
- threshold_speed (speed condition, lower → faster decisions)
- threshold_accuracy (accuracy condition, higher → more accurate)

【4-Parameter SAT Model】
------------------------
- threshold_speed: Learnable
- threshold_accuracy: Learnable
- speed_penalty_speed: Fixed (0.3)
- speed_penalty_accuracy: Fixed (0.08)

================================================================================
                            Theoretical Background
================================================================================

Based on Alós-Ferrer & Garagnani (2026):
"Speed-accuracy tradeoffs can often be captured by assuming that the drift rate 
of a single process remains unchanged, but the thresholds become lower"

Human Data Support:
- Speed focus: RT=0.855s, Acc=69.2%
- Accuracy focus: RT=1.045s, Acc=71.2%
- RT difference: 0.189s (speed condition is faster)

================================================================================
                            Archived Files
================================================================================

The archive/ directory contains archived experimental scripts:

1. 04_train_rt_distribution.py
   - Function: RT distribution matching experiment
   - Archive reason: Experimental feature, large codebase

2. train_two_stage.py
   - Function: Two-stage training strategy
   - Archive reason: Experimental training strategy

3. train_model_with_noise.py
   - Function: Input noise training
   - Archive reason: Functionality integrated into other scripts

================================================================================
                            Deleted Files
================================================================================

The following redundant files were deleted (2025-03-19):

1. transfer_to_sat.py
   - Reason: Completely replaced by transfer_to_4param_sat.py

2. transfer_to_sat_fixed.py
   - Reason: Completely replaced by transfer_to_4param_sat.py

================================================================================
                            Usage Recommendations
================================================================================

【Recommended Workflow】

1. Base Training:
   python -m src.experiments.mnist_convlstm.train_model_balanced \
       --use_rt_loss --rt_loss_weight 2.0 --epochs 70

2. Weight Transfer:
   python -m src.experiments.mnist_convlstm.transfer_to_4param_sat \
       --exp11_path outputs/convlstm_balanced.pth \
       --output_path outputs/convlstm_4param_sat.pth

3. 4-Parameter SAT Training:
   python -m src.experiments.mnist_convlstm.train_sat_4param \
       --pretrained_path outputs/convlstm_4param_sat.pth \
       --epochs 40

4. Result Visualization:
   python -m src.experiments.mnist_convlstm.03_visualize_results_apa

================================================================================
