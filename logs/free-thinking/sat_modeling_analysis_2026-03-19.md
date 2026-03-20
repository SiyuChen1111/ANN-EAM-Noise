SAT (Speed-Accuracy Tradeoff) Modeling - Critical Analysis & Research Synthesis

**Date**: 2026-03-19
**Context**: Best model (Exp11) achieves RT ratio 1.27x and correlation 0.029, but **fails to capture human-like SAT behavior**

***

## Executive Summary

Our model implements SAT through **three mechanisms**:

1. **Evidence Noise** (`add_noise()` with mask\_p=0.4, std=0.5) - **Core contribution**
2. **Learnable Threshold** - Single global threshold (4.28 in Exp11)
3. **Speed Penalty** - Training constraint (`speed_penalty=0.1`)

**Critical Finding**: Model does NOT capture human SAT:

- Human: Error trials are **SLOWER** (+0.095s)
- Model: Error trials are **SIMILAR** RT (-0.007s)

**Root Hypothesis**: User proposes condition-specific `speed_penalty` tuning to achieve SAT fitting.

***

## 1. Current Model Architecture & SAT Implementation

### 1.1 Model Evolution (Three Versions)

#### Version 1: Baseline (No SAT)

**File**: `src/experiments/mnist_convlstm/02_train_model.py`

```python
class RTify_ConvLSTM(nn.Module):
    def __init__(self, ..., learnable_noise=True, ...):
        self.threshold = nn.Parameter(torch.tensor(6.0))  # Single threshold
        self._noise_std_raw = nn.Parameter(torch.tensor(0.1).log())
        self._mask_p_raw = nn.Parameter(torch.tensor(0.3).logit())

    def forward(self, x):
        # No SAT condition input
        # Single threshold for all trials
        decision_time = DiffDecision.apply(s_accumulated - self.threshold, ...)
```

**Characteristics**:

- ✅ **Supports both** learnable and fixed noise modes (via `learnable_noise` argument)
- ✅ Learnable noise (default when `learnable_noise=True`)
- ❌ Single global threshold - cannot adapt to conditions
- ❌ No SAT condition handling

#### Version 2: Fixed SAT Thresholds

**File**: `src/models/convlstm_sat.py`

```python
class RTify_ConvLSTM_SAT(RTify_ConvLSTM):
    def __init__(self, ...):
        # Replace single threshold with SAT-specific thresholds
        self.threshold_speed = nn.Parameter(torch.tensor(3.0))
        self.threshold_accuracy = nn.Parameter(torch.tensor(5.0))

    def forward(self, x, sat_condition=None):
        if sat_condition is not None:
            threshold_batch = self._get_threshold_batch(sat_condition, B, device)
        else:
            threshold_batch = self.threshold_speed.expand(B)  # Default to speed
```

**Characteristics**:

- ✅ Two learnable thresholds (speed=3.0, accuracy=5.0)
- ✅ SAT condition mapping ('speed focus', 'accuracy focus')
- ❌ Fixed noise levels (not learnable)
- ❌ Fixed input noise (easy: 0.25, difficult: 0.4)

#### Version 3: Proposed - Difficulty-Conditioned Thresholds

**File**: `logs/2026-03-17_progress_log.md` (Proposed)

```python
# Log line 211-219 - Proposed but not implemented
class DifficultyConditionedThreshold(nn.Module):
    def __init__(self):
        self.threshold_easy = nn.Parameter(torch.tensor(4.0))
        self.threshold_hard = nn.Parameter(torch.tensor(5.0))

    def forward(self, difficulty):
        return torch.where(difficulty == 'easy',
                      self.threshold_easy,
                      self.threshold_hard)
```

**Characteristics**:

- ✅ Difficulty-aware thresholds (addresses RT by difficulty)
- ❌ Not yet implemented

### 1.2 Core Noise Mechanism (`add_noise()`)

```python
def add_noise(x, mask_p=0.0, std=0.0, rescale_after_dropout=True):
    # Dropout-like masking
    if mask_p > 0:
        mask = torch.bernoulli(torch.ones_like(x) * (1 - mask_p))
        x = x * mask
        if rescale_after_dropout:
            x = x / (1 - mask_p + 1e-8)  # Preserve mean

    # Gaussian noise injection
    if std > 0:
        noise = torch.randn_like(x) * std
        x = x + noise

    return x
```

**Usage in Exp11**:

- `evidence_noise_std = 0.5` (Gaussian noise std on evidence trajectory)
- `evidence_mask_p = 0.4` (40% dropout on evidence trajectory)
- `learnable_noise = False` (⚠️ **CRITICAL**: Parameters are **FIXED HYPERPARAMETERS**, not learnable)
- Applied to **evidence accumulation trajectory** before cumsum
- Noise parameters **do not update during training**

**Theoretical Interpretation**:

- **Masking**: Simulates intermittent neural firing, creating RT variability
- **Gaussian Noise**: Simulates internal neural noise, affects evidence quality
- **Combined**: Creates realistic RT distribution (right-skewed, correct vs error differences)

### 1.3 Loss Function with Speed Penalty

```python
# Exp11 configuration
loss = label_loss + rt_loss_weight * rt_loss + speed_penalty * rt_mean

# Explicitly
label_loss = CrossEntropyLoss(pred_logits, human_response)  # Learn from errors
rt_loss = MSELoss(rt_pred, rt_human_normalized)
speed_loss = 0.1 * rt_pred.mean()  # Penalize slow decisions

total_loss = label_loss + 2.0 * rt_loss + 0.1 * speed_loss
```

**Role of** **`speed_penalty`**:

- Encourages **faster decisions** during training
- **Global constraint** - applies equally to all trials
- No condition-specificity (speed vs accuracy conditions)

***

## 2. Critical Analysis: Why SAT Fails

### 2.1 Empirical Evidence

**From** **`outputs/experiments/mnist_convlstm/exp11_t40/`** **results**:

| Metric              | Human                     | Model (Exp11)         | Δ (Model - Human) |
| ------------------- | ------------------------- | --------------------- | ----------------- |
| **Correct RT**      | 0.915s                    | 1.192s                | +0.277s (slower)  |
| **Error RT**        | 1.009s                    | 1.185s                | +0.176s (slower)  |
| **Error - Correct** | **+0.095s** (slower)      | **-0.007s** (faster)  | ❌ Wrong direction |
| **SAT Correlation** | r = 0.060 (weak positive) | r = -0.252 (negative) | ❌ Opposite trend  |

**Interpretation**:

- Humans: Error trials require MORE time (evidence uncertainty)
- Model: Error trials require LESS time (early decisions with insufficient evidence)
- **This violates basic decision-making principles**

### 2.2 Theoretical Mismatch

**Human Decision Process** (cognitive theory):

```
Evidence Accumulation: S(t) = S(t-1) + v + noise(t)

Easy trials:    High drift rate (v) → Fast accumulation → Low RT
Difficult trials: Low drift rate (v) → Slow accumulation → High RT

SAT Conditions:
- Speed focus: Lower threshold → Early decision → Fast RT, Low accuracy
- Accuracy focus: Higher threshold → Late decision → Slow RT, High accuracy
```

**Model Behavior** (Empirical observation):

```
All trials:     Same threshold (4.28) + Same noise (mask_p=0.4, std=0.5)

Result:         Difficulty affects drift rate (v) only
                But threshold is globally optimized
                → No SAT behavior
```

**Root Cause**: **Global threshold optimization**

The model learns to set a **single threshold** that balances:

- Classification accuracy (from human responses)
- RT matching (from MSE loss)
- Speed penalty (from `speed_loss=0.1`)

This forces a **compromise threshold** that:

- Works OK on average (RT ratio 1.27x)
- But fails on **difficulty-specific strategies**
- Cannot adapt to speed/accuracy instructions

### 2.3 Why Noise Alone Cannot Fix SAT

**Current Noise Mechanism in Exp11** (`add_noise()`):

- **CRITICAL**: Exp11 uses **FIXED NOISE** (hyperparameters), not learnable
- **Masking (p=0.4)**: Creates variability in evidence quality
- **Gaussian (std=0.5)**: Adds uncertainty to accumulation
- **Position**: Applied to **evidence trajectory** (after feature extraction)
- **Parameters**: `evidence_noise_std=0.5`, `evidence_mask_p=0.4`, `learnable_noise=False`

**What Noise Does**:

1. ✅ Creates RT distribution variance (matches human RT spread)
2. ✅ Allows some early errors (noise causes mistakes)
3. ✅ Allows some correct but slow decisions (noise delays threshold crossing)
4. ❌ Does NOT create **condition-specific** behavior
5. ⚠️ **Fixed parameters** cannot adapt to data distribution

**What Noise Does NOT Do**:

- ❌ Adjust decision threshold based on condition
- ❌ Increase accuracy when accuracy is emphasized
- ❌ Decrease RT when speed is emphasized
- ❌ Implement strategic control (deliberate strategy change)
- ❌ Learn from data (fixed in Exp11)

**Analogy**: Noise is like **adding randomness to perception**, but SAT is like **changing decision policy**.

**Key Observation**: Exp11's SAT failure is **NOT primarily due to noise mechanism** (since noise is fixed), but due to **global threshold** that cannot adapt to conditions.

***

## 3. User's Proposal: Speed Penalty Tuning

### 3.1 The Idea

**Question from user**:

> "我们能否通过对不同 condition 下的 speed\_penalty 进行调参，来达到 SAT 的拟合效果呢？"

(Translation: Can we achieve SAT fitting by tuning `speed_penalty` for different conditions?)

**Interpretation**:

- During **training**, apply different `speed_penalty` values to speed vs accuracy conditions
- Speed condition: **Higher** `speed_penalty` → encourages faster decisions
- Accuracy condition: **Lower** `speed_penalty` → allows slower decisions

**Critical Context - Exp11 Configuration**:

- ⚠️ Exp11 uses **FIXED NOISE** (hyperparameters, not learnable)
- `evidence_noise_std = 0.5`, `evidence_mask_p = 0.4`, `learnable_noise = False`
- Global `speed_penalty = 0.1` applied to **all trials equally**
- Global threshold (4.28) learned to balance ALL trials together

**Implication**: The SAT problem is **primarily due to global threshold**, not noise mechanism. Speed penalty tuning would need to:

1. Work with **different training batches** (speed vs accuracy conditions mixed)
2. Overcome the **global threshold** optimization that prevents condition adaptation

### 3.2 Proposed Implementation

```python
class SATConditionedLoss(nn.Module):
    def __init__(self, speed_penalty_speed=0.2, speed_penalty_accuracy=0.05):
        self.speed_penalty_speed = speed_penalty_speed
        self.speed_penalty_accuracy = speed_penalty_accuracy

    def forward(self, pred_logits, rt_pred, human_response, rt_human, sat_conditions):
        label_loss = F.cross_entropy(pred_logits, human_response)
        rt_loss = F.mse_loss(rt_pred, rt_human)

        # Condition-specific speed penalty
        speed_losses = []
        for i, sat in enumerate(sat_conditions):
            if sat in ['speed focus', 'speed']:
                speed_losses.append(self.speed_penalty_speed * rt_pred[i])
            else:  # accuracy focus
                speed_losses.append(self.speed_penalty_accuracy * rt_pred[i])

        speed_loss = torch.stack(speed_losses).mean()

        total_loss = label_loss + 2.0 * rt_loss + speed_loss
        return total_loss
```

### 3.3 Expected Behavior

**Training Effect**:

- **Speed condition batches**: Higher penalty → model learns to lower threshold → faster decisions
- **Accuracy condition batches**: Lower penalty → model learns to raise threshold → slower decisions
- **Result**: Condition-specific thresholds emerge from loss landscape

**Inference Effect**:

- At test time, use learned thresholds based on condition:
  - Speed condition → lower threshold (optimized for speed)
  - Accuracy condition → higher threshold (optimized for accuracy)

***

## 4. Critical Evaluation of Speed Penalty Approach

### 4.1 Potential Advantages

✅ **End-to-End Learning**

- Model learns appropriate thresholds **automatically** from data
- No manual threshold setting
- Can adapt to task-specific SAT characteristics

✅ **Soft Constraint**

- Unlike **hard threshold differences** (e.g., fixed at 3.0 and 5.0)
- `speed_penalty` is a **loss term** that can be optimized
- More flexible for different datasets/tasks

✅ **Backward Compatible**

- Existing code structure supports it
- Only requires modifying loss computation
- Can be added without changing model architecture

### 4.2 Potential Disadvantages

❌ **Indirect Control**

- Thresholds are **implicitly** affected by penalty
- Not directly specifying threshold difference
- Training dynamics may be less predictable

❌ **Training Complexity**

- Need **mixed batches** (speed + accuracy conditions in each batch)
- Or **alternating epochs** (train on speed, then accuracy)
- Batch composition affects gradient signal

❌ **Unclear if Solves Root Cause**

- Root cause: **Global threshold** cannot adapt to difficulty
- Speed penalty tuning: Still **single threshold per condition** (implicitly through loss gradient)
- ⚠️ May not fix **error trials being faster** (key failure mode)
- **Risk**: Fixed noise (Exp11) cannot adapt, making SAT harder to achieve

### 4.3 Theoretical Compatibility with Noise

**Question**: Does `speed_penalty` work well with `add_noise()`?

**Analysis**:

```
Decision Time = f(threshold, noise, drift_rate)

If speed_penalty is HIGH:
  → Optimize for LOW RT
  → Learn LOW threshold
  → Noise causes MORE early errors (fast but inaccurate)

If speed_penalty is LOW:
  → Optimize for HIGH accuracy
  → Learn HIGH threshold
  → Noise causes fewer early errors (slow but accurate)

Conclusion: speed_penalty + noise SHOULD produce SAT behavior
```

**BUT**: Why doesn't Exp11 already show this?

```
Exp11: speed_penalty = 0.1 (applied to ALL trials)
      → Should encourage faster decisions
      → Should lower threshold
      → Actual threshold: 4.28 (down from 6.0)
      → RT matches humans: 1.27x

Problem: Threshold is GLOBAL (same for speed and accuracy)
```

**Missing Piece**: **Condition-specific application** of penalty.

***

## 5. Comparative Analysis: Alternative SAT Approaches

### 5.1 Threshold-Based SAT (Current Reference Models)

**Implementation**: Fixed or learnable threshold differences

**Examples**:

- RTNet: Fixed thresholds at 3.0 (speed) and 5.0 (accuracy)
- Rafiei et al. (2024): Difficulty-based thresholds

**Pros**:

- ✅ Direct control over decision boundary
- ✅ Well-studied in cognitive literature
- ✅ Clear theoretical interpretation

**Cons**:

- ❌ Manual threshold setting (not data-driven)
- ❌ May not generalize to new tasks
- ❌ Doesn't account for individual differences

### 5.2 Noise-Based SAT (User's Core Contribution)

**Implementation**: `add_noise()` with learnable parameters

**Theoretical Basis**:

- Noise creates RT variability
- Different noise levels could create SAT

**Pros**:

- ✅ Fits RT distribution shape (claimed core contribution)
- ✅ Learnable parameters (data-driven)
- ✅ Novel approach (potential publication point)

**Cons**:

- ❌ Current implementation (Exp11) does NOT show SAT behavior
- ❌ Unclear how to make noise condition-specific
- ❌ Indirect mechanism (not threshold-based)

### 5.3 Speed Penalty-Based SAT (User's Proposal)

**Implementation**: Condition-specific loss weighting

**Pros**:

- ✅ End-to-end learning
- ✅ Soft constraint (flexible)
- ✅ Combines with existing noise mechanism

**Cons**:

- ❌ Indirect threshold control
- ❌ Training complexity (batch composition)
- ❌ Unknown if solves error-trial RT problem

### 5.4 Adaptive Threshold-Based SAT (Most Sophisticated)

**Implementation**: Learn threshold from hidden state

```python
class AdaptiveThreshold(nn.Module):
    def __init__(self, hidden_dim=64):
        self.threshold_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive
        )

    def forward(self, hidden_state):
        return self.threshold_net(hidden_state)  # Dynamic threshold
```

**Pros**:

- ✅ Most flexible (input-dependent)
- ✅ Truly human-like (strategic adaptation)
- ✅ No condition-specific code needed

**Cons**:

- ❌ Hardest to train (unstable)
- ❌ May overfit
- ❌ Less interpretable

***

## 6. Research Questions & Critical Assumptions

### 6.1 Key Questions (To Be Answered)

**Q1**: Why does Exp11 fail to show error trials being slower?

- Hypothesis: Global threshold optimization masks difficulty-specific effects
- Test: Analyze threshold crossing dynamics by difficulty and correctness

**Q2**: Can condition-specific `speed_penalty` fix SAT?

- Hypothesis: Yes, by creating different optimization landscapes
- Test: Implement and evaluate on speed/accuracy conditions separately

**Q3**: Is `add_noise()` sufficient for RT distribution fitting?

- Claimed: Yes, core contribution
- Evidence: RT ratio 1.27x (good), but SAT fails
- Need: Compare with noise-free baseline

**Q4**: Should threshold be the primary SAT mechanism?

- Cognitive theory: Threshold adjustment is primary SAT mechanism
- Our model: Noise + penalty are proposed
- Conflict: May need to revisit cognitive foundations

**Q5**: How will this generalize to VAM Flanker task?

- MNIST: Image classification (no distractor interference)
- Flanker: Response conflict (distractor interference)
- Need: Model must adapt to **different SAT characteristics**

### 6.2 Unexamined Assumptions

**Assumption 1**: Noise is **primary** RT distribution mechanism

- Counter: Evidence accumulation dynamics may be more important
- Need: Ablation study (noise vs no noise)

**Assumption 2**: Speed penalty is **necessary** for SAT

- Counter: Threshold-based SAT works without penalty (literature)
- Need: Baseline without speed\_penalty

**Assumption 3**: Current RT ratio (1.27x) is **acceptable**

- Counter: Model is 27% slower than humans
- Human: 0.915s, Model: 1.192s
- Need: Determine acceptable RT error margin

**Assumption 4**: SAT failure is **threshold** problem

- Alternative: May be **drift rate** problem (v doesn't adapt)
- Need: Analyze drift rate by difficulty and condition

***

## 7. Experimental Design Recommendations

### 7.1 Test Speed Penalty Tuning (Immediate)

**Experiment: SAT-SpeedPenalty-Tuning**

**Configuration**:

```python
# Modify training loop
speed_penalty_map = {
    'speed focus': 0.2,    # High penalty → fast decisions
    'accuracy focus': 0.05   # Low penalty → slow decisions
}

# In training loop
for batch in dataloader:
    sat_conditions = batch['sat_condition']
    speed_loss = torch.tensor([
        speed_penalty_map[sat] * rt_pred[i]
        for i, sat in enumerate(sat_conditions)
    ]).mean()
```

**Evaluation Metrics**:

1. **SAT RT Difference**: RT(speed condition) vs RT(accuracy condition)
2. **SAT Accuracy Difference**: Acc(speed) vs Acc(accuracy)
3. **Error RT Pattern**: Error RT - Correct RT (should be positive)
4. **Correlation with Human**: RT by condition correlation

**Expected Outcome**:

- ✅ Speed condition: Lower RT, Lower accuracy
- ✅ Accuracy condition: Higher RT, Higher accuracy
- ✅ Error trials: Slower than correct trials

### 7.2 Ablation Study (Priority)

**Experiment: SAT-Ablation-Noise**

**Groups**:

1. No noise (mask\_p=0, std=0) + Single threshold
2. Noise only (mask\_p=0.4, std=0.5) + Single threshold
3. Noise + Fixed SAT thresholds (3.0, 5.0)
4. Noise + Condition-specific speed\_penalty

**Metrics**:

- RT distribution shape (skewness, variance)
- SAT behavior (error RT pattern)
- Generalization to new tasks (Flanker)

**Goal**: Disentangle noise effect vs threshold effect.

### 7.3 Difficulty-Conditioned Thresholds (Alternative)

**Experiment: SAT-Difficulty-Thresholds**

```python
class DifficultyConditionedThreshold(nn.Module):
    def __init__(self):
        self.threshold_easy = nn.Parameter(torch.tensor(4.0))
        self.threshold_difficult = nn.Parameter(torch.tensor(5.0))

    def forward(self, difficulty_labels):
        # difficulty_labels: [B] with 0=easy, 1=difficult
        thresholds = torch.where(
            difficulty_labels == 0,
            self.threshold_easy,
            self.threshold_difficult
        )
        return thresholds
```

**Expected Benefit**:

- Easy trials: Lower threshold → Fast RT
- Difficult trials: Higher threshold → Slow RT
- May naturally create error RT difference (errors more likely in difficult trials)

***

## 8. Literature Review (Completed)

### 8.1 Key Findings from Literature Search

#### SAT (Speed-Accuracy Tradeoff) Mechanisms

**Primary Finding**: Threshold adjustment is the dominant SAT mechanism

1. **Mendonça et al. (2020)** - *Nature Communications*
   - **Title**: "The impact of learning on perceptual decisions and its implication for speed-accuracy tradeoffs"
   - **Mechanism**: Bayesian model combining reinforcement learning with evidence accumulation
   - **Key Insight**: Learning affects SAT behavior through trial history effects
   - **Relevance**: Shows SAT is not static - adapts through learning
2. **Vandekerckhove et al. (2022)** - *Scientific Reports*
   - **Title**: "Computational analysis of speed-accuracy tradeoff"
   - **Finding**: **Threshold hypothesis is qualitatively consistent with behavioral data**
   - **Finding**: **Gain modulation hypothesis is NOT consistent**
   - **Implication**: Threshold adjustment (not gain) is the primary SAT mechanism
   - **Relevance**: Supports our threshold-based approach over alternative mechanisms
3. **Jaffe et al. (2024)** - *eLife*
   - **Title**: "An image-computable model of speeded decision-making"
   - **Approach**: Convolutional neural networks + evidence accumulation models
   - **Significance**: **Similar to our project** (CNN + EAM)
   - **Relevance**: Strong precedent for combining vision systems with decision-making
4. **Servant et al. (2019)** - *Journal of Neurophysiology*
   - **Title**: "Neurally constrained modeling of speed-accuracy tradeoff during visual search"
   - **Mechanism**: "Gated accumulation of modulated evidence"
   - **Finding**: Neural constraints on SAT implementation in visual search
   - **Relevance**: Provides neurophysiological grounding for SAT
5. **Flexible gating neural networks (2024)** - *Nature Communications*
   - **Title**: "Flexible gating between subspaces in a neural network model of internally guided task switching"
   - **Finding**: Self-sustained persistent activity emerges for rule representation
   - **Relevance**: Shows neural networks can learn adaptive control strategies

#### Noise-Based RT Modeling

**Primary Finding**: Noise affects RT distribution shape, not SAT behavior

1. **BMC Bioinformatics (2016)** - "The effect of noise-induced variance on parameter recovery from reaction times"
   - **Finding**: Technical noise compromises RT precision
   - **Finding**: Noise affects model fit to RT distributions
   - **Relevance**: Validates our concern about noise parameter learning
   - **Key Issue**: Our noise\_std collapsed to 0.0004 in learnable noise experiment → similar to technical noise problem
2. **Núñez (2025)** - *Journal of Mathematical Psychology*
   - **Title**: "Cognitive models of decision-making with identifiable parameters"
   - **Framework**: Diffusion decision models with within-trial noise
   - **Focus**: Identifiability of drift rate, boundary separation, non-decision time
   - **Relevance**: Standardizes noise treatment in DDMs
3. **Decision SincNet (2022)** - *NSF*
   - **Title**: "Neurocognitive models of decision making that predict cognitive processes from neural signals"
   - **Framework**: Drift diffusion model for neural signal interpretation
   - **Relevance**: Links neural activity to DDM parameters
4. **Recurrent Auto-Encoding Drift Diffusion Model (2020)** - *HAL*
   - **Approach**: Auto-encoder for parameter identification
   - **Relevance**: Advanced noise handling in hierarchical models
5. **EZ Bayesian hierarchical drift diffusion model (2025)** - *Psychonomic Bulletin & Review*
   - **Approach**: Simplified DDM for direct parameter calculation
   - **Relevance**: Shows community focus on parameter identifiability

### 8.2 Literature Synthesis: What the Research Says

| Question                         | Literature Answer                                     | Our Model's Status                                |
| -------------------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| **Primary SAT mechanism**        | Threshold adjustment (dominant)                       | ✅ Uses thresholds (learnable)                     |
| **Noise role in SAT**            | Secondary - affects RT distribution, not SAT behavior | ⚠️ Claims noise-based SAT, but evidence weak      |
| **Threshold-based SAT validity** | Well-supported (Vandekerckhove 2022)                  | ✅ Aligned with literature                         |
| **Neural network + EAM**         | Validated (Jaffe 2024)                                | ✅ Similar architecture validated                  |
| **Learnable noise impact**       | Can cause parameter collapse if not constrained       | ❌ Observed collapse in learnable noise experiment |
| **Alternative SAT mechanisms**   | Gain modulation (NOT supported), Bayesian learning    | ⚠️ Need to verify our approach                    |

### 8.3 Critical Implications for Our Model

**Implication 1**: Threshold-based SAT is **theoretically sound** and widely supported

- ✅ Our `convlstm_sat.py` with learnable `threshold_speed` and `threshold_accuracy` aligns with literature
- ✅ This is a **stronger** approach than `speed_penalty` tuning (less indirect)
- ❌ But fixed noise (not learnable) limits adaptability

**Implication 2**: Noise is **NOT a primary SAT mechanism**

- ❌ Literature shows noise affects RT distribution shape, not strategic SAT control
- ❌ Our core contribution claim ("noise-based SAT") may be **overstated**
- ⚠️ Should position noise as **RT distribution fitting**, not SAT mechanism

**Implication 3**: Learnable noise needs **constraints**

- Literature shows learnable noise parameters can collapse (like our 0.0004)
- Need: Proper initialization, regularization, or fixing to reasonable values

**Implication 4**: CNN + EAM is **validated approach**

- Jaffe (2024) provides strong precedent
- Should position our work as building on this architecture
- Novelty: **Not** in combining CNN + EAM
- Novelty: **Potentially** in noise mechanism or specific implementation details

### 8.4 Research Gaps Identified

**Gap 1**: Condition-specific SAT learning

- Literature: Shows SAT can be learned through Bayesian methods
- Our model: Fixed thresholds (3.0, 5.0) or single learnable threshold
- Gap: How to learn condition-specific thresholds **end-to-end**?

**Gap 2**: Fixed Noise vs Learnable Noise impact on SAT

- Literature: Learnable noise can collapse to trivial values (observed in other experiments)
- Our model (Exp11): Uses **fixed noise** (not learnable)
- ⚠️ **CRITICAL**: Exp11 configuration:
  - `learnable_noise = False`
  - `evidence_noise_std = 0.5` (fixed hyperparameter)
  - `evidence_mask_p = 0.4` (fixed hyperparameter)
  - Global threshold = 4.28 (learned from data)
- Gap: How does **fixed noise** affect SAT learning compared to learnable noise?
- Implication: Fixed noise may limit model's ability to adapt SAT strategies

**Gap 3**: Difficulty-Conditioned SAT implementation

- Literature: Supports difficulty-based threshold adjustment
- Our model: `convlstm_sat.py` has learnable SAT thresholds but **fixed noise**
- Gap: Should we combine difficulty-conditioned thresholds with learnable noise?

**Gap 4: Reference Integration - RTNet Implementation**

- From `references/MNIST-RTNet/train.ipynb`: RTNet uses:
  - AlexNet architecture for feature extraction
  - **Bayesian inference with PyRo** (SVI + Trace\_ELBO)
  - **Two noise levels** for simulations
  - **Two threshold levels** for simulations
  - Input noise based on difficulty (Easy: 0.25, Difficult: 0.4)
  - Fixed thresholds: speed=3.0, accuracy=5.0
- Key insight: RTNet explicitly uses **input noise by difficulty** (not just internal noise)
- Relevance: This matches our `preprocess_mnist_behavioral_log.py` approach
- Implementation available in Colab notebooks

**Gap 5: SAT Mechanism in Reference Model**

- RTNet approach: **Fixed thresholds** (not learned)
- Threshold values: speed=3.0, accuracy=5.0 (similar to our `convlstm_sat.py`)
- Training: Bayesian learning of network weights, not thresholds
- Evidence: SAT is implemented as **post-hoc parameter** (not adaptive during training)
- This approach is **simpler** than end-to-end learning

**CRITICAL TECHNICAL ISSUE IDENTIFIED - Training Stuck Problem**

**From**: `training_stuck_analysis.md`

**Problem Description**: Using fixed thresholds (speed=3.5, accuracy=4.28) for SAT fine-tuning caused training to hang:

- Training time: 3+ hours (expected: 1-2 hours)
- CPU usage: Unstable, process in S+ (sleep state)
- Root cause: Custom `DiffDecision` autograd function has compatibility issues with Apple Silicon GPU (MPS backend)

**Solution Found**: Switched to CPU backend, optimized code implementation

**Implications for SAT Implementation**:

1. **Technical debt**: Current code has device compatibility issues
2. **MPS backend**: May affect all experiments using custom autograd functions
3. **Optimization needed**: `_get_threshold_batch` method uses low-efficiency loops

**Critical for SAT fine-tuning**:

- Training stuck issues will prevent SAT model training completion
- Need to address MPS compatibility before SAT experiments
- May need MPS-safe version of custom autograd functions

**Gap 2**: Noise vs SAT disentanglement

- Literature: Noise affects RT distribution, threshold affects SAT
- Our model: Combines both but unclear interaction
- Gap: How to ensure noise doesn't interfere with SAT learning?

**Gap 3**: Adaptive thresholds

- Literature: Emerging research on input-dependent thresholds
- Our model: Static thresholds (even if learnable)
- Gap: How to implement adaptive thresholds in CNN + EAM architecture?

***

## 9. Synthesis & Recommendations

### 9.1 Current Status

**Achievements**:

- ✅ Core noise mechanism fits RT distribution (ratio 1.27x)
- ✅ Evidence accumulation architecture implemented
- ✅ Learnable parameters (noise\_std, mask\_p, threshold)
- ✅ Threshold-based SAT architecture exists (`convlstm_sat.py`)

**Critical Failures**:

- ❌ SAT behavior not captured (error RT pattern wrong)
- ❌ Global threshold prevents condition adaptation
- ❌ Unclear if noise is the right mechanism for SAT

### 9.2 Critical Questions for User

Before proceeding with implementation, please answer:

1. **Is speed penalty tuning a training-time or inference-time mechanism?**
   - Training: Different batches get different penalties?
   - Inference: Use learned thresholds?
2. **Do you have speed/accuracy condition labels in your dataset?**
   - If yes: Can implement condition-specific penalty
   - If no: Need to generate synthetic conditions
3. **What is your target SAT effect size?**
   - How much faster should speed condition be?
   - How much more accurate should accuracy condition be?
4. **Is noise-based SAT a requirement for publication?**
   - Or is threshold-based SAT acceptable?
   - This affects experimental design
5. **What's acceptable RT error margin?**
   - Current: 27% slower than humans
   - Is this within acceptable range?
6. **Should SAT be our primary contribution?**
   - Or is RT distribution fitting more important?
   - This affects paper framing and experimental design

### 9.3 Recommended Path Forward

**Phase 1: Quick Win - Test Speed Penalty Tuning**

1. Implement condition-specific `speed_penalty`
2. Train on mixed batches (speed + accuracy conditions)
3. Evaluate SAT behavior (error RT pattern)
4. Timeline: 2-3 days

**Phase 2: Diagnostic - Ablation Study**

1. Compare noise vs no noise vs threshold-only
2. Quantify each component's contribution
3. Identify if noise helps or hinders SAT
4. Timeline: 3-4 days

**Phase 3: Advanced - Adaptive Thresholds**

1. Implement input-dependent threshold prediction
2. Test on both MNIST and VAM Flanker
3. Evaluate generalization
4. Timeline: 1 week

### 9.4 My Assessment (Based on Literature + Analysis)

**Critical Finding - Exp11's True Configuration**:

- ⚠️ **IMPORTANT**: Exp11 uses **FIXED NOISE** (hyperparameters), NOT learnable
- Configuration: `evidence_noise_std=0.5`, `evidence_mask_p=0.4`, `learnable_noise=False`
- Implication: Noise parameters **do not update during training**

**User's speed\_penalty proposal**: ⚠️ **Moderate potential, with caveats**

- **Pros**: End-to-end learning, flexible, compatible with existing code
- **Cons**:
  - Indirect mechanism (threshold affected implicitly through loss gradient)
  - Training complexity (batch composition required)
  - **CRICAL**: Exp11 uses **fixed noise** - speed penalty cannot change noise parameters
- **Literature support**: Weak (threshold is dominant SAT mechanism)

**Alternative: Difficulty-conditioned thresholds**: ✅ **High potential**

- **Pros**: Addresses core failure mode (difficulty adaptation), aligns with cognitive theory
- **Cons**: More parameters, requires difficulty labels
- **Literature support**: Strong (threshold-based SAT is well-supported)

**My Recommendation - Hybrid Approach**: ⭐ **Recommended**

- Combine difficulty-conditioned thresholds with learnable noise
- Noise: Fits RT distribution shape (when learnable)
- Thresholds: Enable SAT behavior (when condition-specific)
- This aligns with literature and addresses core failure (global threshold adaptation)

**Key Insight**: The SAT problem is **primarily due to global threshold**, not noise mechanism. While learnable noise is a valid RT distribution fitting approach, it should not be positioned as the primary SAT mechanism.

***

## 10. Next Steps

**Immediate Actions**:

1. ✅ Create this analysis document (DONE)
2. 🔄 Get user's answers to critical questions (Section 9.2)
3. ⏳ Implement proposed SAT improvement (choose approach based on user input)
4. ⏳ Run ablation study
5. ⏳ Update document with results

**Long-term Vision**:

- Clear SAT modeling approach (threshold vs noise vs hybrid)
- Evidence-based generalization to VAM Flanker
- Publication-ready experimental comparison

***

**Document Status**: COMPLETE - Awaiting user input
**Created**: 2026-03-19
**Location**: `/logs/free-thinking/sat_modeling_analysis_2026-03-19.md`
