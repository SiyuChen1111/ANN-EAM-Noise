# FS-Net (Fast-Same Network) 模型架构与运作说明

## 1. 模型概述

FS-Net是一个用于模拟人类感知决策过程的深度神经网络模型，特别设计用于研究**Fast-Same效应**（匹配试次的反应时间快于不匹配试次）。该模型结合了对比学习预训练的图像编码器和基于ConvLSTM的证据累积机制。

### 1.1 核心功能

- **双图像匹配任务**: 判断两个视觉刺激（形状图像和符号图像）是否匹配
- **反应时间预测**: 模拟人类决策过程的时间动态
- **Fast-Same效应建模**: 捕捉匹配/不匹配条件下的反应时间差异

### 1.2 模型特点

| 特性 | 说明 |
|------|------|
| **架构类型** | 编码器-解码器结构（Encoder-RTify） |
| **核心组件** | CNN编码器 + ConvLSTM + 可微分决策模块 |
| **训练方式** | 监督学习（可选RT监督）+ 对比学习预训练 |
| **噪声机制** | 支持输入噪声和证据轨迹噪声 |

---

## 2. 模型架构详解

### 2.1 整体架构

```
输入图像1 (形状)          输入图像2 (符号)
    │                         │
    ▼                         ▼
┌─────────────┐          ┌─────────────┐
│  ImageEncoder│          │  ImageEncoder│  (共享或独立权重)
│   (CNN)     │          │   (CNN)     │
└──────┬──────┘          └──────┬──────┘
       │                        │
       └────────┬───────────────┘
                │ z_concat (2*z_dim)
                ▼
       ┌─────────────────┐
       │   RTify_LSTM    │  (证据累积模块)
       │  (ConvLSTM)     │
       └────────┬────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
 决策 logits       决策时间 (RT)
 (分类结果)        (归一化 0-1)
```

### 2.2 关键模块

#### 2.2.1 ImageEncoder (图像编码器)

**功能**: 将输入图像编码为低维特征向量

**网络结构**:
```python
CNN Backbone:
  Conv2d(1 → 32, kernel=3, stride=2) + BatchNorm + ReLU + Dropout(0.2)
  Conv2d(32 → 64, kernel=3, stride=2) + BatchNorm + ReLU + Dropout(0.2)
  Conv2d(64 → 128, kernel=3, stride=2) + BatchNorm + ReLU + Dropout(0.2)
  Conv2d(128 → 256, kernel=3, stride=2) + BatchNorm + ReLU
  
Global Average Pooling
FC(256 → z_dim)  # 默认 z_dim=128
```

**输入**: `(B, 1, 64, 64)` - 灰度图像批次
**输出**: `(B, z_dim)` - 特征向量

#### 2.2.2 ConvLSTM (卷积LSTM)

**功能**: 处理时序特征，模拟证据随时间的累积过程

**数学原理**:
```
输入门:  i_t = σ(W_i * [x_t, h_{t-1}] + b_i + W_ci ⊙ c_{t-1})
遗忘门:  f_t = σ(W_f * [x_t, h_{t-1}] + b_f + W_cf ⊙ c_{t-1})
候选态:  ĉ_t = tanh(W_c * [x_t, h_{t-1}] + b_c)
细胞态:  c_t = f_t ⊙ c_{t-1} + i_t ⊙ ĉ_t
输出门:  o_t = σ(W_o * [x_t, h_{t-1}] + b_o + W_co ⊙ c_t)
隐藏态:  h_t = o_t ⊙ tanh(c_t)
```

**特点**:
- 使用卷积操作替代全连接，保留空间信息
- 添加peephole连接（W_ci, W_cf, W_co）增强时序依赖

#### 2.2.3 RTify_LSTM (证据累积模块)

**功能**: 核心决策模块，实现证据累积和决策时间预测

**结构**:
```python
输入处理:
  - LayerNorm: 输入归一化（防止tanh饱和）
  - 时序扩展: 将输入重复time_steps次
  - 噪声注入: 可选的dropout和高斯噪声

ConvLSTM处理:
  - 输入: (time_steps, B, 2*z_dim, 1, 1)
  - 输出: (time_steps, B, num_filter, H, W)

证据提取:
  - AdaptiveAvgPool2d → FC → Evidence Network
  - Evidence Network: FC → ReLU → FC → Tanh
  - 输出范围: [-1, 1]（通过Tanh约束）

决策机制:
  - 累积证据: s_accumulated = cumsum(s_traj)
  - 阈值判断: 当 s_accumulated > threshold 时触发决策
  - 可微分决策: DiffDecision.apply()
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `time_steps` | 20 | 最大决策时间步 |
| `threshold` | 6.0 | 决策阈值 |
| `evidence_scale` | 1.0 | 证据缩放因子 |
| `sigma` | 2.0 | 软决策的平滑参数 |

#### 2.2.4 DiffDecision (可微分决策函数)

**功能**: 实现决策时间的可微分计算，支持反向传播

**前向传播**:
```python
decision_time = argmax(trajectory > 0, dim=1)
# 如果没有超过阈值，设为最后一个时间步
```

**反向传播**:
```python
# 使用梯度近似
decision_time ≈ -1 / (dsdt_trajectory + ε)
# 在决策时间点处计算梯度
```

**软决策索引**:
```python
soft_index = exp(-0.5 * (decision_time - arange(T))^2 / sigma^2)
soft_index = soft_index / sum(soft_index)
```

---

## 3. 数据流详解

### 3.1 输入数据

**HumanAlignedDataset** 处理的数据格式：

| 字段 | 说明 | 示例 |
|------|------|------|
| `shape_en` | 形状类型 | 'square', 'triangle', 'circle' |
| `word` | 文字标签 | '方形', '三角', '圆形' |
| `matchness` | 匹配状态 | 'match' / 'mismatch' |
| `rt` | 反应时间 (ms) | 200-1200 |
| `correct` | 是否正确 | True / False |

**数据预处理**:
1. 过滤NaN RT值
2. 过滤RT范围（200-1200 ms）
3. RT归一化到[0, 1]范围
4. 生成形状矩阵（64x64二值图像）
5. 生成符号矩阵（对应形状的符号表示）

### 3.2 前向传播流程

```python
# 1. 编码阶段
z1 = encoder1(img1)  # (B, z_dim)
z2 = encoder2(img2)  # (B, z_dim)
z_concat = cat([z1, z2], dim=1)  # (B, 2*z_dim)

# 2. 时序扩展
z_4d = z_concat.unsqueeze(-1).unsqueeze(-1)  # (B, 2*z_dim, 1, 1)
x_seq = z_4d.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)  # (T, B, C, 1, 1)

# 3. 噪声注入（可选）
if noise_position in ['input', 'both']:
    x_seq = add_noise(x_seq, mask_p, gaussian_std)

# 4. ConvLSTM处理
hidden_states, _ = convlstm(x_seq)  # (T, B, num_filter, H, W)

# 5. 证据提取
pooled = adaptive_avg_pool2d(hidden_states)  # (T, B, num_filter)
logit_trajectory = fc(pooled)  # (T, B, output_size)
s_traj = evidence(pooled) * evidence_scale  # (T, B), range [-1, 1]

# 6. 证据累积
s_accumulated = cumsum(s_traj, dim=1)  # (B, T)

# 7. 决策时间计算
decision_time = DiffDecision.apply(s_accumulated - threshold, dsdt)
decision_time_norm = (decision_time + 1) / time_steps  # 归一化到[0, 1]

# 8. 软决策
soft_index = gaussian_kernel(decision_time)
decision_logits = sum(logit_trajectory * soft_index, dim=1)  # (B, output_size)
```

---

## 4. 训练过程

### 4.1 损失函数

**分类损失** (CrossEntropyLoss):
```python
label_loss = CrossEntropyLoss(decision_logits, labels)
```

**RT损失** (MSELoss, 可选):
```python
rt_loss = MSELoss(rt_pred, rt_target)
```

**速度惩罚** (可选):
```python
speed_loss = speed_penalty * rt_pred.mean()
```

**总损失**:
```python
if use_rt_loss:
    total_loss = rt_loss + label_loss + speed_loss
else:
    total_loss = label_loss + speed_loss
```

### 4.2 训练流程

```python
for epoch in range(num_epochs):
    # 训练阶段
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 前向传播
        decision_logits, rt_pred, _, _ = model(img1, img2)
        
        # 计算损失
        loss = compute_loss(decision_logits, rt_pred, labels, rt)
        
        # 反向传播
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # 评估阶段
    evaluate_on_test_set()
    
    # 检查Fast-Same效应
    if has_significant_fast_same and accuracy >= threshold:
        save_best_model()
```

### 4.3 模型选择标准

**优先级**:
1. **显著性**: Fast-Same效应是否显著（p < 0.05）
2. **效应量**: 效应大小（mismatch_mean - match_mean）
3. **准确性**: 分类准确率

---

## 5. 噪声机制

### 5.1 噪声类型

**输入噪声** (`noise_position='input'`):
```python
# Dropout
mask = bernoulli(ones_like(x) * (1 - mask_p))
x_noisy = x * mask / (1 - mask_p)  # 可选rescale

# 高斯噪声
noise = randn_like(x) * gaussian_std
x_noisy = x_noisy + noise
```

**证据噪声** (`noise_position='evidence'`):
```python
# 在证据轨迹上添加噪声
s_traj = add_noise(s_traj, evidence_mask_p, evidence_noise_std, 
                   rescale_after_dropout=evidence_dropout_rescale)
```

### 5.2 噪声参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mask_p` | 0.0 | Dropout概率 |
| `gaussian_std` | 0.0 | 高斯噪声标准差 |
| `evidence_mask_p` | 0.4 | 证据轨迹dropout概率 |
| `evidence_noise_std` | 0.5 | 证据轨迹高斯噪声标准差 |
| `evidence_dropout_rescale` | False | 是否rescale dropout后的证据 |

---

## 6. 评估与分析

### 6.1 评估指标

**准确性**:
```python
accuracy = (decision_logits.argmax(-1) == labels).float().mean()
```

**RT相关性**:
```python
correlation = corrcoef(rt_pred, rt_human)[0, 1]
```

**Fast-Same效应**:
```python
# 计算匹配/不匹配条件下的平均RT
match_mean = mean(rt_pred[labels == 1])
mismatch_mean = mean(rt_pred[labels == 0])
effect_size = mismatch_mean - match_mean

# t检验
t_stat, p_value = ttest_ind(match_times, mismatch_times)
has_fast_same = effect_size > 0 and p_value < 0.05
```

### 6.2 可视化

**Raincloud Plot**:
- 展示匹配/不匹配条件下的RT分布
- 包含小提琴图、箱线图和散点图
- APA风格格式

**归一化直方图**:
- 比较模型和人类RT分布
- 概率密度形式，便于比较不同样本量

### 6.3 表示距离分析

```python
# 计算编码器输出的欧氏距离
z_dist = norm(z1 - z2, p=2, dim=1)

# 分析匹配/不匹配条件下的距离差异
mean_match_dist = mean(z_dist[labels == 1])
mean_mismatch_dist = mean(z_dist[labels == 0])
delta_dist = mean_mismatch_dist - mean_match_dist
```

---

## 7. 使用示例

### 7.1 基础训练

```bash
python 1_train_rt_matched.py \
    --data_path data/Exp1_postpro.csv \
    --output_dir ./output \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_rt_loss \
    --evidence_noise_std 0.5 \
    --evidence_mask_p 0.4 \
    --time_steps 20 \
    --threshold 6.0
```

### 7.2 使用预训练编码器

```bash
python 1_train_rt_matched.py \
    --data_path data/Exp1_postpro.csv \
    --pretrained_encoder_path models/contrastive_encoder.pth \
    --freeze_encoder True \
    --share_encoder_backbone \
    --use_rt_loss
```

### 7.3 消融研究

```bash
# 仅对比学习（无RT监督）
python 1_train_rt_matched.py \
    --condition_name cl_only \
    --use_rt_loss False

# 仅噪声（无监督）
python 1_train_rt_matched.py \
    --condition_name noise_only \
    --noise_position evidence \
    --use_rt_loss False
```

---

## 8. 代码结构

```
1_train_rt_matched.py
├── 噪声函数
│   └── add_noise()                    # 支持多种噪声类型
│
├── 形状生成器
│   └── ShapeGenerator                 # 生成实验刺激
│       ├── generate_symbol_matrix()   # 生成符号图像
│       ├── generate_square()          # 生成方形
│       ├── generate_triangle()        # 生成三角形
│       └── generate_circle()          # 生成圆形
│
├── 数据集
│   └── HumanAlignedDataset            # 人类行为数据加载
│       ├── __getitem__()              # 返回(img1, img2, label, rt)
│       └── denormalize_rt()           # RT反归一化
│
├── 模型组件
│   ├── ImageEncoder                   # CNN图像编码器
│   ├── ConvLSTM                       # 卷积LSTM
│   ├── DiffDecision                   # 可微分决策函数
│   ├── RTify_LSTM                     # 证据累积模块
│   ├── DualImageContrastiveModel      # 对比学习模型
│   └── EncoderRTifyModel              # 完整模型
│
├── 训练函数
│   └── train_model()                  # 主训练循环
│       ├── 训练阶段
│       ├── 评估阶段
│       └── 最佳模型选择
│
├── 分析函数
│   ├── setup_apa_style()              # APA样式设置
│   ├── create_half_raincloud_plot()   # 半云雨图
│   └── analyze_and_plot_decision_time() # 完整分析
│
└── 主函数
    └── main()                         # 参数解析和运行
```

---

## 9. 关键设计决策

### 9.1 为什么选择ConvLSTM？

- **空间信息保留**: 相比标准LSTM，ConvLSTM通过卷积操作保留空间结构
- **生物合理性**: 模拟大脑皮层中神经元的局部连接模式
- **计算效率**: 对于小空间尺寸（1x1）仍然有效

### 9.2 可微分决策的意义

- **端到端训练**: 决策时间可以作为损失函数的一部分进行优化
- **RT监督**: 允许直接使用人类RT数据进行训练
- **梯度流动**: 通过近似梯度，使阈值穿越操作可微分

### 9.3 噪声的作用

- **增加鲁棒性**: 模拟人类决策的内在变异性
- **防止过拟合**: 作为一种正则化手段
- **生物合理性**: 神经信号 inherently noisy

---

## 10. 扩展与改进方向

### 10.1 可能的扩展

1. **多模态输入**: 整合视觉、听觉等多种感觉模态
2. **更复杂的决策策略**: 引入元认知（confidence）机制
3. **时变阈值**: 实现边界坍塌（collapsing boundary）
4. **注意力机制**: 集成aDDM的注意力组件

### 10.2 性能优化

1. **混合精度训练**: 使用fp16加速训练
2. **数据并行**: 多GPU训练支持
3. **缓存机制**: 预计算和缓存编码器输出

---

## 11. 参考与引用

本模型基于以下工作构建：

1. **证据累积理论**: Ratcliff, R. (1978). A theory of memory retrieval.
2. **神经回路模型**: Wang, X. J. (2002). Probabilistic decision making by slow reverberation in cortical circuits.
3. **RTNet**: Rafiei et al. (2024). The neural network RTNet exhibits the signatures of human perceptual decision-making.
4. **RTify**: arXiv:2411.03630. RTify: A Framework for Aligning Neural Networks with Human Decision-Making Dynamics.
5. **DDM预设检验**: Liu & Hu (2023). Behavioral and cognitive neuroscience findings regarding assumptions of the evidence accumulation model.

---

## 附录：参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z_dim` | int | 128 | 编码器输出维度 |
| `num_filter` | int | 64 | ConvLSTM滤波器数量 |
| `time_steps` | int | 20 | 最大时间步数 |
| `threshold` | float | 6.0 | 决策阈值 |
| `evidence_scale` | float | 1.0 | 证据缩放因子 |
| `sigma` | float | 2.0 | 软决策平滑参数 |
| `evidence_noise_std` | float | 0.0 | 证据噪声标准差 |
| `evidence_mask_p` | float | 0.0 | 证据dropout概率 |
| `lr` | float | 1e-3 | 学习率 |
| `batch_size` | int | 16 | 批次大小 |
| `use_rt_loss` | bool | False | 是否使用RT监督 |
| `speed_penalty` | float | 0.0 | 速度惩罚系数 |
