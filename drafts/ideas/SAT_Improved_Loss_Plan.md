# SAT模型改进Loss设计方案

## 1. 问题分析

### 1.1 当前遇到的问题

**问题表现：**
- 模型学习人类response（包含错误），但准确率远低于人类水平（~19% vs ~70%）
- RT相关性有所提升（0.17-0.20），但准确率无法提升
- 之前可学习threshold训练中，threshold演化方向不符合预期

**问题定位：**
- Threshold设置问题：极端值（2.5/8.0）导致决策时机不对
- Loss设计问题：可能没有给模型正确的学习信号
- 需要进一步分析确定主要原因

### 1.2 关键发现

1. **Threshold直接影响decision_logits**
   ```python
   decision_time = DiffDecision.apply(s_accumulated - threshold, ...)
   soft_index = exp(-0.5 * (decision_time - t)**2 / sigma**2)
   decision_logits = (logit_trajectory * soft_index).sum()
   ```
   - Threshold不是只影响RT，它直接影响决策时选取哪个时间步的logits
   - Th太低 → 决策太早 → 证据不足 → logits质量差
   - Th太高 → 可能超时 → 被迫在最后时刻决策

2. **当前Loss设计的问题**
   ```python
   loss = label_loss + rt_loss_weight * rt_loss + speed_penalty * rt
   ```
   - 所有条件使用相同的label_loss权重
   - RT loss没有区分条件
   - Speed_penalty只是简单的线性惩罚

---

## 2. 解决方案

### 2.1 核心思路

**模拟人类决策过程：**
```
实验指令（外部压力）→ 被试调整策略 → 行为变化
        ↓                  ↓              ↓
   Speed_penalty      Threshold调整    RT/Acc变化
```

**正确的关系：**
- Speed condition → 高SP → 模型学习降低Th → 快速决策
- Accuracy condition → 低SP → 模型学习提高Th → 准确决策

### 2.2 改进的Loss设计

#### 2.2.1 条件特定的权重

| 条件 | Label Loss权重 | RT Loss权重 | Speed Penalty |
|------|----------------|-------------|---------------|
| Speed | 1.0 | 3.0 | +0.2 (惩罚长RT) |
| Accuracy | 2.0 | 1.0 | -0.05 (鼓励稍长RT) |

**设计理由：**
- Speed条件：更关注RT拟合，Acc相对次要
- Accuracy条件：更关注Acc，RT相对次要

#### 2.2.2 Threshold分化正则化

```python
th_diff = threshold_accuracy - threshold_speed
th_diff_loss = torch.relu(2.0 - th_diff)  # 确保差异至少为2
```

**目的：** 防止两个threshold收敛到相同值

#### 2.2.3 完整的Loss公式

```python
total_loss = (
    w_label_speed * label_loss_speed + 
    w_label_acc * label_loss_acc +
    w_rt_speed * rt_loss_speed + 
    w_rt_acc * rt_loss_acc +
    speed_penalty_loss +
    th_diff_weight * th_diff_loss
)
```

### 2.3 超参数设置

```python
# 条件权重
W_LABEL_SPEED = 1.0    # Speed条件的label loss权重
W_LABEL_ACC = 2.0      # Accuracy条件的label loss权重
W_RT_SPEED = 3.0       # Speed条件的RT loss权重
W_RT_ACC = 1.0         # Accuracy条件的RT loss权重

# Speed penalty
SP_SPEED = 0.2         # Speed条件惩罚系数（惩罚长RT）
SP_ACC = -0.05         # Accuracy条件系数（鼓励稍长RT）

# Threshold正则化
TH_DIFF_TARGET = 2.0   # 目标差异
TH_DIFF_WEIGHT = 0.1   # 正则化权重

# 学习率（统一）
LR = 0.001

# Threshold初始值
TH_INIT = 4.28         # 从exp11的最佳单一threshold开始
```

---

## 3. 训练配置

### 3.1 模型配置

| 参数 | 值 |
|------|-----|
| 预训练权重 | exp11 (convlstm_balanced_rt2.0_sp0.1_ep70.pth) |
| Threshold初始值 | Speed=4.28, Accuracy=4.28 |
| Threshold类型 | 可学习 |
| Time steps | 40 |
| Device | CPU |

### 3.2 数据配置

| 参数 | 值 |
|------|-----|
| 数据集 | MNIST Behavioral (Log Normalization) |
| 训练/测试比例 | 80%/20% |
| Batch size | 64 |
| Label目标 | response（人类反应） |

### 3.3 训练策略

| 参数 | 值 |
|------|-----|
| Epochs | 20（测试性训练） |
| Learning rate | 0.001 |
| 优化器 | Adam |
| 评估频率 | 每个epoch |

---

## 4. 评估指标

### 4.1 主要指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Speed Acc | Speed条件下的准确率 | ~65-70% |
| Accuracy Acc | Accuracy条件下的准确率 | ~70-75% |
| Speed RT Corr | Speed条件下的RT相关性 | >0.15 |
| Accuracy RT Corr | Accuracy条件下的RT相关性 | >0.15 |
| RT差异 | Accuracy RT - Speed RT | ~0.19s (接近人类) |

### 4.2 监控指标

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| Th_Speed | Speed threshold | 逐渐降低 |
| Th_Acc | Accuracy threshold | 逐渐升高 |
| Th差异 | Th_Acc - Th_Speed | 保持>2.0 |
| 总Loss | 训练loss | 逐渐降低 |

---

## 5. 预期结果与风险

### 5.1 预期改进

1. **准确率提升** - 从~19%提升到~60-70%
2. **RT相关性保持** - 维持在0.15-0.20
3. **Threshold正确分化** - Speed降低，Accuracy升高

### 5.2 潜在风险

1. **Acc与RT Corr的trade-off** - 可能需要在两者之间权衡
2. **Threshold不稳定** - 可能需要调整正则化权重
3. **训练时间** - CPU训练较慢，需要耐心

### 5.3 后续方案

如果改进的Loss设计效果不理想：
1. 调整超参数权重
2. 尝试不同的threshold初始值
3. 考虑引入强化学习框架

---

## 6. 实施步骤

1. ✅ 创建计划文档
2. ⬜ 创建改进Loss的训练脚本
3. ⬜ 运行测试性训练（20 epochs）
4. ⬜ 分析结果，决定是否继续训练
5. ⬜ 根据结果调整方案

---

*创建时间: 2026-03-20*
*状态: Planning*
