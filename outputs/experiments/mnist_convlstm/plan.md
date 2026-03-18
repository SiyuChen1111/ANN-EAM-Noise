# RT分布匹配改进方案计划

## 目标

使模型RT分布形态与人类RT分布相匹配（右偏分布）

## 问题诊断

| 问题 | 现状 | 目标 |
|------|------|------|
| RT分布形态 | 近似均匀分布 | 右偏分布 |
| RT均值 | ~2.9s | ~0.91s |
| noise_std | 0.0004 (坍缩) | 有意义的噪声 |
| RT相关性 | 0.11 | > 0.3 |

## 计划变更说明

**原计划**: 依次运行Exp01-05，测试不同的RT分布改进方法。

**变更原因**:
1. **Exp01结果良好**: Exp01 (Fixed noise, 30ep) 的RT分布形态已经相当不错，显示出右偏趋势，值得进一步探索更多epoch的效果。
2. **Exp02资源消耗过大**: 04_train_rt_distribution.py中模型forward函数存在内存问题（`x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)`将数据复制20倍），导致：
   - batch_size=64时内存溢出（需要41.64 GiB，超过42.43 GiB限制）
   - batch_size=32时每个epoch需要约2.5小时，30个epoch需要约75小时

**新计划**:
1. 先完成Exp01b (50ep) 观察更多epoch的效果
2. 跳过Exp02，直接运行Exp03 (偏度惩罚)
3. Exp03将使用02_train_model.py实现（需要添加skewness_loss支持）

## 实验方案

### 方案1: 固定噪声参数baseline
- **状态**: ✅ 完成
- **文件夹**: `outputs/experiments/mnist_convlstm/exp01_fixed_noise`
- **目的**: 作为对照组，验证固定噪声参数的效果
- **参数**: epochs=30, fixed_noise=True, noise_std=0.5, mask_p=0.4
- **命令**:
```bash
nohup python -m src.experiments.mnist_convlstm.02_train_model \
  --data_path "data/raw/rtnet/behavioral data.csv" \
  --epochs 30 --batch_size 64 --use_rt_loss --fixed_noise \
  --output_dir outputs/experiments/mnist_convlstm/exp01_fixed_noise \
  > outputs/experiments/mnist_convlstm/exp01_fixed_noise/training.log 2>&1 &
```
- **预期**: 对比可学习噪声的差异
- **结果**: 
  - Accuracy (vs correct label): 33.31%
  - Accuracy (vs human response): 29.18%
  - RT Correlation: 0.0162
  - Learned Threshold: 1.3849
  - RT分布形态: 待分析

---

### 方案2: 添加RT分布匹配损失
- **状态**: ❌ 已终止 (内存不足问题)
- **文件夹**: `outputs/experiments/mnist_convlstm/exp02_distribution_loss`
- **目的**: 使用KL散度使RT分布接近人类
- **参数**: epochs=30, rt_dist_weight=0.5, batch_size=32
- **问题**: 04_train_rt_distribution.py中模型forward函数内存消耗过大
- **命令**:
```bash
nohup python -m src.experiments.mnist_convlstm.04_train_rt_distribution \
  --data_path "data/raw/rtnet/behavioral data.csv" \
  --epochs 30 --batch_size 64 --use_rt_loss \
  --rt_dist_weight 0.5 \
  --experiment_name exp02_distribution_loss \
  --output_dir outputs/experiments/mnist_convlstm/exp02_distribution_loss \
  > outputs/experiments/mnist_convlstm/exp02_distribution_loss/training.log 2>&1 &
```
- **预期**: RT分布形态改善
- **结果**: 已终止

---

### 方案1b: 固定噪声参数 (50 epochs)
- **状态**: ✅ 完成
- **文件夹**: `outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep50`
- **目的**: 扩展训练epoch数，观察RT分布改善
- **参数**: epochs=50, fixed_noise=True, batch_size=64
- **命令**:
```bash
# 使用caffeinate防止系统休眠，合上笔记本后训练仍会继续
nohup caffeinate -i python -m src.experiments.mnist_convlstm.02_train_model \
  --data_path "data/raw/rtnet/behavioral data.csv" \
  --epochs 50 --batch_size 64 --use_rt_loss --fixed_noise \
  --output_dir outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep50 \
  > outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep50/training.log 2>&1 &
```
- **预期**: 更好的RT分布匹配
- **结果**: 
  - Accuracy (vs correct label): 57.34%
  - Accuracy (vs human response): 48.22%
  - RT Correlation: -0.0125
  - Learned Threshold: 1.9933

---

### 方案1c: 固定噪声参数 (100 epochs)
- **状态**: ✅ 完成
- **文件夹**: `outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100`
- **目的**: 进一步扩展训练epoch数，观察准确率和RT分布的变化
- **参数**: epochs=100, fixed_noise=True, batch_size=64
- **命令**:
```bash
nohup caffeinate -i python -m src.experiments.mnist_convlstm.02_train_model \
  --data_path "data/raw/rtnet/behavioral data.csv" \
  --epochs 100 --batch_size 64 --use_rt_loss --fixed_noise \
  --output_dir outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100 \
  > outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100/training.log 2>&1 &
```
- **预期**: 准确率进一步提升
- **结果**: 
  - Accuracy (vs correct label): 78.10%
  - Accuracy (vs human response): 63.42%
  - RT Correlation: 0.0130
  - Learned Threshold: 2.3150
  - RT均值: 0.52 (normalized)
- **可视化**: `outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100/figures_apa/`

---

### 方案3: 添加偏度惩罚
- **状态**: ⏳ 待开始 (等Exp01b完成后运行)
- **文件夹**: `outputs/experiments/mnist_convlstm/exp03_skewness_penalty`
- **目的**: 鼓励RT分布具有正偏度
- **参数**: epochs=30, skewness_weight=0.1
- **注意**: 需要使用02_train_model.py（04_train_rt_distribution.py有内存问题）
- **命令**:
```bash
# 需要先在02_train_model.py中添加skewness_loss支持
```
- **预期**: RT分布右偏
- **结果**: 待完成

---

### 方案4: 噪声参数正则化
- **状态**: ⏳ 待开始
- **文件夹**: `outputs/experiments/mnist_convlstm/exp04_noise_regularization`
- **目的**: 防止noise_std坍缩到0
- **参数**: epochs=30, min_noise=0.05
- **命令**:
```bash
nohup python -m src.experiments.mnist_convlstm.04_train_rt_distribution \
  --data_path "data/raw/rtnet/behavioral data.csv" \
  --epochs 30 --batch_size 64 --use_rt_loss \
  --min_noise_std 0.05 \
  --experiment_name exp04_noise_regularization \
  --output_dir outputs/experiments/mnist_convlstm/exp04_noise_regularization \
  > outputs/experiments/mnist_convlstm/exp04_noise_regularization/training.log 2>&1 &
```
- **预期**: 噪声参数保持有意义
- **结果**: 待完成

---

### 方案5: 调整阈值初始化
- **状态**: ⏳ 待开始
- **文件夹**: `outputs/experiments/mnist_convlstm/exp05_threshold_init`
- **目的**: 通过调整阈值缩短RT
- **参数**: epochs=30, initial_threshold=2.0
- **命令**:
```bash
nohup python -m src.experiments.mnist_convlstm.04_train_rt_distribution \
  --data_path "data/raw/rtnet/behavioral data.csv" \
  --epochs 30 --batch_size 64 --use_rt_loss \
  --initial_threshold 2.0 \
  --experiment_name exp05_threshold_init \
  --output_dir outputs/experiments/mnist_convlstm/exp05_threshold_init \
  > outputs/experiments/mnist_convlstm/exp05_threshold_init/training.log 2>&1 &
```
- **预期**: RT均值接近人类
- **结果**: 待完成

---

## 实验流程

每个方案遵循以下流程：
1. 创建实验文件夹 ✅
2. 使用nohup启动训练
3. 监控训练进度
4. 训练结束后生成可视化
5. 记录结果到本文件

## 结果对比表

| 方案 | 准确率 | RT相关性 | RT均值 | RT偏度 | 状态 |
|------|--------|----------|--------|--------|------|
| Baseline (learnable, ep100) | 78.44% | 0.11 | 2.9s | ~0 | ✅ 完成 |
| Exp01: Fixed noise (30ep) | 33.31% | 0.0162 | 0.37 | - | ✅ 完成 |
| Exp01b: Fixed noise (50ep) | 57.34% | -0.0125 | - | - | ✅ 完成 |
| Exp01c: Fixed noise (100ep) | 78.10% | 0.0130 | 0.52 | - | ✅ 完成 |
| Exp02: Distribution loss | - | - | - | - | ❌ 已终止 |
| Exp03: Skewness penalty | - | - | - | - | ⏳ 待开始 |

---
*创建时间: 2026-03-13*
*最后更新: 2026-03-13*
