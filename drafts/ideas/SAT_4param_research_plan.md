# SAT模型微调研究计划

## 研究目标

基于最优的全局threshold模型（exp11），发展出SAT（Speed-Accuracy Trade-off）特异性参数的四参数方案，以更全面地拟合人类在决策任务中的行为数据特征。

## 背景

### 现有模型（exp11）
- **模型架构**：ConvLSTM-based evidence accumulation model
- **训练数据**：MNIST Behavioral Dataset（61,440条样本，无过滤）
- **最优性能**：在不考虑SAT条件的情况下，单一全局threshold（4.28）能够最好地拟合人类行为数据

### 人类数据的SAT特征
基于对人类行为数据的分析：

| 条件 | RT均值 | 准确率 | 样本数 |
|------|--------|--------|--------|
| **speed focus** | 0.855s | 69.2% | 30,720 |
| **accuracy focus** | 1.045s | 71.2% | 30,720 |
| **差异** | -0.189s | -2.1% | - |

关键发现：
- Speed条件下RT比Accuracy条件快约0.19秒
- Speed条件下准确率比Accuracy条件低约2.1%

## 研究假设

1. **Threshold差异假设**：在Speed和Accuracy条件下，threshold应该有不同的值
   - Speed条件：较低的threshold（鼓励快速决策）
   - Accuracy条件：较高的threshold（鼓励准确决策）

2. **Speed Penalty差异假设**：在Speed和Accuracy条件下，speed_penalty应该有不同的值
   - Speed条件：较高的speed_penalty（惩罚过慢决策）
   - Accuracy条件：较低的speed_penalty（允许更多思考时间）

## 四参数方案设计

### 参数定义

| 参数 | 类型 | 初始值 | 说明 |
|------|------|--------|------|
| `threshold_speed` | 可学习 | 4.28 | Speed条件下的决策阈值 |
| `threshold_accuracy` | 可学习 | 4.28 | Accuracy条件下的决策阈值 |
| `speed_penalty_speed` | 固定 | 0.3 | Speed条件的速度惩罚系数 |
| `speed_penalty_accuracy` | 固定 | 0.08 | Accuracy条件的速度惩罚系数 |

### 参数设计理由

#### Threshold设计
- 两个threshold从相同的初始值（4.28，来自exp11）开始
- 通过训练，threshold会自然地向不同方向演化
- Speed条件的threshold预期会下降
- Accuracy条件的threshold预期会上升或保持

#### Speed Penalty设计
- 基于人类数据的RT差异分析
- Speed条件speed_penalty（0.3）约为Accuracy条件（0.08）的3.75倍
- 这个比例反映了Speed条件需要更强的速度激励

## 损失函数设计

### 总损失
```
total_loss = label_loss + rt_loss_weight * rt_loss + speed_loss
```

### 各组成部分

1. **Label Loss**（分类损失）
   - 使用CrossEntropyLoss
   - 基于模型的决策logits和被试的实际反应（response）
   - **注意**：模型学习人类反应（包括错误），而非ground truth

2. **RT Loss**（反应时间损失）
   - 使用MSELoss
   - 预测RT与人类RT的均方误差

3. **Speed Loss**（速度惩罚）
   - 逐样本条件特异性应用
   - Speed条件：`speed_penalty_speed * rt_pred[i]`
   - Accuracy条件：`speed_penalty_accuracy * rt_pred[i]`
   - 最后对所有样本的speed_loss取mean

### 损失权重
- `rt_loss_weight`：2.0
- `speed_penalty_speed`：0.3（固定）
- `speed_penalty_accuracy`：0.08（固定）

## 实验设置

### 数据集
- **来源**：MNIST Behavioral Dataset
- **样本数**：61,440条（30,720 speed focus + 30,720 accuracy focus）
- **训练集**：~49,152条（80%）
- **测试集**：~12,288条（20%）
- **噪声添加**：基于难度（Easy: 0.25, Difficult: 0.4）

### 训练参数
- **Epochs**：25-40（代码默认40，可根据时间调整）
- **Batch Size**：64
- **Learning Rate**：0.001
- **Device**：CPU（MPS存在兼容性问题）

### 训练策略
1. 从exp11模型迁移权重
2. 固定speed_penalty（作为动态属性，不参与梯度更新）
3. 让threshold（threshold_speed, threshold_accuracy）可学习，自然演化
4. 网络权重和threshold一起通过Adam优化器更新

## 预期结果与分析

### 主要观测指标
1. **RT Correlation**：模型预测RT与人类RT的相关性
2. **Accuracy**：分类准确率（与ground truth labels比较，非human responses）
3. **Threshold演化**：threshold_speed和threshold_accuracy的变化趋势
4. **SAT效应复现**：能否复现人类数据中的SAT差异（Speed RT < Accuracy RT）

### 分析维度
1. **总体性能**：整体RT相关性和准确率（vs ground truth labels）
2. **SAT条件对比**：Speed vs Accuracy条件下的RT相关性和准确率差异
3. **难度效应**：Easy vs Difficult条件下的表现差异
4. **Threshold演化**：两个threshold如何向不同方向演化

## 技术实现

### 模型架构修改
在`RTify_ConvLSTM_SAT`模型中：
- 删除原有的单一threshold参数
- 添加`threshold_speed`和`threshold_accuracy`两个可学习参数（nn.Parameter）
- 添加SAT条件到threshold的映射逻辑
- `speed_penalty_speed`和`speed_penalty_accuracy`作为动态属性（非nn.Parameter），不参与梯度更新

### 关键代码实现

```python
# SAT条件下的threshold选择
def _get_threshold_batch(self, sat_conditions, batch_size, device):
    thresholds = []
    for sat in sat_conditions:
        if isinstance(sat, str):
            if 'accuracy' in sat.lower():
                thresholds.append(self.threshold_accuracy)
            else:
                thresholds.append(self.threshold_speed)
        else:
            thresholds.append(self.threshold_speed)
    return torch.stack(thresholds)

# 条件特异性speed_penalty应用（逐样本计算）
speed_losses = []
for i, sat in enumerate(sat_conditions):
    if 'accuracy' in sat.lower():
        speed_losses.append(speed_penalty_accuracy * rt_pred[i])
    else:
        speed_losses.append(speed_penalty_speed * rt_pred[i])
speed_loss = torch.stack(speed_losses).mean()
```

## 文件结构

### 相关脚本
- `transfer_to_4param_sat.py`：权重迁移脚本（exp11 → 4-param SAT）
- `train_sat_4param.py`：四参数SAT模型训练脚本

### 执行流程
```bash
# Step 1: 从exp11迁移权重到4-param SAT模型
python src/experiments/mnist_convlstm/transfer_to_4param_sat.py \
  --exp11_path outputs/experiments/mnist_convlstm/exp11_t40/convlstm_balanced_rt2.0_sp0.1_ep70.pth \
  --output_path outputs/experiments/mnist_convlstm/exp11_t40/convlstm_4param_sat.pth

# Step 2: 训练4-param SAT模型
python src/experiments/mnist_convlstm/train_sat_4param.py \
  --pretrained_path outputs/experiments/mnist_convlstm/exp11_t40/convlstm_4param_sat.pth \
  --data_path "data/raw/rtnet/behavioral data.csv" \
  --output_dir outputs/experiments/mnist_convlstm/exp_sat_4param \
  --epochs 40 \
  --device cpu
```

### 模型检查点
- 输入：`outputs/experiments/mnist_convlstm/exp11_t40/convlstm_4param_sat.pth`（从exp11迁移）
- 输出：`outputs/experiments/mnist_convlstm/exp_sat_4param/convlstm_4param_sat_ep{epochs}_spd{sp}_acc{acc}.pth`

## 研究意义

1. **理论验证**：验证threshold和speed_penalty在SAT条件下的差异化假设
2. **模型优化**：提升模型对人类行为的拟合能力
3. **认知建模**：为理解人类决策的速度-准确性权衡提供计算框架

## 后续计划

1. **短期**：
   - 完成25-40 epochs训练
   - 分析threshold演化趋势
   - 评估SAT效应复现程度

2. **中期**：
   - 尝试不同的speed_penalty初始值组合
   - 探索threshold和speed_penalty的联动关系

3. **长期**：
   - 将成功经验推广到其他数据集
   - 探索更复杂的SAT模型架构
