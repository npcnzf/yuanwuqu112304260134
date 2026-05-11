# 手写数字识别 - CNN实验项目

基于PyTorch的CNN手写数字识别项目,实现从模型训练到Web应用部署的完整流程。

## 项目结构

```
.
├── train.csv              # 训练数据集
├── test.csv               # 测试数据集
├── sample_submission.csv  # 提交样例
├── requirements.txt       # Python依赖包
├── data_loader.py         # 数据加载和预处理
├── model.py              # CNN模型定义
├── train.py              # 完整训练脚本(含对比实验)
├── train_fast.py         # 快速训练脚本(仅最终模型)
├── app.py                # Gradio Web应用
└── README.md             # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选,用于加速训练)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 训练模型

#### 快速训练(推荐)

直接训练最终的高性能模型:

```bash
python train_fast.py
```

这将:
- 使用数据增强
- 使用Adam优化器(lr=0.001)
- 使用早停策略
- 训练完成后保存模型为 `model_best.pth`
- 生成测试集预测文件 `submission.csv`

#### 完整训练(含对比实验)

运行所有4组对比实验:

```bash
python train.py
```

这将:
- 运行4组超参数对比实验
- 保存每个实验的模型
- 生成Loss对比曲线图 `loss_comparison.png`
- 训练最终的高性能模型

### 2. 启动Web应用

训练完成后,启动Gradio Web应用:

```bash
python app.py
```

应用包含两个功能标签页:
- **画板绘制**: 直接在网页上绘制数字进行识别
- **图片上传**: 上传手写数字图片进行识别

## 模型结构

```
输入(1×28×28)
  ↓
Conv2d(32 filters, 3×3) + ReLU + MaxPool(2×2)
  ↓
Conv2d(64 filters, 3×3) + ReLU + MaxPool(2×2)
  ↓
Dropout(0.25)
  ↓
Flatten → FC(128) + ReLU + Dropout(0.5)
  ↓
FC(10) + LogSoftmax
```

## 超参数配置

### 最佳配置(达到0.988+准确率)

- 优化器: Adam
- 学习率: 0.001
- Batch Size: 64
- 数据增强: 是(RandomRotation, RandomAffine)
- 早停: 是(patience=10)
- Dropout: 0.25, 0.5

### 对比实验

| 实验 | 优化器 | 学习率 | Batch Size | 数据增强 | 早停 |
|------|--------|--------|------------|----------|------|
| Exp1 | SGD | 0.01 | 64 | 否 | 否 |
| Exp2 | Adam | 0.001 | 64 | 否 | 否 |
| Exp3 | Adam | 0.001 | 128 | 否 | 是 |
| Exp4 | Adam | 0.001 | 64 | 是 | 是 |

## 实验结果

运行完整训练后,将获得:
- 4组实验的训练/验证Loss曲线
- 各实验的准确率对比
- 最终模型的测试集预测结果(submission.csv)

## Web应用功能

### 画板绘制
- 支持直接在浏览器中绘制数字
- 实时显示识别结果和置信度
- 显示Top 3预测结果
- 显示预处理后的图片
- 支持清除画板重新绘制

### 图片上传
- 支持上传包含手写数字的图片
- 自动预处理和识别
- 显示识别结果和置信度

## 性能指标

- 验证集准确率: ≥98.8%
- 测试集准确率: ≥0.988 (Kaggle评分)

## 常见问题

### GPU使用
如果有NVIDIA GPU和CUDA环境,训练将自动使用GPU加速。

### 模型加载失败
确保先运行训练脚本生成 `model_best.pth` 文件。

### 数据文件位置
确保train.csv、test.csv、sample_submission.csv在项目根目录。

## 作者

机器学习实验项目
