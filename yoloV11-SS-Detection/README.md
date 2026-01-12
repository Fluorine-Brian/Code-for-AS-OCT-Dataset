# AS-OCT 巩膜突关键点检测

基于 YOLOv11 Pose 的 AS-OCT 图像巩膜突自动定位系统，支持 4 类眼科疾病分类。

## 项目特点

- **关键点检测**: 精准定位左右巩膜突位置
- **疾病分类**: 自动识别正常、白内障、青光眼、青光眼合并白内障
- **医学评估**: 提供像素级精度评估和临床适用性分析
- **可视化**: 完整的训练曲线、混淆矩阵和预测误差可视化

## 环境安装

```bash
conda create -n yolo11 python=3.11 -y
conda activate yolo11
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install ultralytics scikit-learn matplotlib
```

## 数据集准备

### 原始数据结构

```
datasets/
├── Cataract/
│   ├── Original Images/
│   └── Annotated Images/
├── Normal/
│   ├── Original Images/
│   └── Annotated Images/
├── Glaucoma/
│   ├── Original Images/
│   └── Annotated Images/
└── Glaucoma_Cataract/
    ├── Original Images/
    └── Annotated Images/
```

### 标注要求

LabelMe JSON 文件需包含：
- 关键点: `left_scleral_spur`, `right_scleral_spur`
- 辅助区域: `lens`, `nucleus`, `iris`, `anterior_chamber`

### 数据转换

将 LabelMe 格式转换为 YOLO 格式：

```bash
python convert_asoct_to_yolo_pose.py
```

转换后生成:
```
datasets/ASOCT_YOLO/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 快速开始

### 1. 选择模型

编辑 `train_asoct_pose.py` 第 78 行：

```python
model_name = 'yolo11l-pose.pt'  # 可选: n, s, m, l, x
```

模型对比：

| 模型 | 参数量 | 精度 | 速度 | 推荐场景 |
|------|--------|------|------|---------|
| yolo11n-pose | 2.9M | ★★ | ⚡⚡⚡⚡⚡ | 快速测试 |
| yolo11s-pose | 9.9M | ★★★ | ⚡⚡⚡ | 移动端部署 |
| yolo11m-pose | 21M | ★★★★ | ⚡⚡ | 平衡性能 |
| yolo11l-pose | 26M | ★★★★ | ⚡⚡ | 高精度需求 |
| yolo11x-pose | 59M | ★★★★★ | ⚡ | 最高精度 |

### 2. 训练模型

```bash
python train_asoct_pose.py
```

训练输出目录: `runs/pose/asoct_yolo11{model_size}_4class/`

包含文件:
- `weights/best.pt` - 最佳模型权重
- `weights/last.pt` - 最后轮次权重
- `results.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵
- `val_batch*.jpg` - 验证集预测可视化

### 3. 模型评估

**YOLO 标准评估**:
```bash
python validate_asoct_pose.py
```

**像素距离评估**:
```bash
python evaluate_pixel_distance.py
```

生成评估报告:
- 整体评估: MPD、PCK@Xpx、标准差、中位数
- 按类别评估: 每个疾病类别的独立性能分析
- 可视化图表: `runs/pose/evaluation_results.png`

**误差可视化**:
```bash
python visualize_prediction_errors.py
```

生成 40 张可视化图片，每类 10 张，保存至 `runs/pose/visualize_errors/`

### 4. 模型预测

```bash
python predict_asoct_pose.py
```

## 数据集信息

### 疾病类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | normal | 正常眼部 |
| 1 | cataract | 白内障 |
| 2 | glaucoma | 青光眼 |
| 3 | glaucoma_cataract | 青光眼合并白内障 |

### 关键点定义

- `left_scleral_spur` - 左侧巩膜突
- `right_scleral_spur` - 右侧巩膜突

## 评估指标说明

### 医学关键指标

**平均像素距离 MPD**:
- 优秀: < 5 像素
- 良好: < 10 像素
- 可用: < 20 像素

**关键点准确率 PCK**:
- PCK@5px: 高精度临床诊断标准
- PCK@10px: 临床辅助诊断标准，>95% 为优秀
- PCK@20px: 临床可接受精度，>90% 为良好
- PCK@50px: 研究初筛标准

### YOLO 检测指标

- **mAP50**: IoU=0.5 时的平均精度
- **mAP50-95**: IoU=0.5 到 0.95 的平均精度
- **box**: 边界框检测性能
- **pose**: 关键点检测性能，>0.80 为良好

## 损失函数配置

`train_asoct_pose.py` 中的损失权重设置：

```python
'box': 0,        # 边界框损失权重
'cls': 0.3,      # 分类损失权重
'pose': 15.0,    # 关键点损失权重
'kobj': 1.0,     # 关键点置信度权重
```

设计理念: 关键点检测为主要任务，疾病分类为辅助任务。

## Python API 使用

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/pose/asoct_yolo11l_4class/weights/best.pt')

# 对单张图片进行预测
results = model.predict('image.jpg', conf=0.25)

# 获取关键点坐标
keypoints = results[0].keypoints.xy[0]
left_spur = keypoints[0]   # 左侧巩膜突 [x, y]
right_spur = keypoints[1]  # 右侧巩膜突 [x, y]

# 获取疾病类别
class_id = int(results[0].boxes.cls[0])
class_names = ['normal', 'cataract', 'glaucoma', 'glaucoma_cataract']
disease = class_names[class_id]

# 获取置信度
confidence = float(results[0].boxes.conf[0])

print(f"疾病类别: {disease}, 置信度: {confidence:.2f}")
print(f"左侧巩膜突: {left_spur}")
print(f"右侧巩膜突: {right_spur}")
```

## 命令行使用

训练模型:
```bash
yolo pose train data=datasets/asoct-pose-4class.yaml model=yolo11l-pose.pt epochs=150
```

验证模型:
```bash
yolo pose val model=runs/pose/asoct_yolo11l_4class/weights/best.pt data=datasets/asoct-pose-4class.yaml
```

批量预测:
```bash
yolo pose predict model=runs/pose/asoct_yolo11l_4class/weights/best.pt source=path/to/images
```

导出模型:
```bash
yolo export model=runs/pose/asoct_yolo11l_4class/weights/best.pt format=onnx
```

## 项目结构

```
./
├── datasets/
│   ├── Cataract/              原始标注数据
│   ├── Normal/
│   ├── Glaucoma/
│   ├── Glaucoma_Cataract/
│   ├── asoct-pose-4class.yaml 数据集配置文件
│   └── ASOCT_YOLO/            转换后的 YOLO 格式数据
│
├── runs/pose/
│   └── asoct_yolo11l_4class/  训练输出目录
│       ├── weights/
│       │   ├── best.pt        最佳模型
│       │   └── last.pt        最后模型
│       ├── results.png        训练曲线
│       └── ...
│
├── yolo11l-pose.pt            预训练权重
│
├── convert_asoct_to_yolo_pose.py      数据格式转换
├── train_asoct_pose.py                模型训练
├── validate_asoct_pose.py             YOLO 标准验证
├── evaluate_pixel_distance.py         像素距离评估
├── visualize_prediction_errors.py     误差可视化
└── predict_asoct_pose.py              模型预测
```

## 核心脚本说明

| 脚本 | 功能 | 配置位置 |
|------|------|---------|
| `convert_asoct_to_yolo_pose.py` | 数据格式转换 | - |
| `train_asoct_pose.py` | 模型训练 | 第 78 行: model_name |
| `validate_asoct_pose.py` | YOLO 验证 | 第 34 行: model_size |
| `evaluate_pixel_distance.py` | 像素距离评估 | 第 467 行: model_size |
| `visualize_prediction_errors.py` | 误差可视化 | 第 223 行: model_size |
| `predict_asoct_pose.py` | 模型预测 | 第 17 行: model_size |

所有脚本使用统一的 `model_size` 配置，修改一处自动同步。

## 临床应用建议

根据不同应用场景选择合适的精度标准:

**临床诊断**: MPD < 5px, PCK@10px > 95%
- 推荐模型: yolo11l-pose 或 yolo11x-pose
- 适用于临床诊断决策支持

**辅助诊断**: MPD < 10px, PCK@20px > 90%
- 推荐模型: yolo11m-pose 或 yolo11l-pose
- 适用于术前评估和随访监测

**研究初筛**: MPD < 20px, PCK@50px > 85%
- 推荐模型: yolo11s-pose 或 yolo11m-pose
- 适用于大规模人群筛查

## 参考文档

[Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com)
