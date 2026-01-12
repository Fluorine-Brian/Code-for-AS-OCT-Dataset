"""
YOLOv11 Pose 训练脚本
用于AS-OCT (Anterior Segment OCT) 图像的巩膜突关键点检测
支持4个类别分类训练: Normal, Cataract, Glaucoma, Glaucoma_Cataract
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
from collections import Counter

def load_dataset_info(yaml_path):
    """从YAML配置文件加载数据集信息"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config['path'])
    train_img_dir = dataset_path / config['train']
    val_img_dir = dataset_path / config['val']

    # 统计图片数量
    train_images = list(train_img_dir.glob('*.jpg')) if train_img_dir.exists() else []
    val_images = list(val_img_dir.glob('*.jpg')) if val_img_dir.exists() else []

    # 统计各类别数量（从文件名前缀识别）
    class_counts = Counter()
    for img in train_images + val_images:
        # 文件名格式: normal_1.jpg, cataract_327.jpg, glaucoma_cataract_1000.jpg
        # 需要识别复合类别名（如 glaucoma_cataract）
        stem = img.stem
        # 从文件名中提取类别（最后一个下划线之前的部分）
        prefix = stem.rsplit('_', 1)[0]  # 从右侧分割，保留复合类别名
        class_counts[prefix] += 1

    return {
        'names': config.get('names', {}),
        'kpt_shape': config.get('kpt_shape', [2, 3]),
        'kpt_names': config.get('kpt_names', {}),
        'train_count': len(train_images),
        'val_count': len(val_images),
        'total_count': len(train_images) + len(val_images),
        'class_counts': class_counts,
        'dataset_exists': train_img_dir.exists() and val_img_dir.exists()
    }

def main():
    print("=" * 70)
    print("YOLOv11 Pose 训练 - AS-OCT 巩膜突关键点检测")
    print("=" * 70)

    # 数据集配置文件
    data_yaml = 'datasets/asoct-pose-4class.yaml'

    # 加载数据集信息
    print("\n正在加载数据集信息...")
    try:
        dataset_info = load_dataset_info(data_yaml)
        if not dataset_info['dataset_exists']:
            print(f"\n错误: 数据集不存在!")
            print(f"请先运行: python convert_asoct_to_yolo_pose.py")
            return
    except Exception as e:
        print(f"\n错误: 无法加载数据集配置 - {str(e)}")
        print(f"请检查文件是否存在: {data_yaml}")
        return

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        device = 0
    else:
        print("\n警告: CUDA不可用，使用CPU训练 (速度会很慢)")
        device = 'cpu'

    # 加载预训练模型
    # 可选模型: yolo11n-pose.pt, yolo11s-pose.pt, yolo11m-pose.pt, yolo11l-pose.pt, yolo11x-pose.pt
    # n(nano) < s(small) < m(medium) < l(large) < x(xlarge)
    # 推荐: 开始用n或s快速验证，效果好再用m或l
    model_name = 'yolo11l-pose.pt'  # 使用large模型，平衡速度和精度
    print(f"\n加载模型: {model_name}")

    model = YOLO(model_name)

    # 从模型名称提取模型大小 (例如: yolo11l-pose.pt -> l)
    model_size = model_name.replace('yolo11', '').replace('-pose.pt', '')  # 提取 'l', 'x', 's' 等
    experiment_name = f'asoct_yolo11{model_size}_4class'  # 自动生成实验名称

    # 训练参数
    print("\n训练配置:")
    print("-" * 70)

    config = {
        'data': data_yaml,                      # 数据集配置文件
        'epochs': 150,                          # 训练轮数
        'imgsz': 640,                           # 输入图片大小
        'batch': 16,                            # 批次大小
        'device': device,                       # GPU设备
        'workers': 8,                           # 数据加载线程数
        'project': 'runs/pose',                 # 项目保存路径
        'name': experiment_name,                # 实验名称（自动匹配模型）
        'exist_ok': False,                      # 是否覆盖已存在的实验
        'pretrained': True,                     # 使用预训练权重
        'optimizer': 'auto',                    # 优化器
        'lr0': 0.01,                            # 初始学习率
        'lrf': 0.01,                            # 最终学习率
        'momentum': 0.937,                      # SGD动量
        'weight_decay': 0.0005,                 # 权重衰减
        'warmup_epochs': 3.0,                   # 预热轮数
        'warmup_momentum': 0.8,                 # 预热初始动量
        'warmup_bias_lr': 0.1,                  # 预热偏置学习率
        'box': 0,                               # 边界框损失权重
        'cls': 0.3,                             # 分类损失权重
        'pose': 15.0,                           # 关键点损失权重
        'kobj': 1.0,                            # 关键点目标损失权重
        'label_smoothing': 0.0,                 # 标签平滑
        'save': True,                           # 保存检查点
        'save_period': -1,                      # 每N轮保存一次
        'val': True,                            # 训练期间验证
        'plots': True,                          # 保存训练曲线和预测可视化
        'patience': 50,                         # 早停耐心值
        'resume': False,                        # 是否从上次中断处恢复训练
        'amp': True,                            # 自动混合精度训练
        'fraction': 1.0,                        # 使用数据集的比例
        'profile': False,                       # 性能分析
        'freeze': None,                         # 冻结层数
        'multi_scale': False,                   # 多尺度训练
        'single_cls': False,                    # 多类训练
        'rect': False,                          # 矩形训练
        'cos_lr': False,                        # 余弦学习率调度
        'close_mosaic': 10,                     # 最后N轮关闭mosaic增强
        'overlap_mask': True,                   # 训练时mask可重叠
        'mask_ratio': 4,                        # mask下采样率
        'dropout': 0.0,                         # Dropout率
        'cache': False,                         # 缓存图片到内存
        'verbose': True,                        # 详细输出
    }

    # 打印配置
    for key, value in config.items():
        print(f"  {key:20s}: {value}")

    print("-" * 70)
    print(f"\n数据集信息:")
    print(f"  总样本数: {dataset_info['total_count']} 张")
    print(f"  训练集: {dataset_info['train_count']} 张 ({dataset_info['train_count']/dataset_info['total_count']*100:.1f}%)")
    print(f"  验证集: {dataset_info['val_count']} 张 ({dataset_info['val_count']/dataset_info['total_count']*100:.1f}%)")
    print(f"  类别: {len(dataset_info['names'])} ({', '.join(dataset_info['names'].values())})")
    print(f"  关键点: {dataset_info['kpt_shape'][0]} ({', '.join(dataset_info['kpt_names'].get(0, []))})")

    print("\n类别分布:")
    total = dataset_info['total_count']
    for class_name, count in sorted(dataset_info['class_counts'].items()):
        percentage = (count / total) * 100
        print(f"  {class_name:20s}: {count:4d} 张 ({percentage:.1f}%)")

    print("\n损失权重说明:")
    print(f"  cls (分类): {config['cls']} - 较低权重，辅助分类")
    print(f"  pose (关键点): {config['pose']} - 最高权重，主要任务")
    print(f"  关键点检测是主要目标，类别分类作为辅助信息")

    print("\n开始训练...")
    print("提示: 训练过程中可以按 Ctrl+C 安全停止\n")

    # 开始训练
    try:
        results = model.train(**config)

        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        print(f"\n最佳模型: {config['project']}/{config['name']}/weights/best.pt")
        print(f"最后模型: {config['project']}/{config['name']}/weights/last.pt")
        print(f"训练曲线: {config['project']}/{config['name']}/")

        print("\n下一步:")
        print("1. 查看训练曲线: results.png")
        print("2. 验证模型: python validate_asoct_pose.py")
        print("3. 预测图片: python predict_asoct_pose.py")
        print(f"4. 导出模型: yolo export model={config['project']}/{config['name']}/weights/best.pt format=onnx")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        print(f"已保存的检查点: {config['project']}/{config['name']}/weights/")
        print("可以使用 resume=True 继续训练")

    except Exception as e:
        print(f"\n训练出错: {str(e)}")
        print("\n常见问题解决:")
        print("1. GPU内存不足: 减小 batch 大小 (如: batch=8 或 batch=4)")
        print("2. 数据集路径错误: 检查 asoct-pose.yaml 中的路径")
        print("3. 数据未转换: 先运行 python convert_asoct_to_yolo_pose.py")
        print("4. 依赖缺失: pip install ultralytics torch torchvision scikit-learn")


if __name__ == '__main__':
    main()
