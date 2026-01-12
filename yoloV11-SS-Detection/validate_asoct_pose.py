"""
YOLOv11 Pose 验证脚本
用于评估训练好的AS-OCT巩膜突检测模型性能
支持4个类别: Normal, Cataract, Glaucoma, Glaucoma_Cataract
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def load_dataset_info(yaml_path):
    """从YAML配置文件加载数据集信息"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config['path'])
    val_img_dir = dataset_path / config['val']

    val_images = list(val_img_dir.glob('*.jpg')) if val_img_dir.exists() else []

    return {
        'val_count': len(val_images),
        'names': config.get('names', {}),
        'kpt_shape': config.get('kpt_shape', [2, 3])
    }

def main():
    print("=" * 70)
    print("YOLOv11 Pose 模型验证 - AS-OCT 巩膜突检测")
    print("=" * 70)

    # 模型配置 - 修改这里来切换模型
    model_size = 'l'  # 可选: 'n', 's', 'm', 'l', 'x'
    model_path = f'runs/pose/asoct_yolo11{model_size}_4class/weights/best.pt'

    # 数据集配置
    data_yaml = 'datasets/asoct-pose-4class.yaml'

    print(f"\n加载模型: {model_path}")
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"\n错误: 模型文件不存在 - {model_path}")
        print("请先运行 train_asoct_pose.py 训练模型")
        return

    # 加载数据集信息
    try:
        dataset_info = load_dataset_info(data_yaml)
    except Exception as e:
        print(f"\n警告: 无法加载数据集信息 - {str(e)}")
        dataset_info = {'val_count': 0, 'names': {}, 'kpt_shape': [2, 3]}

    # 检查设备
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 验证配置
    print("\n验证配置:")
    print("-" * 70)

    config = {
        'data': data_yaml,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'workers': 8,
        'verbose': True,
        'save_json': True,
        'save_hybrid': False,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'half': False,
        'plots': True,
        'split': 'val',
    }

    for key, value in config.items():
        print(f"  {key:20s}: {value}")

    print("-" * 70)
    if dataset_info['val_count'] > 0:
        print(f"\n验证集: {dataset_info['val_count']} 张图片")
        print(f"类别数: {len(dataset_info['names'])}")
        print(f"关键点数: {dataset_info['kpt_shape'][0]}")
    print("\n开始验证...")

    # 执行验证
    try:
        metrics = model.val(**config)

        print("\n" + "=" * 70)
        print("验证完成!")
        print("=" * 70)

        # 打印关键指标
        print("\n关键指标:")
        print("-" * 70)
        print(f"  mAP50 (box):    {metrics.box.map50:.4f}")
        print(f"  mAP50-95 (box): {metrics.box.map:.4f}")
        print(f"  mAP50 (pose):   {metrics.pose.map50:.4f}")
        print(f"  mAP50-95 (pose):{metrics.pose.map:.4f}")
        print("-" * 70)

        print("\n指标说明:")
        print("- mAP50: 在IoU=0.5时的平均精度")
        print("- mAP50-95: 在IoU=0.5:0.95时的平均精度")
        print("- box: 边界框检测指标")
        print("- pose: 关键点检测指标")

        print(f"\n关键点: left_scleral_spur, right_scleral_spur")
        print("\n结果已保存到模型目录")

    except Exception as e:
        print(f"\n验证出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
