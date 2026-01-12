"""
YOLOv11 Pose 预测脚本
用于对AS-OCT图像进行巩膜突关键点检测
支持4个类别: Normal, Cataract, Glaucoma, Glaucoma_Cataract
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    print("=" * 70)
    print("YOLOv11 Pose 图像预测 - AS-OCT 巩膜突检测")
    print("=" * 70)

    # 模型配置 - 修改这里来切换模型
    model_size = 'l'  # 可选: 'n', 's', 'm', 'l', 'x'
    model_path = f'runs/pose/asoct_yolo11{model_size}_4class/weights/best.pt'

    # 要预测的图片路径
    # 示例:
    # - 单个图片: 'datasets/ASOCT_YOLO/images/val/cataract_xxx.jpg'
    # - 文件夹: 'datasets/ASOCT_YOLO/images/val/'
    # - 原始数据: 'datasets/Cataract/Original Images/'
    # - 视频: 'path/to/video.mp4'
    # - 摄像头: 0
    source = 'datasets/ASOCT_YOLO/images/val/'

    print(f"\n加载模型: {model_path}")
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"\n错误: 模型文件不存在 - {model_path}")
        print("请先运行 train_asoct_pose.py 训练模型")
        return

    # 检查设备
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 预测配置
    print("\n预测配置:")
    print("-" * 70)

    config = {
        'source': source,
        'imgsz': 640,
        'conf': 0.25,
        'iou': 0.7,
        'device': device,
        'max_det': 300,
        'half': False,
        'show': False,
        'save': True,
        'save_txt': True,
        'save_conf': True,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'retina_masks': False,
        'project': 'runs/pose',
        'name': 'predict',
        'exist_ok': True,
        'verbose': True,
    }

    for key, value in config.items():
        if key != 'source' or len(str(value)) < 80:
            print(f"  {key:20s}: {value}")

    print("-" * 70)
    print(f"\n预测输入: {source}")
    print("开始预测...\n")

    # 执行预测
    try:
        results = model.predict(**config)

        print("\n" + "=" * 70)
        print("预测完成!")
        print("=" * 70)

        # 统计结果
        total_images = len(results)
        total_detections = sum(len(r.boxes) for r in results)

        print(f"\n处理图片数: {total_images}")
        print(f"总检测数: {total_detections}")
        print(f"平均每张: {total_detections/total_images:.2f} 个检测" if total_images > 0 else "")

        # 找到保存路径
        if results and hasattr(results[0], 'save_dir'):
            save_dir = results[0].save_dir
            print(f"\n结果保存在: {save_dir}")
            print("  - 可视化图片: 带关键点标注的图片")
            print("  - labels/: 文本格式标注")

        print("\n提示:")
        print("- 如需调整检测灵敏度,修改 conf 参数")
        print("- 如需处理单张图片,修改 source 参数")
        print("- 可视化结果在输出目录中查看")

        # 显示第一张图的关键点信息
        if results and len(results) > 0 and hasattr(results[0], 'keypoints'):
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                print("\n第一张图片的关键点检测:")
                print("-" * 70)
                try:
                    kpts = results[0].keypoints.xy[0]
                    conf = results[0].boxes.conf[0] if len(results[0].boxes) > 0 else 0.0
                    kpt_names = ['left_scleral_spur', 'right_scleral_spur']

                    print(f"  检测置信度: {conf:.3f}")
                    for i, (kpt, name) in enumerate(zip(kpts, kpt_names)):
                        x, y = kpt
                        print(f"  {name:25s}: ({x:.1f}, {y:.1f})")
                except:
                    pass

        print("\n医学应用:")
        print("- 巩膜突定位用于房角评估")
        print("- 辅助青光眼诊断")
        print("- 白内障手术规划")

    except Exception as e:
        print(f"\n预测出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
