"""
AS-OCT巩膜突关键点检测评估脚本
计算预测关键点与真实标注之间的像素距离

评估指标:
1. 平均像素距离 (Mean Pixel Distance, MPD)
2. 标准差 (Standard Deviation, STD)
3. 中位数距离 (Median Distance)
4. PCK@Xpx (Percentage of Correct Keypoints at X pixels)
"""

import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt


def load_ground_truth_from_txt(label_path, img_width, img_height):
    """
    从YOLO格式的txt文件加载真实关键点
    返回: [(x1, y1), (x2, y2)] 像素坐标
    """
    if not label_path.exists():
        return None

    with open(label_path, 'r') as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = list(map(float, line.split()))

    # YOLO格式: class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
    if len(parts) < 11:
        return None

    # 提取关键点 (归一化坐标)
    kp1_x_norm, kp1_y_norm, kp1_v = parts[5], parts[6], parts[7]
    kp2_x_norm, kp2_y_norm, kp2_v = parts[8], parts[9], parts[10]

    keypoints = []

    # left_scleral_spur
    if kp1_v > 0:  # 可见
        x = kp1_x_norm * img_width
        y = kp1_y_norm * img_height
        keypoints.append((x, y))
    else:
        keypoints.append(None)

    # right_scleral_spur
    if kp2_v > 0:  # 可见
        x = kp2_x_norm * img_width
        y = kp2_y_norm * img_height
        keypoints.append((x, y))
    else:
        keypoints.append(None)

    return keypoints


def calculate_pixel_distance(pred_kp, gt_kp):
    """
    计算两个关键点之间的欧氏距离（像素）
    pred_kp: (x, y) 预测坐标
    gt_kp: (x, y) 真实坐标
    返回: 距离（像素）
    """
    if pred_kp is None or gt_kp is None:
        return None

    pred_x, pred_y = pred_kp
    gt_x, gt_y = gt_kp

    distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    return distance


def evaluate_model(model_path, val_images_dir, val_labels_dir, conf_threshold=0.25):
    """
    评估模型在验证集上的关键点定位精度
    """
    print("=" * 80)
    print("AS-OCT 巩膜突关键点定位精度评估")
    print("=" * 80)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 获取所有验证图片
    val_images_path = Path(val_images_dir)
    val_labels_path = Path(val_labels_dir)

    image_files = list(val_images_path.glob('*.jpg')) + list(val_images_path.glob('*.png'))

    if len(image_files) == 0:
        print(f"错误: 未找到验证图片在 {val_images_dir}")
        return

    print(f"\n找到 {len(image_files)} 张验证图片")
    print(f"置信度阈值: {conf_threshold}")
    print("\n开始评估...\n")

    # 类别定义 (按长度降序排列，确保复合类别名先匹配)
    CATEGORIES = sorted(['cataract', 'normal', 'glaucoma', 'glaucoma_cataract'], key=len, reverse=True)

    # 统计数据 - 整体统计
    stats = {
        'left_scleral_spur': [],
        'right_scleral_spur': [],
        'all': []
    }

    # 统计数据 - 按类别统计
    stats_by_category = {
        category: {
            'left_scleral_spur': [],
            'right_scleral_spur': [],
            'all': []
        }
        for category in CATEGORIES
    }

    keypoint_names = ['left_scleral_spur', 'right_scleral_spur']

    no_detection_count = 0
    no_detection_by_category = {cat: 0 for cat in CATEGORIES}
    processed_count = 0
    processed_by_category = {cat: 0 for cat in CATEGORIES}

    # 逐张图片评估
    for img_path in image_files:
        # 获取对应的标注文件
        label_name = img_path.stem + '.txt'
        label_path = val_labels_path / label_name

        if not label_path.exists():
            continue

        # 从文件名提取类别 (格式: category_originalname.jpg)
        img_name = img_path.stem
        category = None
        for cat in CATEGORIES:
            if img_name.startswith(cat + '_'):
                category = cat
                break

        if category is None:
            # 如果无法识别类别，跳过或归类为unknown
            continue

        # 读取图片尺寸
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]

        # 加载真实标注
        gt_keypoints = load_ground_truth_from_txt(label_path, img_width, img_height)

        if gt_keypoints is None:
            continue

        # 模型预测
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            device=device,
            verbose=False
        )

        # 提取预测的关键点
        if len(results) == 0 or len(results[0].boxes) == 0:
            no_detection_count += 1
            no_detection_by_category[category] += 1
            continue

        result = results[0]

        # 获取第一个检测框的关键点
        if result.keypoints is None or len(result.keypoints) == 0:
            no_detection_count += 1
            no_detection_by_category[category] += 1
            continue

        pred_keypoints_norm = result.keypoints.xy[0].cpu().numpy()  # [2, 2]

        # 转换为像素坐标
        pred_keypoints = []
        for kp in pred_keypoints_norm:
            x, y = kp
            pred_keypoints.append((float(x), float(y)))

        # 计算每个关键点的距离
        for i, (pred_kp, gt_kp, kp_name) in enumerate(zip(pred_keypoints, gt_keypoints, keypoint_names)):
            if gt_kp is not None:
                distance = calculate_pixel_distance(pred_kp, gt_kp)

                if distance is not None:
                    # 整体统计
                    stats[kp_name].append(distance)
                    stats['all'].append(distance)

                    # 按类别统计
                    stats_by_category[category][kp_name].append(distance)
                    stats_by_category[category]['all'].append(distance)

        processed_count += 1
        processed_by_category[category] += 1

        if processed_count % 20 == 0:
            print(f"  已处理: {processed_count}/{len(image_files)}")

    # 打印结果
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)

    print(f"\n处理统计 (整体):")
    print(f"  总图片数: {len(image_files)}")
    print(f"  成功检测: {processed_count}")
    print(f"  未检测到: {no_detection_count}")
    print(f"  检测率: {processed_count / len(image_files) * 100:.1f}%")

    # 打印各类别的处理统计
    print(f"\n处理统计 (按类别):")
    print("-" * 80)
    for category in CATEGORIES:
        total_cat = processed_by_category[category] + no_detection_by_category[category]
        if total_cat > 0:
            detection_rate = processed_by_category[category] / total_cat * 100
            print(f"  {category.upper():15s}: 总数 {total_cat:3d}, 检测 {processed_by_category[category]:3d}, "
                  f"未检测 {no_detection_by_category[category]:2d}, 检测率 {detection_rate:.1f}%")

    # 计算并打印整体统计
    print("\n" + "=" * 80)
    print("【整体评估】关键点定位精度 (像素距离):")
    print("=" * 80)

    for kp_name in keypoint_names + ['all']:
        distances = stats[kp_name]

        if len(distances) == 0:
            continue

        distances = np.array(distances)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        median_dist = np.median(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # PCK@Xpx (Percentage of Correct Keypoints)
        pck_5 = np.sum(distances <= 5) / len(distances) * 100
        pck_10 = np.sum(distances <= 10) / len(distances) * 100
        pck_20 = np.sum(distances <= 20) / len(distances) * 100
        pck_50 = np.sum(distances <= 50) / len(distances) * 100

        display_name = "整体" if kp_name == 'all' else kp_name

        print(f"\n{display_name}:")
        print(f"  样本数量: {len(distances)}")
        print(f"  平均距离 (MPD): {mean_dist:.2f} ± {std_dist:.2f} 像素")
        print(f"  中位数距离: {median_dist:.2f} 像素")
        print(f"  最小距离: {min_dist:.2f} 像素")
        print(f"  最大距离: {max_dist:.2f} 像素")
        print(f"  PCK@5px:  {pck_5:.1f}%")
        print(f"  PCK@10px: {pck_10:.1f}%")
        print(f"  PCK@20px: {pck_20:.1f}%")
        print(f"  PCK@50px: {pck_50:.1f}%")

    # 计算并打印各类别的统计
    print("\n" + "=" * 80)
    print("【按类别评估】关键点定位精度 (像素距离):")
    print("=" * 80)

    for category in CATEGORIES:
        cat_stats = stats_by_category[category]

        # 只处理有数据的类别
        if len(cat_stats['all']) == 0:
            continue

        print(f"\n--- {category.upper()} ---")

        for kp_name in keypoint_names + ['all']:
            distances = cat_stats[kp_name]

            if len(distances) == 0:
                continue

            distances = np.array(distances)

            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            median_dist = np.median(distances)
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            # PCK@Xpx
            pck_5 = np.sum(distances <= 5) / len(distances) * 100
            pck_10 = np.sum(distances <= 10) / len(distances) * 100
            pck_20 = np.sum(distances <= 20) / len(distances) * 100
            pck_50 = np.sum(distances <= 50) / len(distances) * 100

            display_name = "整体" if kp_name == 'all' else kp_name

            print(f"\n  {display_name}:")
            print(f"    样本数量: {len(distances)}")
            print(f"    平均距离 (MPD): {mean_dist:.2f} ± {std_dist:.2f} 像素")
            print(f"    中位数距离: {median_dist:.2f} 像素")
            print(f"    范围: [{min_dist:.2f}, {max_dist:.2f}] 像素")
            print(f"    PCK@5px:  {pck_5:.1f}%  |  PCK@10px: {pck_10:.1f}%  |  PCK@20px: {pck_20:.1f}%  |  PCK@50px: {pck_50:.1f}%")

    # 打印所有类别的对比汇总表
    print("\n" + "=" * 80)
    print("【类别对比汇总】")
    print("=" * 80)

    # 表头
    print("\n类别性能对比表:")
    print("-" * 120)
    print(f"{'类别':15s} | {'样本数':>6s} | {'MPD':>8s} | {'中位数':>8s} | {'PCK@5px':>9s} | {'PCK@10px':>10s} | {'PCK@20px':>10s} | {'PCK@50px':>10s}")
    print("-" * 120)

    # 按类别输出汇总数据
    for category in CATEGORIES:
        if len(stats_by_category[category]['all']) == 0:
            continue

        distances = np.array(stats_by_category[category]['all'])
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        pck_5 = np.sum(distances <= 5) / len(distances) * 100
        pck_10 = np.sum(distances <= 10) / len(distances) * 100
        pck_20 = np.sum(distances <= 20) / len(distances) * 100
        pck_50 = np.sum(distances <= 50) / len(distances) * 100

        cat_display = category.upper().replace('_', ' ')
        print(f"{cat_display:15s} | {len(distances):6d} | {mean_dist:7.2f}px | {median_dist:7.2f}px | "
              f"{pck_5:8.1f}% | {pck_10:9.1f}% | {pck_20:9.1f}% | {pck_50:9.1f}%")

    print("-" * 120)

    # 输出整体数据
    all_distances = np.array(stats['all'])
    overall_mean = np.mean(all_distances)
    overall_median = np.median(all_distances)
    overall_pck5 = np.sum(all_distances <= 5) / len(all_distances) * 100
    overall_pck10 = np.sum(all_distances <= 10) / len(all_distances) * 100
    overall_pck20 = np.sum(all_distances <= 20) / len(all_distances) * 100
    overall_pck50 = np.sum(all_distances <= 50) / len(all_distances) * 100

    print(f"{'整体平均':15s} | {len(all_distances):6d} | {overall_mean:7.2f}px | {overall_median:7.2f}px | "
          f"{overall_pck5:8.1f}% | {overall_pck10:9.1f}% | {overall_pck20:9.1f}% | {overall_pck50:9.1f}%")
    print("-" * 120)

    # 找出最佳和最差类别
    category_mpds = []
    for category in CATEGORIES:
        if len(stats_by_category[category]['all']) > 0:
            mpd = np.mean(stats_by_category[category]['all'])
            category_mpds.append((category, mpd))

    if category_mpds:
        best_cat, best_mpd = min(category_mpds, key=lambda x: x[1])
        worst_cat, worst_mpd = max(category_mpds, key=lambda x: x[1])

        print(f"\n性能分析:")
        print(f"  最佳类别: {best_cat.upper().replace('_', ' ')} (MPD: {best_mpd:.2f}px)")
        print(f"  最差类别: {worst_cat.upper().replace('_', ' ')} (MPD: {worst_mpd:.2f}px)")
        print(f"  性能差异: {worst_mpd - best_mpd:.2f}px")

    # 绘制距离分布图
    print("\n生成可视化图表...")
    plot_distance_distribution(stats, stats_by_category, keypoint_names)

    return stats, stats_by_category


def plot_distance_distribution(stats, stats_by_category, keypoint_names):
    """
    绘制距离分布图（包含整体和按类别的可视化）
    """
    # 类别定义 (按长度降序排列，确保复合类别名先匹配)
    CATEGORIES = sorted(['cataract', 'normal', 'glaucoma', 'glaucoma_cataract'], key=len, reverse=True)

    # 创建更大的画布，包含按类别的统计
    fig = plt.figure(figsize=(18, 14))

    # 使用GridSpec创建布局
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('AS-OCT Scleral Spur Keypoint Localization Error Distribution',
                 fontsize=16, fontweight='bold')

    # ========== 第一行：整体统计 (原有的4个图) ==========

    # 1. 整体距离分布直方图
    ax1 = fig.add_subplot(gs[0, 0])
    all_distances = np.array(stats['all'])
    ax1.hist(all_distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(all_distances), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_distances):.2f}px')
    ax1.axvline(np.median(all_distances), color='green', linestyle='--',
                label=f'Median: {np.median(all_distances):.2f}px')
    ax1.set_xlabel('Pixel Distance (px)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Overall Distance Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 左右关键点对比箱线图
    ax2 = fig.add_subplot(gs[0, 1])
    data_to_plot = [stats[kp] for kp in keypoint_names if len(stats[kp]) > 0]
    labels = [kp.replace('_', ' ').title() for kp in keypoint_names if len(stats[kp]) > 0]
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_ylabel('Pixel Distance (px)', fontsize=10)
    ax2.set_title('Left vs Right Scleral Spur', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', labelsize=9)

    # 3. PCK曲线
    ax3 = fig.add_subplot(gs[0, 2])
    thresholds = np.arange(0, 101, 1)

    for kp_name in keypoint_names:
        if len(stats[kp_name]) == 0:
            continue

        distances = np.array(stats[kp_name])
        pck_values = [np.sum(distances <= t) / len(distances) * 100 for t in thresholds]

        label = kp_name.replace('_', ' ').title()
        ax3.plot(thresholds, pck_values, marker='o', markersize=2, label=label)

    ax3.set_xlabel('Distance Threshold (px)', fontsize=10)
    ax3.set_ylabel('PCK (%)', fontsize=10)
    ax3.set_title('PCK Curve', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0, 105])

    # ========== 第二行：按类别的箱线图对比 ==========

    # 4. 各类别整体距离箱线图
    ax4 = fig.add_subplot(gs[1, :])
    category_data = []
    category_labels = []
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    for i, category in enumerate(CATEGORIES):
        if len(stats_by_category[category]['all']) > 0:
            category_data.append(stats_by_category[category]['all'])
            category_labels.append(category.upper().replace('_', ' '))

    if category_data:
        bp = ax4.boxplot(category_data, labels=category_labels, patch_artist=True, showfliers=True)
        for patch, color in zip(bp['boxes'], colors[:len(category_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.set_ylabel('Pixel Distance (px)', fontsize=11)
        ax4.set_title('Distance Distribution by Category', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', labelsize=10)

        # 添加统计信息
        for i, (data, label) in enumerate(zip(category_data, category_labels)):
            mean_val = np.mean(data)
            ax4.text(i+1, mean_val, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ========== 第三行：各类别的PCK对比 ==========

    # 5. 各类别PCK@10px条形图
    ax5 = fig.add_subplot(gs[2, 0])
    pck10_values = []
    pck10_labels = []

    for category in CATEGORIES:
        if len(stats_by_category[category]['all']) > 0:
            distances = np.array(stats_by_category[category]['all'])
            pck10 = np.sum(distances <= 10) / len(distances) * 100
            pck10_values.append(pck10)
            pck10_labels.append(category.upper().replace('_', '\n'))

    if pck10_values:
        bars = ax5.bar(range(len(pck10_values)), pck10_values, color=colors[:len(pck10_values)], alpha=0.7, edgecolor='black')
        ax5.set_xticks(range(len(pck10_labels)))
        ax5.set_xticklabels(pck10_labels, fontsize=9)
        ax5.set_ylabel('PCK@10px (%)', fontsize=10)
        ax5.set_title('PCK@10px by Category', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim([0, 105])

        # 在柱子上显示数值
        for bar, val in zip(bars, pck10_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6. 各类别MPD条形图
    ax6 = fig.add_subplot(gs[2, 1])
    mpd_values = []
    mpd_labels = []

    for category in CATEGORIES:
        if len(stats_by_category[category]['all']) > 0:
            mpd = np.mean(stats_by_category[category]['all'])
            mpd_values.append(mpd)
            mpd_labels.append(category.upper().replace('_', '\n'))

    if mpd_values:
        bars = ax6.bar(range(len(mpd_values)), mpd_values, color=colors[:len(mpd_values)], alpha=0.7, edgecolor='black')
        ax6.set_xticks(range(len(mpd_labels)))
        ax6.set_xticklabels(mpd_labels, fontsize=9)
        ax6.set_ylabel('Mean Pixel Distance (px)', fontsize=10)
        ax6.set_title('MPD by Category', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # 在柱子上显示数值
        for bar, val in zip(bars, mpd_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 7. 各类别PCK曲线对比
    ax7 = fig.add_subplot(gs[2, 2])
    thresholds = np.arange(0, 101, 2)

    for i, category in enumerate(CATEGORIES):
        if len(stats_by_category[category]['all']) == 0:
            continue

        distances = np.array(stats_by_category[category]['all'])
        pck_values = [np.sum(distances <= t) / len(distances) * 100 for t in thresholds]

        ax7.plot(thresholds, pck_values, marker='o', markersize=3,
                label=category.upper(), color=colors[i], linewidth=2, alpha=0.8)

    ax7.set_xlabel('Distance Threshold (px)', fontsize=10)
    ax7.set_ylabel('PCK (%)', fontsize=10)
    ax7.set_title('PCK Curves by Category', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9, loc='lower right')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 100])
    ax7.set_ylim([0, 105])

    # 保存图表
    save_path = 'runs/pose/evaluation_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图表保存至: {save_path}")

    plt.close()


def main():
    # 配置 - 修改这里来切换模型
    model_size = 'l'  # 可选: 'n', 's', 'm', 'l', 'x'
    model_path = f'runs/pose/asoct_yolo11{model_size}_4class/weights/best.pt'
    val_images_dir = 'datasets/ASOCT_YOLO/images/val'
    val_labels_dir = 'datasets/ASOCT_YOLO/labels/val'
    conf_threshold = 0.25  # 置信度阈值

    # 检查路径
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在 - {model_path}")
        print("请先运行 train_asoct_pose.py 训练模型")
        return

    if not Path(val_images_dir).exists():
        print(f"错误: 验证集图片目录不存在 - {val_images_dir}")
        print("请先运行 convert_asoct_to_yolo_pose.py 转换数据")
        return

    if not Path(val_labels_dir).exists():
        print(f"错误: 验证集标注目录不存在 - {val_labels_dir}")
        print("请先运行 convert_asoct_to_yolo_pose.py 转换数据")
        return

    # 执行评估
    stats = evaluate_model(
        model_path=model_path,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        conf_threshold=conf_threshold
    )

    print("\n" + "=" * 80)
    print("评估指标说明:")
    print("=" * 80)
    print("1. MPD (Mean Pixel Distance): 平均像素距离，越小越好")
    print("2. STD (Standard Deviation): 标准差，反映稳定性")
    print("3. Median Distance: 中位数距离，对异常值更鲁棒")
    print("4. PCK@Xpx: X像素范围内正确关键点的百分比")
    print("   - PCK@5px: 临床高精度要求")
    print("   - PCK@10px: 临床一般精度要求")
    print("   - PCK@20px: 临床可接受精度")
    print("\n医学应用标准:")
    print("  优秀: MPD < 5px, PCK@10px > 95%")
    print("  良好: MPD < 10px, PCK@20px > 90%")
    print("  可用: MPD < 20px, PCK@50px > 85%")


if __name__ == '__main__':
    main()
