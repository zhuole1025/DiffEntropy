import argparse
import glob
import os

import numpy as np
import pyiqa
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Image Quality Assessment using PyIQA')
    parser.add_argument(
        '--root_dir',
        default='/ceph/data-bk/zl/DiffEntropy/flux/samples/eval/lq250_v3_with_single_control_redux_tiled_multi_degradation_train_wo_noise_wo_usm_0040000_gate_1.0_1.0_cfg_2.0_1.0_image_prompt/',
        help='根目录，包含多个子文件夹',
    )
    parser.add_argument('--is_root_dir', action='store_true', help='是否是根目录')
    parser.set_defaults(is_root_dir=False)
    parser.add_argument('--output_dir', required=False, help='Directory to save the evaluation results')
    parser.add_argument(
        '--metrics',
        nargs='+',
        required=False,
        default=['clipiqa', 'maniqa', 'niqe', 'musiq'],
        choices=['psnr', 'ssim', 'niqe', 'lpips', 'clipiqa', 'maniqa', 'musiq', 'brisque', 'fid'],
        help='List of metrics to use for evaluation (e.g., psnr ssim niqe lpips clipiqa maniqa musiq brisque)',
    )

    parser.add_argument(
        '--size',
        default=1024,
        type=int,
        help='Size of the images for evaluation',
    )

    return parser.parse_args()


def load_image(image_path, size=None):
    image = Image.open(image_path).convert('RGB')
    if size:
        image = image.resize(size, Image.BILINEAR)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = image.astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
    return image


def construct_image_dict(input_dir, gt_dir=None, sort=True):
    image_data = []

    if gt_dir is None:
        # 只有预测图像
        pred_images = glob.glob(os.path.join(input_dir, '*.*'))
        for pred_image_path in pred_images:
            # base_name = os.path.basename(pred_image_path).split('_')[0].split('.')[0]
            base_name = os.path.basename(pred_image_path).split('_')[0].split('.')[2]
            image_data.append(
                {
                    'pred_image_path': pred_image_path,
                    'base_name': base_name,
                }
            )
    else:
        # GT 和 pred 分别存放，基于 base_name 匹配
        pred_images = sorted([f for f in glob.glob(os.path.join(input_dir, '*')) if '_input' not in os.path.basename(f)])
        for pred_image_path in pred_images:
            # base_name = os.path.basename(pred_image_path).split('_')[0].split('.')[0]
            base_name = os.path.basename(pred_image_path).split('_')[2]
            step = os.path.basename(pred_image_path).split('_')[1]
            # 使用 glob 匹配包含 base_name 的所有 gt 图像
            gt_candidates = glob.glob(os.path.join(gt_dir, f"euler_{step}_{base_name}_*.*"))
            # 根据需求，可以设置优先选择某种格式的逻辑，或直接选取第一个找到的匹配
            if gt_candidates:
                gt_image_path = gt_candidates[0]  # 假设使用第一个匹配的文件
            else:
                gt_image_path = None  # 没有找到匹配的 GT 图像
            print(f"pred_image_path:{pred_image_path}")
            print(f"gt_image_path:{gt_image_path}")
            image_data.append(
                {
                    'input_image_path': None,  # 如果没有单独输入图像，可以设置为 None
                    'pred_image_path': pred_image_path,
                    'gt_image_path': gt_image_path,
                    'base_name': base_name,
                }
            )

    if sort:
        image_data.sort(key=lambda x: x['base_name'])

    return image_data


def evaluate_images(input_dir, output_dir, gt_dir, metrics):
    if output_dir is None:
        output_dir = os.path.dirname(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    metric_dict = {name: pyiqa.create_metric(name) for name in metrics}
    results = {name: [] for name in metrics}
    dir_name = os.path.basename(input_dir)
    overall_results = {}
    if 'fid' in metrics:
        # Directly compute FID for highest efficiency
        if gt_dir is not None:
            fid_metric = pyiqa.create_metric('fid')
            fid_score = fid_metric(input_dir, gt_dir)
            results['fid'].append(('fid_score', fid_score))
            print(f'FID score between generated images and ground truth images: {fid_score}')
            with open(os.path.join(output_dir, 'fid_score.txt'), 'w') as f:
                f.write(f'FID: {fid_score:.4f}\n')
            if len(metrics) == 1:
                return
        else:
            raise ValueError("Ground truth directory (gt_dir) is required for FID calculation.")
    
    # 调用构造字典函数
    image_data = construct_image_dict(input_dir, gt_dir)
    for data in tqdm(image_data):
        pred_image = (
            load_image(data['pred_image_path'], size=(args.size, args.size)) if args.size else load_image(data['pred_image_path'])
        )
        pred_img_h, pred_img_w = pred_image.shape[2], pred_image.shape[3]

        if 'gt_image_path' in data.keys() and data['gt_image_path'] is not None and os.path.exists(data['gt_image_path']):
            gt_image = load_image(data['gt_image_path'], size=(pred_img_w, pred_img_h))
        else:
            gt_image = None

        for metric_name, metric in metric_dict.items():
            if metric_name in ['psnr', 'ssim', 'lpips'] and gt_image is not None:
                score = metric(pred_image, gt_image).item()
            elif metric_name in ['niqe', 'clipiqa', 'maniqa', 'musiq', 'brisque']:
                score = metric(pred_image).item()
            else:
                continue
            results[metric_name].append((data['base_name'], score))

    for metric_name, scores in results.items():
        metric_file = os.path.join(output_dir, f'{dir_name}_{metric_name}_results.txt')
        with open(metric_file, 'w') as f:
            total_score = 0
            for img_name, score in scores:
                f.write(f"{img_name}: {score:.4f}\n")
                total_score += score
            if scores:
                avg_score = total_score / len(scores)
                f.write(f"Average: {avg_score:.4f}\n")
                overall_results[metric_name] = avg_score
            else:
                f.write("No valid scores available\n")
                overall_results[metric_name] = 0

    overall_file = os.path.join(output_dir, f'{dir_name}_overall_average_results.txt')
    with open(overall_file, 'w') as f:
        for metric_name, avg_score in overall_results.items():
            f.write(f"{metric_name}: {avg_score:.4f}\n")
        print(f'Overall results saved to {overall_file}')


def process_all_subdirs(root_dir, output_dir, metrics, size, is_root_dir):
    """
    处理根目录下的所有子文件夹
    每个子文件夹应包含 images, target_images, input_image 三个子文件夹
    """
    # 获取所有子文件夹
    if is_root_dir:
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    else:
        subdirs = [root_dir]

    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)

        # 构建输入输出路径
        pred_dir = os.path.join(subdir_path, 'images')
        gt_dir = os.path.join(subdir_path, 'target_images')
        
        # 检查必要的文件夹是否存在
        if not os.path.exists(pred_dir) and not os.path.exists(gt_dir):
            print(f"警告: {subdir} 中缺少必要的文件夹结构，跳过处理")
            continue

        # 创建该子文件夹的输出目录
        subdir_output = os.path.join(output_dir, subdir) if output_dir else os.path.join(subdir_path, 'metrics_results')
        os.makedirs(subdir_output, exist_ok=True)

        print(f"\n处理子文件夹: {subdir}")
        evaluate_images(pred_dir, subdir_output, gt_dir, metrics)


if __name__ == "__main__":
    args = parse_args()

    # 如果没有指定输出目录，在根目录创建一个
    output_dir = args.output_dir if args.output_dir else os.path.join(args.root_dir, 'metrics_results')
    os.makedirs(output_dir, exist_ok=True)

    # 处理所有子文件夹
    process_all_subdirs(args.root_dir, output_dir, args.metrics, args.size, args.is_root_dir)