import json
import numpy as np
import matplotlib.pyplot as plt

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 解析分辨率和长宽比
def parse_resolution_and_ratio(data):
    resolutions = []
    aspect_ratios = []
    for item in data:
        resolution_str = item['resolution'].split(':')[1]
        width, height = map(int, resolution_str.split('x'))
        resolution = int((width * height)**0.5)
        resolution = max(min(resolution, 6144), 256)
        resolutions.append(resolution)
        aspect_ratio = round(width / height, 1)
        aspect_ratio = max(min(aspect_ratio, 2.5), 0.3)
        aspect_ratios.append(aspect_ratio)
    return resolutions, aspect_ratios

# 绘制柱形图并保存
def plot_histogram(data, bins, title, xlabel, ylabel, output_file):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(output_file)
    plt.close()

# 主函数
def main():
    # 读取数据
    file_path = 'krea_2M_filtered-by-goose.json'  # 替换为你的 JSON 文件路径
    data = read_json(file_path)
    
    # 解析数据
    resolutions, aspect_ratios = parse_resolution_and_ratio(data)  
    
    # 绘制分辨率分布柱形图
    resolution_bins = range(0, max(resolutions) + 256, 256)
    plot_histogram(
        resolutions,
        bins=resolution_bins,
        title='Image Resolution Distribution',
        xlabel='Resolution (pixels)',
        ylabel='Frequency',
        output_file='resolution_distribution_goose.png'
    )
    
    # 绘制长宽比分布柱形图
    aspect_ratio_bins = np.arange(0, max(aspect_ratios) + 0.1, 0.1)
    plot_histogram(
        aspect_ratios,
        bins=aspect_ratio_bins,
        title='Image Aspect Ratio Distribution',
        xlabel='Aspect Ratio (Width/Height)',
        ylabel='Frequency',
        output_file='aspect_ratio_distribution_goose.png'
    )

if __name__ == '__main__':
    main()
