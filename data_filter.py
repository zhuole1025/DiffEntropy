import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
import cv2

# 配置参数
INPUT_FILE = './krea_2M.json'
OUTPUT_FILE = './krea_2M_filtered-by-goose.json'

# INPUT_FILE = 'demo_50.json'
# OUTPUT_FILE = 'demo_50_filtered.json'
THREAD_COUNT = 32
MIN_RESOLUTION = 1024 ** 2
BATCH_SIZE = 1000

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 保存 JSON 文件
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 过滤元素，根据图像路径获取分辨率
def filter_element(element):
    try:
        with Image.open('../../goosedata/images/' + element['path']) as img:
            width, height = img.size
            scale = int((height * width) ** 0.5)
            element['resolution'] = f"{scale}:{width}x{height}"
            # return width * height >= MIN_RESOLUTION
            return True
    except Exception as e:
        print(f"Error opening image {element['path']}: {e}")
        return False

# # 过滤元素，根据图像路径获取分辨率 (使用 OpenCV)
# def filter_element(element):
#     try:
#         img = cv2.imread('/goosedata/images/' + element['path'])
#         if img is None:
#             raise ValueError("Image could not be loaded")
#         height, width = img.shape[:2]
#         scale = int((height * width) ** 0.5)
#         element['resolution'] = f"{scale}:{width}x{height}"
#         return width * height >= MIN_RESOLUTION
#     except Exception as e:
#         print(f"Error opening image {element['path']}: {e}")
#         return False


# 并行处理数据
def process_data(input_data):
    # 尝试加载上次中断后的进度
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as temp_file:
            processed_data = json.load(temp_file)
    else:
        processed_data = []

    total_filtered = len(processed_data)
    batch_counter = 0
    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = {executor.submit(filter_element, element): element for element in input_data[len(processed_data):]}
        with tqdm(total=len(input_data), initial=len(processed_data), desc="Processing") as pbar:
            for future in as_completed(futures):
                element = futures[future]
                try:
                    if future.result():
                        processed_data.append(element)
                        total_filtered += 1
                except Exception as e:
                    print(f"Error processing element {element['path']}: {e}")
                # 保存中间结果
                # 保存中间结果，每处理 BATCH_SIZE 个元素后保存一次
                batch_counter += 1
                if batch_counter >= BATCH_SIZE:
                    save_json(processed_data, OUTPUT_FILE)
                    batch_counter = 0
                pbar.update(1)
                pbar.set_postfix(filtered=f"{total_filtered}/{len(input_data)} ({total_filtered/len(input_data):.2%})")

    return processed_data

if __name__ == "__main__":
    # 加载输入数据
    input_data = load_json(INPUT_FILE)
    
    # 处理数据
    output_data = process_data(input_data)
    
    # 保存最终输出
    save_json(output_data, OUTPUT_FILE)

    print(f"Processing completed. Filtered data saved to {OUTPUT_FILE}")
