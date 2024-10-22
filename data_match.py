import json

# 配置参数
KREA1_FILE = 'krea_2M_full.json'
KREA2_FILE = 'krea_2M_filtered-by-goose.json'
OUTPUT_FILE = 'krea_2M_full_filtered.json'

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 保存 JSON 文件
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 将 KREA1 中的 text_embeddings_path 添加到 KREA2 中对应的元素
def merge_text_embeddings(krea1_data, krea2_data):
    # 创建一个字典用于快速查找 KREA1 中的 text_embeddings_path
    path_to_embeddings = {item['path']: item['text_embeddings_path'] for item in krea1_data if 'text_embeddings_path' in item}

    # 更新 KREA2 中的元素，添加 text_embeddings_path，未找到则删除对应元素
    merged_data = [item for item in krea2_data if item['path'] in path_to_embeddings]
    for item in merged_data:
        item['text_embeddings_path'] = path_to_embeddings[item['path']]

    print(merged_data[100:10000:100])

    return merged_data

if __name__ == "__main__":
    # 加载输入数据
    krea1_data = load_json(KREA1_FILE)
    krea2_data = load_json(KREA2_FILE)

    print(len(krea1_data), len(krea2_data))

    # 合并 text_embeddings_path 信息
    merged_data = merge_text_embeddings(krea1_data, krea2_data)

    # 保存最终输出
    save_json(merged_data, OUTPUT_FILE)

    print(f"Merging completed. Merged data saved to {OUTPUT_FILE}")
