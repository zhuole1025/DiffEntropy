import json
from tqdm import tqdm
from PIL import Image
from functools import partial

# Example aspect ratio dictionaries; replace with your imports
from imgproc import ASPECT_RATIO_1024, ASPECT_RATIO_2048, ASPECT_RATIO_256, ASPECT_RATIO_512


# If you need to read data from remote or specialized client:
from petrel_client.client import Client
from data import read_general
global client
client = Client("./petreloss.conf")

def compute_item_aspect_ratios(
    item, 
    aspect_ratios_list_1024,
    aspect_ratios_list_2048,
    aspect_ratios_list_256,
    aspect_ratios_list_512
):
    """Compute and add closest aspect ratios to a single item."""
    # Get resolution string and parse it
    if 'resolution' in item:
        resolution = item['resolution']
        # If resolution is in the format e.g. 'res:1920x1080'
        try:
            width, height = resolution.split(':')[1].split('x')
        except ValueError:
            # Fallback or custom parse
            width, height = resolution.split('x')
    else:
        # Example of opening from local or remote
        image = Image.open(read_general(item['image_path']))
        # image = Image.open(item['image_path'])  # Simplify for demo
        width, height = image.size

    # Compute aspect ratio
    aspect_ratio = float(height) / float(width)

    # Find closest ratio in each dictionary
    closest_ratio_1024 = min(aspect_ratios_list_1024, key=lambda r: abs(float(r) - aspect_ratio))
    closest_ratio_2048 = min(aspect_ratios_list_2048, key=lambda r: abs(float(r) - aspect_ratio))
    closest_ratio_256  = min(aspect_ratios_list_256,  key=lambda r: abs(float(r) - aspect_ratio))
    closest_ratio_512  = min(aspect_ratios_list_512,  key=lambda r: abs(float(r) - aspect_ratio))

    # Update item
    item['closest_ratio_1024'] = closest_ratio_1024
    item['closest_ratio_2048'] = closest_ratio_2048
    item['closest_ratio_256']  = closest_ratio_256
    item['closest_ratio_512']  = closest_ratio_512
    return item

def update_json_with_aspect_ratio(input_file, output_file):
    # 1. Read data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 2. Prepare aspect ratio lists
    aspect_ratios_list_1024 = list(ASPECT_RATIO_1024.keys())
    aspect_ratios_list_2048 = list(ASPECT_RATIO_2048.keys())
    aspect_ratios_list_256  = list(ASPECT_RATIO_256.keys())
    aspect_ratios_list_512  = list(ASPECT_RATIO_512.keys())

    # 3. Process items sequentially
    results = []
    for item in tqdm(data):
        result = compute_item_aspect_ratios(
            item,
            aspect_ratios_list_1024=aspect_ratios_list_1024,
            aspect_ratios_list_2048=aspect_ratios_list_2048,
            aspect_ratios_list_256=aspect_ratios_list_256,
            aspect_ratios_list_512=aspect_ratios_list_512
        )
        results.append(result)

    # 4. Write updated results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    input_file = '/mnt/petrelfs/gaopeng/zl/Lumina_v2/zl_2m/2m_new_lc2_30k_768.json'
    output_file = '/mnt/petrelfs/gaopeng/zl/Lumina_v2/zl_2m/2m_new_lc2_30k_768_aspect_ratio.json'

    update_json_with_aspect_ratio(input_file, output_file)