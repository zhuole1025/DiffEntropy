import json
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
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

def update_json_with_aspect_ratio(input_file, output_file, max_workers=4):
    # 1. Read data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 2. Prepare aspect ratio lists (as in your original code)
    aspect_ratios_list_1024 = list(ASPECT_RATIO_1024.keys())
    aspect_ratios_list_2048 = list(ASPECT_RATIO_2048.keys())
    aspect_ratios_list_256  = list(ASPECT_RATIO_256.keys())
    aspect_ratios_list_512  = list(ASPECT_RATIO_512.keys())

    # Partially apply fixed arguments for our function
    partial_func = partial(
        compute_item_aspect_ratios,
        aspect_ratios_list_1024=aspect_ratios_list_1024,
        aspect_ratios_list_2048=aspect_ratios_list_2048,
        aspect_ratios_list_256=aspect_ratios_list_256,
        aspect_ratios_list_512=aspect_ratios_list_512
    )

    # 3. Process items in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Using 'map' here won't give you a progress bar directly.
        # For a progress bar, collect futures individually:
        futures = [executor.submit(partial_func, item) for item in data]
        for f in tqdm(futures):
            results.append(f.result())

    # 4. Write updated results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    input_file = '/mnt/petrelfs/gaopeng/qinqi/lumina2/data_folder/old/flux_pro_EN_Tag.json'
    output_file = '/mnt/petrelfs/gaopeng/zl/Lumina_v2/zl_2m/flux_pro_EN_Tag.json'

    # Adjust max_workers based on your CPU cores and I/O constraints
    update_json_with_aspect_ratio(input_file, output_file, max_workers=128)