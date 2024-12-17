import json
from imgproc import generate_crop_size_list, ASPECT_RATIO_1024, ASPECT_RATIO_2048, ASPECT_RATIO_256, ASPECT_RATIO_512
from tqdm import tqdm

# Read the JSON file
def update_json_with_aspect_ratio(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    aspect_ratios_dict_1024 = eval('ASPECT_RATIO_1024')
    aspect_ratios_dict_2048 = eval('ASPECT_RATIO_2048')
    aspect_ratios_dict_256 = eval('ASPECT_RATIO_256')
    aspect_ratios_dict_512 = eval('ASPECT_RATIO_512')
    aspect_ratios_list_1024 = list(aspect_ratios_dict_1024.keys())
    aspect_ratios_list_2048 = list(aspect_ratios_dict_2048.keys())
    aspect_ratios_list_256 = list(aspect_ratios_dict_256.keys())
    aspect_ratios_list_512 = list(aspect_ratios_dict_512.keys())
    
    # Loop through each item in the JSON
    for item in tqdm(data):
        # Get resolution string and parse it
        resolution = item['resolution']
        # Split "3135:3840x2560" into width and height
        width, height = resolution.split(':')[1].split('x')
        # Calculate aspect ratio (height/width) and round to 3 decimal places
        aspect_ratio = float(height) / float(width)
        closest_ratio_1024 = min(aspect_ratios_list_1024, key=lambda r: abs(float(r) - aspect_ratio))
        closest_ratio_2048 = min(aspect_ratios_list_2048, key=lambda r: abs(float(r) - aspect_ratio))
        closest_ratio_256 = min(aspect_ratios_list_256, key=lambda r: abs(float(r) - aspect_ratio))
        closest_ratio_512 = min(aspect_ratios_list_512, key=lambda r: abs(float(r) - aspect_ratio))
        
        # Add new key-value pair to the item
        item['closest_ratio_1024'] = closest_ratio_1024
        item['closest_ratio_2048'] = closest_ratio_2048
        item['closest_ratio_256'] = closest_ratio_256
        item['closest_ratio_512'] = closest_ratio_512
    
    # Save the updated JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage
input_file = '/data/zl/DiffEntropy/krea_2M_full_filtered.json'
output_file = '/data/zl/DiffEntropy/krea_2M_full_bucket.json'
update_json_with_aspect_ratio(input_file, output_file)