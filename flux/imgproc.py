import random

from PIL import Image
import PIL.Image
import numpy as np


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def center_crop(pil_image, crop_size, is_tiled=False):
    if not is_tiled:
        while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

        scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def pad(pil_image, pad_size):
    while pil_image.size[0] >= 2 * pad_size[0] and pil_image.size[1] >= 2 * pad_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = min(pad_size[0] / pil_image.size[0], pad_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    new_image = Image.new('RGB', pad_size, (255, 255, 255))
    new_image.paste(pil_image, (0, 0))
    return new_image


def var_center_crop(pil_image, crop_size_list, random_top_k=4, is_tiled=False):
    w, h = pil_image.size
    pre_crop_size_list = [
        (cw, ch) for cw, ch in crop_size_list if cw <= w and ch <= h
    ]
    if is_tiled and len(pre_crop_size_list) > 0:
        crop_size = random.choice(pre_crop_size_list)
    else:
        is_tiled = False
        rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
        crop_size = random.choice(
            sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
        )[1]
    return center_crop(pil_image, crop_size, is_tiled)


def var_pad(pil_image, pad_size_list, random_top_k=4):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in pad_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, pad_size_list)), reverse=True)[:random_top_k]
    )[1]
    return pad(pil_image, crop_size)


def match_size(w, h, crop_size_list, random_top_k=4):
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return crop_size


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0, step_size=1):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, step_size
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + step_size) * wp <= num_patches:
            hp += step_size
        else:
            wp -= step_size
    return crop_size_list

def to_rgb_if_rgba(img: Image.Image):
    if img.mode.upper() == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        return rgb_img
    else:
        return img

def match_histograms(source, reference):
    src_hist, _ = np.histogram(source.ravel(), 256, [0,256])
    src_cdf = np.cumsum(src_hist) / float(src_hist.sum())
    ref_hist, _ = np.histogram(reference.ravel(), 256, [0,256])
    ref_cdf = np.cumsum(ref_hist) / float(ref_hist.sum())

    # Create a lookup table to map pixel values in the source
    # to their corresponding values in the target
    lut = np.interp(src_cdf, ref_cdf, np.arange(256))

    return lut[source]

def apply_histogram_matching(upscaled, original):
    """
    Apply histogram matching for each channel of the RGB image
    """
    if isinstance(upscaled, Image.Image):
        upscaled = np.array(upscaled)
        # upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    else:
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)

    if isinstance(original, Image.Image):
        original = np.array(original)
        # original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    else:
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)


    matched = np.zeros_like(upscaled)
    for channel in range(3):
        matched[:,:,channel] = match_histograms(upscaled[:,:,channel], original[:,:,channel])
    return matched

def apply_statistical_color_matching(upscaled, original):
    """
    Match colors between upscaled and original images using mean and standard deviation.
    Applies the transformation independently for each RGB channel.
    """
    if isinstance(upscaled, Image.Image):
        upscaled = np.array(upscaled, dtype=np.float32)
    else:
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB).astype(np.float32)

    if isinstance(original, Image.Image):
        original = np.array(original, dtype=np.float32)
    else:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float32)

    matched = np.zeros_like(upscaled)
    
    # Process each channel independently
    for channel in range(3):
        up_mean = np.mean(upscaled[:,:,channel])
        up_std = np.std(upscaled[:,:,channel])
        orig_mean = np.mean(original[:,:,channel])
        orig_std = np.std(original[:,:,channel])
        
        # Normalize, scale, and shift
        matched[:,:,channel] = (((upscaled[:,:,channel] - up_mean) / up_std) 
                               * orig_std + orig_mean)
    
    # Clip values to valid range [0, 255]
    matched = np.clip(matched, 0, 255)
    return matched