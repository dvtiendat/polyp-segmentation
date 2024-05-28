import numpy as np

def mask_to_rgb(mask, color_dict):
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for key in color_dict:
        rgb_mask[mask == key] = color_dict[key]
    return rgb_mask

color_dict = {
    0: [0, 0, 0],  # Background
    1: [255, 0, 0],  # Red
    2: [0, 255, 0]   # Green
}