import numpy as np
from PIL import Image

# Palette standard Cityscapes
CITYSCAPES_8_COLORS = np.array([
            (128, 64,128),  
            [244, 35,232], 
            [70,  70, 70],  
            [102,102,156],  
            [107,142, 35], 
            [70,130,180],                
            [220, 20, 60], 
            [0,   0,142], ])


def colorize_mask(mask):
    color_mask = CITYSCAPES_8_COLORS[mask]
    return Image.fromarray(color_mask.astype(np.uint8))

def overlay_mask(image, mask, alpha=0.5):
    image = image.resize(mask.size).convert("RGB")
    return Image.blend(image, mask, alpha)
