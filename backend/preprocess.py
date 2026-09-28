import base64
import io
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass

def preprocess_image(base64_str: str) -> np.ndarray:
    # 1. Decode the base64 string and convert to grayscale
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    try:
        image_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_data)).convert('L')
    except Exception:
        raise ValueError("Invalid base64 image data")

    # 2. Ensure it's a white digit on a black background
    img_arr = np.array(img)
    # If the background is mostly white, invert the image colors
    if np.mean(img_arr) > 127:
        img = ImageOps.invert(img)

    # 3. Crop to the bounding box of the digit
    bbox = img.getbbox()
    if not bbox:
        raise ValueError("Image is completely blank. Draw something first!")
    img = img.crop(bbox)

    # 4. Resize so the longest dimension fits in 20x20, keeping aspect ratio
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 5. Paste it into a 28x28 black canvas and center it by center of mass
    # WHY THIS MATTERS: MNIST images aren't just scaled; they are centered by their 
    # center of mass. Skipping this step makes real-world drawings highly inaccurate.
    temp_arr = np.array(img)
    cy, cx = center_of_mass(temp_arr)

    # Fallback to simple geometric center if center of mass calculation fails
    if np.isnan(cy) or np.isnan(cx):
        cy, cx = temp_arr.shape[0] / 2, temp_arr.shape[1] / 2

    # Calculate exact paste offset to place the center of mass at (14, 14)
    offset_x = int(np.round(14 - cx))
    offset_y = int(np.round(14 - cy))

    new_img = Image.new('L', (28, 28), color=0)
    new_img.paste(img, (offset_x, offset_y))

    # 6. Normalize to [0,1] and reshape to match model input (1, 28, 28, 1)
    final_arr = np.array(new_img, dtype=np.float32) / 255.0
    return final_arr.reshape(1, 28, 28, 1)
