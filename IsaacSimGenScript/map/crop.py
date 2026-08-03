import cv2
import numpy as np
import yaml
import os

def refine_and_crop_map(image_path, yaml_path, output_name=None):
    # 1. Load the YAML metadata.
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    res = data['resolution']
    old_origin = np.array(data['origin']) # [x, y, z]

    # 2. Load the image.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 3. Build a mask that covers all non-gray regions.
    # The background gray level is approximately 126. Keep clear white (>200)
    # and black (<50) regions, excluding the middle gray range.
    mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]  # White regions
    mask_black = cv2.threshold(
        img, 50, 255, cv2.THRESH_BINARY_INV
    )[1]  # Black regions
    combined_mask = cv2.bitwise_or(mask, mask_black)

    # 4. Compute the minimum bounding box.
    coords = cv2.findNonZero(combined_mask)
    x, y, w, h = cv2.boundingRect(coords)

    # 5. Crop the image.
    cropped_img = img[y:y+h, x:x+w]
    
    # 6. Compute the new origin (coordinate-frame conversion).
    # Image y increases downward, while world-coordinate y increases upward.
    # The crop removes x pixels from the left and
    # img_height - (y + h) pixels from the bottom.
    img_h, img_w = img.shape
    bottom_cut = img_h - (y + h)
    
    new_origin_x = old_origin[0] + (x * res)
    new_origin_y = old_origin[1] + (bottom_cut * res)
    new_origin = [float(new_origin_x), float(new_origin_y), float(old_origin[2])]

    # 7. Save the results.
    crop_output_folder = "cropped_maps"
    os.makedirs(crop_output_folder, exist_ok=True)
    cv2.imwrite(f"{crop_output_folder}/{output_name}.png", cropped_img)
    data['image'] = f"{output_name}.png"
    data['origin'] = new_origin
    
    with open(f"{crop_output_folder}/{output_name}.yaml", 'w') as f:
        yaml.dump(data, f)

    print(f"Cropping complete. New origin: {new_origin}, size: {w}x{h}")

# Run preprocessing.
refine_and_crop_map('raw_maps/office.png', 'raw_maps/office.yaml', output_name='office')
refine_and_crop_map('raw_maps/hospital.png', 'raw_maps/hospital.yaml', output_name='hospital')
refine_and_crop_map('raw_maps/warehouse.png', 'raw_maps/warehouse.yaml', output_name='warehouse')
