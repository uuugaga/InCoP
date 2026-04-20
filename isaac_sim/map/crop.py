import cv2
import numpy as np
import yaml
import os

def refine_and_crop_map(image_path, yaml_path, output_name=None):
    # 1. 讀取 YAML
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    res = data['resolution']
    old_origin = np.array(data['origin']) # [x, y, z]

    # 2. 讀取圖片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 3. 建立遮罩：找出所有「非灰色」的區域
    # 根據分析，灰色大約是 126。我們保留明顯的白色(>200)與黑色(<50)
    # 或者排除掉中間的灰色值
    mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1] # 取得白色
    mask_black = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)[1] # 取得黑色
    combined_mask = cv2.bitwise_or(mask, mask_black)

    # 4. 計算最小 Bounding Box
    coords = cv2.findNonZero(combined_mask)
    x, y, w, h = cv2.boundingRect(coords)

    # 5. 執行裁剪
    cropped_img = img[y:y+h, x:x+w]
    
    # 6. 計算新的 Origin (關鍵：座標系轉換)
    # 圖片 y 軸是向下增長，但世界座標 y 是向上增長
    # 我們裁剪掉的左側像素為 x，底部裁掉的像素量為 img_height - (y + h)
    img_h, img_w = img.shape
    bottom_cut = img_h - (y + h)
    
    new_origin_x = old_origin[0] + (x * res)
    new_origin_y = old_origin[1] + (bottom_cut * res)
    new_origin = [float(new_origin_x), float(new_origin_y), float(old_origin[2])]

    # 7. 儲存結果
    crop_output_folder = "cropped_maps"
    os.makedirs(crop_output_folder, exist_ok=True)
    cv2.imwrite(f"{crop_output_folder}/{output_name}.png", cropped_img)
    data['image'] = f"{output_name}.png"
    data['origin'] = new_origin
    
    with open(f"{crop_output_folder}/{output_name}.yaml", 'w') as f:
        yaml.dump(data, f)

    print(f"裁切完成！新原點：{new_origin}，尺寸：{w}x{h}")

# 執行
refine_and_crop_map('raw_maps/office_v1.png', 'raw_maps/office_v1.yaml', output_name='office')