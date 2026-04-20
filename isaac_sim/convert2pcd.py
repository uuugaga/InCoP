import os
import numpy as np
from pathlib import Path

def convert_npy_to_pcd_binary(npy_path, pcd_path):
    """與單檔測試相同的轉換邏輯"""
    try:
        pc_np = np.load(npy_path).astype(np.float32)
    except Exception as e:
        print(f"⚠️ 無法讀取 {npy_path}: {e}")
        return False

    num_points = pc_np.shape[0]
    num_channels = pc_np.shape[1] if len(pc_np.shape) > 1 else 1
    
    # 防呆機制：如果這個 npy 不是點雲 (例如 shape 不對)，就跳過
    if num_channels not in [3, 4, 5, 6] or len(pc_np.shape) != 2:
        print(f"⚠️ {npy_path} 形狀為 {pc_np.shape}，不像點雲，跳過轉換。")
        return False
        
    header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
    if num_channels >= 4:
        header += "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        pc_np = pc_np[:, :4]
    else:
        header += "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        pc_np = pc_np[:, :3]
        
    header += f"WIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
    header += f"POINTS {num_points}\nDATA binary\n"
    
    with open(pcd_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(pc_np.tobytes())
    return True

def process_dataset(dataset_root):
    """遞迴尋找並轉換"""
    root_path = Path(dataset_root)
    
    # 使用 rglob 找出所有副檔名為 .npy 的檔案
    all_npy_files = list(root_path.rglob("*.npy"))
    
    # 過濾：只要路徑中包含 'lidar' 這個資料夾的 .npy 才處理
    lidar_npy_files = [f for f in all_npy_files if f.parent.name == 'lidar']
    
    print(f"🔍 在 {dataset_root} 中找到 {len(lidar_npy_files)} 個待轉換的 LiDAR .npy 檔案。")
    
    if len(lidar_npy_files) == 0:
        return

    success_count = 0
    # 開始批次處理
    for npy_file in lidar_npy_files:
        # 產生新的 .pcd 路徑 (同檔名，改副檔名)
        pcd_file = npy_file.with_suffix('.pcd')
        
        # 1. 轉換
        if convert_npy_to_pcd_binary(npy_file, pcd_file):
            # 2. 驗證新檔案確實存在且有大小
            if pcd_file.exists() and pcd_file.stat().st_size > 100:
                # 3. 刪除原本的 .npy
                npy_file.unlink()
                success_count += 1
                print(f"🔄 成功轉換並刪除: {npy_file.name} -> {pcd_file.name}")
            else:
                print(f"❌ 轉換失敗或檔案異常，保留原檔: {npy_file}")

    print(f"\n🎉 處理完成！共成功轉換 {success_count} / {len(lidar_npy_files)} 個檔案。")

if __name__ == "__main__":
    DATASET_DIR = "./Dataset" 
    
    print(f"即將對 {DATASET_DIR} 進行遞迴轉換 (NPY -> PCD) 並刪除原檔。")
    # 建議先備份 Dataset，或先用一小部分的資料夾測試！
    response = input("確定要執行嗎？ (y/N): ")
    
    if response.lower() == 'y':
        process_dataset(DATASET_DIR)
    else:
        print("操作已取消。")