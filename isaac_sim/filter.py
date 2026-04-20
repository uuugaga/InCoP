import os
import glob
import json

# ==========================================
# 參數設定
# ==========================================
BASE_PATH = os.path.expanduser("~/_my_code/Dataset/full_warehouse_v1/dual_case/case_1/Jackal_R1")

# 填寫你想保留的類別名稱。如果設為空列表 []，就會保留所有出現過的類別
# KEEP_CLASSES = ["box", "pallet", "rack"] 
KEEP_CLASSES = ["box", 
                "wet_floor_sign", 
                "traffic_cone", 
                "fire_extinguisher", 
                "chair",
                "water_dispenser",
                "medical_bag",
                "trash_can",
                "laptop",
                "monitor",
                "potted_plant",
                "rubiks_cube",
                "table",
                "mug",
                ] 

def process_detection_data(base_dir, keep_classes=None):
    if keep_classes is None:
        keep_classes = []
        
    raw_base = os.path.join(base_dir, "label", "detection_raw")
    out_base = os.path.join(base_dir, "label", "detection")
    os.makedirs(out_base, exist_ok=True)

    tasks = {
        "2D": {
            "dirs": {
                "2d_tight": "2d_tight_raw",
                "2d_loose": "2d_loose_raw"
            },
            "meta_name": "classes_meta_2d.json"
        },
        "3D": {
            "dirs": {
                "3d": "3d_raw"
            },
            "meta_name": "classes_meta_3d.json"
        }
    }

    for task_type, task_info in tasks.items():
        print(f"\n{'='*50}")
        print(f"🚀 開始處理 {task_type} 資料管線...")
        print(f"{'='*50}")
        
        name_set = set()   # 記錄所有出現過的 class_name，徹底無視原始 ID
        
        # ------------------------------------------
        # Pass 1: 掃描並收集所有出現過的字串名稱
        # ------------------------------------------
        for out_sub, raw_sub in task_info["dirs"].items():
            search_path = os.path.join(raw_base, raw_sub, "*.txt")
            for file_path in glob.glob(search_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: 
                            continue
                            
                        # 只要第一個字串當作唯一標準！
                        cls_name = parts[0]
                        name_set.add(cls_name)

        print(f"✅ {task_type} 掃描完成。場景中發現 {len(name_set)} 種物件:\n   {sorted(list(name_set))}")

        # ------------------------------------------
        # 建立全新的 YOLO 標準連續 ID Mapping (0, 1, 2...)
        # ------------------------------------------
        if len(keep_classes) > 0:
            final_names = sorted([n for n in name_set if n in keep_classes])
            print(f"🎯 套用過濾條件，保留 {len(final_names)} 種物件:\n   {final_names}")
        else:
            final_names = sorted(list(name_set))
            print("🎯 未設定過濾條件，保留全數物件。")
            
        # 根據排序好的名稱，賦予全新的乾淨 ID
        new_mapping = {name: idx for idx, name in enumerate(final_names)}
        new_mapping_inv = {idx: name for name, idx in new_mapping.items()}

        # 儲存該維度專屬的 Meta Data
        meta_path = os.path.join(out_base, task_info["meta_name"])
        with open(meta_path, 'w') as f:
            json.dump({
                "names": final_names,          
                "name_to_id": new_mapping,     
                "id_to_name": new_mapping_inv  
            }, f, indent=4)
        print(f"💾 {task_type} Meta Data 已建立: {task_info['meta_name']}")

        # ------------------------------------------
        # Pass 2: 移除 Class Name，寫入純數值標準格式
        # ------------------------------------------
        print(f"⏳ 正在轉換 {task_type} 標籤格式...")
        for out_sub, raw_sub in task_info["dirs"].items():
            raw_dir = os.path.join(raw_base, raw_sub)
            out_dir = os.path.join(out_base, out_sub)
            os.makedirs(out_dir, exist_ok=True)
            
            txt_files = glob.glob(os.path.join(raw_dir, "*.txt"))
            processed_count = 0
            
            for file_path in txt_files:
                filename = os.path.basename(file_path)
                out_lines = []
                
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: 
                            continue
                            
                        cls_name = parts[0]
                        # 只有在 Mapping 裡的才保留
                        if cls_name in new_mapping:
                            new_id = new_mapping[cls_name]
                            # 🚀 關鍵：拋棄 parts[1] (舊的亂數 ID)，直接換成新 ID，然後接上後面的座標數值
                            new_line = f"{new_id} " + " ".join(parts[2:])
                            out_lines.append(new_line)
                
                with open(os.path.join(out_dir, filename), 'w') as f:
                    if out_lines:
                        f.write("\n".join(out_lines) + "\n")
                        
                processed_count += 1
            
            print(f"   📂 [{out_sub}] 處理完畢，共轉換 {processed_count} 個檔案。")

if __name__ == "__main__":
    process_detection_data(BASE_PATH, KEEP_CLASSES)