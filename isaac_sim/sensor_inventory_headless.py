from isaacsim import SimulationApp
import os

# 以 Headless 模式啟動，不消耗 GUI 資源
simulation_app = SimulationApp({"headless": True})

from isaacsim.storage.native import get_assets_root_path
import omni.client

def list_sensors_recursive(current_path, depth=0):
    """遞迴搜尋 USD 檔案"""
    result, entries = omni.client.list(current_path)
    
    if result != omni.client.Result.OK:
        return

    for entry in entries:
        # 忽略隱藏檔案或系統資料夾
        if entry.relative_path.startswith("."):
            continue
            
        full_path = current_path + entry.relative_path
        
        # 如果是 USD 檔案，印出
        if entry.relative_path.endswith(".usd"):
            print("  " * depth + f"📄 {entry.relative_path}")
        
        # 如果是目錄（通常沒有點 .），繼續往下挖
        elif "." not in entry.relative_path:
            print("  " * depth + f"📁 {entry.relative_path}/")
            list_sensors_recursive(full_path + "/", depth + 1)

def main():
    assets_root = get_assets_root_path()
    if not assets_root:
        print("錯誤：無法獲取根路徑。")
        return

    # 我們鎖定您最關心的幾個核心感測器目錄
    target_dirs = [
        "/Isaac/Sensors/Ouster/",
        # "/Isaac/Robots/Clearpath/Jackal/",
    ]

    print("\n" + "="*60)
    print("      ISAAC SIM 5.1.0 雲端感測器模型完整清單")
    print("="*60)

    for target in target_dirs:
        print(f"\n[掃描目錄]: {target}")
        list_sensors_recursive(assets_root + target)

    print("\n" + "="*60)
    print("掃描結束。請從上方清單挑選您需要的型號。")

if __name__ == "__main__":
    main()
    simulation_app.close()