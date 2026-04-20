import argparse
import numpy as np
import os
import glob
import re
from scipy.spatial.transform import Rotation as R

from robot import JackalRobot
from data_utils import DataProcessor
from navigator import *
from isaacsim import SimulationApp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="testing")
    parser.add_argument("--roadmap", type=str, required=True, help="Path to a .pkl file OR a directory of case_*.pkl files")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to the USD scene file")
    args = parser.parse_args()

    # 1. 初始化 Isaac Sim 環境與場景 (只執行一次)
    sim_app = SimulationApp({"headless": True, "enable_motion_bvh": True})

    from isaacsim.core.api import SimulationContext
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.rtx")
    import omni.usd
    from pxr import UsdPhysics

    sim = SimulationContext(
        physics_dt=1.0 / 60.0, 
        rendering_dt=1.0 / 30.0, 
        stage_units_in_meters=1.0
    )

    scene_name = os.path.splitext(os.path.basename(args.usd_path))[0]
    omni.usd.get_context().open_stage(args.usd_path)
    stage = omni.usd.get_context().get_stage()
    
    if not stage.GetPrimAtPath("/World/PhysicsScene"):
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    import glob
    import re
    
    pkl_files = []
    is_batch_mode = False

    if os.path.isdir(args.roadmap):
        # 情況 A：傳入的是資料夾 (Dual batch mode)
        pkl_files = glob.glob(os.path.join(args.roadmap, "case_*.pkl"))
        # 確保按照 case_0, case_1 順序執行
        pkl_files.sort(key=lambda f: int(re.search(r'case_(\d+)', os.path.basename(f)).group(1)))
        is_batch_mode = True
        print(f">> 偵測到批次資料夾，共找到 {len(pkl_files)} 個 cases。")
        
    elif os.path.isfile(args.roadmap) and args.roadmap.endswith('.pkl'):
        # 情況 B：傳入的是單一檔案 (Single/Dual shortest mode)
        pkl_files = [args.roadmap]
        is_batch_mode = False
        print(f">> 偵測到單一 Roadmap 檔案。")
        
    else:
        print(">> 錯誤：--roadmap 必須是有效的 .pkl 檔案或包含 case_*.pkl 的資料夾！")
        sim_app.close()
        return

    robots = {}

    # ================= 外層迴圈 =================
    for pkl_file in pkl_files:
        case_name = os.path.splitext(os.path.basename(pkl_file))[0] if is_batch_mode else None
        
        print(f"\n{'='*40}")
        print(f">> 開始處理: {case_name if case_name else '單一任務'}")
        print(f"{'='*40}")

        robot_paths, scenario_type = load_roadmap_scenario(pkl_file)
        navigators = {}
        data_roots = {}

        # 用來記錄哪些機器人是這個 Case 剛生成的
        newly_created_robots = []

        # 1. 宣告與實例化機器人 (但不在此處 Initialize)
        for name, wps in robot_paths.items():
            if not wps or len(wps) < 2:
                continue
                
            start_x, start_y = wps[0][0], wps[0][1]
            start_z = 0.5
            dx = wps[1][0] - wps[0][0]
            dy = wps[1][1] - wps[0][1]
            start_yaw_deg = np.degrees(np.arctan2(dy, dx))

            if name not in robots:
                print(f"[{name}] 首次建立並放置於起點...")
                robot = JackalRobot(
                    name=name, 
                    prim_path=f"/World/{name}", 
                    translation=(start_x, start_y, start_z),
                    orientation=(0, 0, start_yaw_deg)
                )
                robots[name] = robot
                newly_created_robots.append(name) # 標記為新建立

            navigators[name] = SequentialTracker(wps, max_v=1.5, max_w=3.0, debug=False)
            data_roots[name] = DataProcessor.setup_directories(scene_name, args.condition, name, case_name)

        # ==========================================
        # 🚀 關鍵修正：必須先 play 並 step，才能初始化物理
        # ==========================================
        sim.play()
        sim.step(render=True) 

        # create a .txt to record carb settings for debugging

        import carb.settings
        settings = carb.settings.get_settings()

        # RTX Path Tracing Settings
        settings.set_string("/rtx/rendermode", "RaytracedLighting")
        settings.set_float("/rtx/reflections/maxRoughness", 0.8)
        settings.set_int("/rtx/reflections/maxReflectionBounces", 3)
        settings.set_int("/rtx/indirectDiffuse/maxBounces", 3)
        settings.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", 4)

        # # Path Tracing
        # settings.set_string("/rtx/rendermode", "PathTracing")
        # settings.set_int("/rtx/pathtracing/spp", 16)
        # settings.set_int("/rtx/pathtracing/totalSpp", 16)
        # settings.set_int("/rtx/pathtracing/clampSpp", 16)
        # settings.set_int("/rtx/pathtracing/maxBounces", 6)
        # settings.set_int("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
        # settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)
        # settings.set_bool("/rtx/pathtracing/optixDenoiser/useAlbedo", True)
        # settings.set_bool("/rtx/pathtracing/optixDenoiser/useNormals", True)

        # Tone Mapping
        settings.set_float("/rtx/post/tonemap/exposureKey", 0.25)
        settings.set_float("/rtx/post/tonemap/filmIso", 400.0)
        settings.set_float("/rtx/post/tonemap/fNumber", 1.8)


        # print("\n=== Search RTX Exposure Settings ===")
        # post_settings = settings.get("/rtx")
        # def find_exposure(d, current_path):
        #     if isinstance(d, dict):
        #         for k, v in d.items():
        #             # print(f"Found category:{current_path}/{k}")
        #             find_exposure(v, f"{current_path}/{k}")
        #     else:
        #         print(f"Found Setting: {current_path} = {d}")
        #         # write to file
        #         with open("carb_settings_debug.txt", "a") as f:                    
        #             f.write(f"{current_path} = {d}\n")
        # if post_settings:
        #     find_exposure(post_settings, "/rtx")
        # print("=========================================\n")
        # exit(0)
        
        # 2. 初始化「新機器人」的物理視圖
        for name in newly_created_robots:
            robots[name].initialize_physics()

        # 3. 把所有機器人歸位 (Teleport) 並重置速度
        for name, wps in robot_paths.items():
            if not wps or len(wps) < 2:
                continue
            
            start_x, start_y = wps[0][0], wps[0][1]
            start_z = 0.5
            dx = wps[1][0] - wps[0][0]
            dy = wps[1][1] - wps[0][1]
            start_yaw_deg = np.degrees(np.arctan2(dy, dx))

            # 無論是新舊機器人，都在此處統一重置到正確起點，避免慣性殘留
            robots[name].reset_state(
                translation=(start_x, start_y, start_z),
                orientation=(0, 0, start_yaw_deg)
            )
            
        # 4. 短暫暖機 (讓物理引擎穩定，雷達與相機適應新位置)
        for _ in range(30): 
            sim.step(render=True)

        # 每個 Case 開始前，儲存 Calibration
        for name, robot in robots.items():
            if name in navigators:  # 確保該 case 有這個機器人
                robot.save_calibration(data_roots[name])

        # 4. 內層迴圈：單一 Case 的控制與導航
        all_done = False
        # 用來記錄是否已經印過抵達訊息，避免洗版
        robot_reached_flags = {name: False for name in robots.keys()}
        
        while not all_done:
            all_done = True 
            
            # --- [速度同步邏輯] 計算所有機器人的剩餘步數 ---
            rem_steps = {name: max(0, nav.path_length - nav.current_node_idx) for name, nav in navigators.items()}
            max_rem = max(rem_steps.values()) if rem_steps else 1
            
            for name, robot in robots.items():
                if name not in navigators:
                    continue
                    
                nav = navigators[name]
                data = robot.get_data()
                curr_pos = data["pose"][0]
                curr_quat = data["pose"][1] 

                r = R.from_quat(curr_quat)
                curr_yaw = r.as_euler('zyx')[0]

                # 取得原本 Navigator 算出的預設速度
                v, w, reached = nav.compute_command(curr_pos, curr_yaw)
                
                if not reached:
                    all_done = False
                    
                    # --- [速度同步邏輯] 依照剩餘進度比例降速 ---
                    if max_rem > 0:
                        # 計算比例 (最遠的車 scale 為 1.0，越靠近終點的車 scale 越小)
                        # 最低保底 0.3 倍速，避免完全停滯
                        speed_scale = max(0.3, rem_steps[name] / max_rem)
                        v *= speed_scale
                        
                else:
                    # 如果已經抵達，強制停車
                    v, w = 0.0, 0.0
                    
                    # 確保只印出一次抵達訊息 (英文 CLI Info)
                    if not robot_reached_flags[name]:
                        print(f"[INFO] {name} has reached the destination! Waiting for others and continuing to record data...")
                        robot_reached_flags[name] = True

                # 驅動機器人 (走動或煞車)
                robot.drive(v, w)
                
                # 🚀 關鍵修改：不論到了沒，只要整個 case 還沒結束 (all_done 還沒 trigger)，就持續記錄資料
                robot.record_data(data_roots[name])

            # 推動物理時間步長
            sim.step(render=True)
        
        print(f">> {case_name} 任務完成！\n")

    sim_app.close()
    print(">> 所有模擬 Case 執行完畢。")

if __name__ == "__main__":
    main()