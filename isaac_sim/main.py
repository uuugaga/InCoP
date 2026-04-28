import argparse
import numpy as np
import os
import glob
import re
import shutil
from scipy.spatial.transform import Rotation as R

import config
from robot import JackalRobot
from data_utils import DataProcessor
from navigator import *
from isaacsim import SimulationApp
from time import sleep


def copy_split_jsons_to_dataset(scene_name, roadmap_path):
    split_names = ("path_case_distance_split.json", "path_case_shadow_split.json")
    search_dir = roadmap_path if os.path.isdir(roadmap_path) else os.path.dirname(roadmap_path)
    search_dir = os.path.abspath(search_dir)
    dual_dir = None

    while True:
        if all(os.path.isfile(os.path.join(search_dir, split_name)) for split_name in split_names):
            dual_dir = search_dir
            break

        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir

    if dual_dir is None:
        raise FileNotFoundError(
            "Required split json files not found near roadmap path: "
            f"{roadmap_path}. Expected files: {', '.join(split_names)}"
        )

    dataset_scene_dir = os.path.join(config.BASE_SAVE_PATH, scene_name)
    os.makedirs(dataset_scene_dir, exist_ok=True)

    for split_name in split_names:
        src = os.path.join(dual_dir, split_name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Required split json not found: {src}")

        dst = os.path.join(dataset_scene_dir, split_name)
        shutil.copy2(src, dst)
        print(f">> Copied split json to dataset scene folder: {dst}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="testing")
    parser.add_argument("--roadmap", type=str, required=True, help="Path to a .pkl file OR a directory of case_*.pkl files")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to the USD scene file")
    args = parser.parse_args()

    scene_name = os.path.splitext(os.path.basename(args.usd_path))[0]

    pkl_files = []
    is_batch_mode = False

    if os.path.isdir(args.roadmap):
        pkl_files = glob.glob(os.path.join(args.roadmap, "case_*.pkl"))
        pkl_files.sort(key=lambda f: int(re.search(r'case_(\d+)', os.path.basename(f)).group(1)))
        is_batch_mode = True
        print(f">> Found {len(pkl_files)} roadmap files in directory. Batch mode enabled.")
        if not pkl_files:
            raise FileNotFoundError(f"No case_*.pkl files found in roadmap directory: {args.roadmap}")
        
    elif os.path.isfile(args.roadmap) and args.roadmap.endswith('.pkl'):
        pkl_files = [args.roadmap]
        is_batch_mode = False
        print(f">> Found single roadmap file. Running in single-case mode.")
        
    else:
        print(">> Error: roadmap argument must be a .pkl file or a directory containing case_*.pkl files.")
        return

    copy_split_jsons_to_dataset(scene_name, args.roadmap)

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

    omni.usd.get_context().open_stage(args.usd_path)
    stage = omni.usd.get_context().get_stage()
    
    if not stage.GetPrimAtPath("/World/PhysicsScene"):
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    robots = {}

    for pkl_file in pkl_files:
        case_name = os.path.splitext(os.path.basename(pkl_file))[0] if is_batch_mode else None
        
        print(f"\n{'='*40}")
        print(f">> Starting simulation for case: {case_name if case_name else 'Single Case'}")
        print(f"{'='*40}")

        robot_paths, scenario_type = load_roadmap_scenario(pkl_file)
        navigators = {}
        data_roots = {}


        newly_created_robots = []
        for name, wps in robot_paths.items():
            if not wps or len(wps) < 2:
                continue
                
            start_x, start_y = wps[0][0], wps[0][1]
            start_z = 0.5
            dx = wps[1][0] - wps[0][0]
            dy = wps[1][1] - wps[0][1]
            start_yaw_deg = np.degrees(np.arctan2(dy, dx))

            if name not in robots:
                print(f"[{name}] First time seeing this robot. Creating in simulation...")
                robot = JackalRobot(
                    name=name, 
                    prim_path=f"/World/{name}", 
                    translation=(start_x, start_y, start_z),
                    orientation=(0, 0, start_yaw_deg)
                )
                robots[name] = robot
                newly_created_robots.append(name) 

            navigators[name] = SequentialTracker(wps, max_v=1.5, max_w=3.0)
            data_roots[name] = DataProcessor.setup_directories(scene_name, args.condition, name, case_name)

        sim.play()
        sim.step(render=True) 

        import carb.settings
        settings = carb.settings.get_settings()

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

        # If is hospital scene use this setting to brighten the scene
        if "hospital" in scene_name.lower():
            # Tone Mapping
            settings.set_float("/rtx/post/tonemap/filmIso", 400.0)
            settings.set_float("/rtx/post/tonemap/fNumber", 1.8)

        if "office" in scene_name.lower():
            # Tone Mapping
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
        
        for name in newly_created_robots:
            robots[name].initialize_physics()

        for name, wps in robot_paths.items():
            if not wps or len(wps) < 2:
                continue
            
            start_x, start_y = wps[0][0], wps[0][1]
            start_z = 0.5
            dx = wps[1][0] - wps[0][0]
            dy = wps[1][1] - wps[0][1]
            start_yaw_deg = np.degrees(np.arctan2(dy, dx))

            robots[name].reset_state(
                translation=(start_x, start_y, start_z),
                orientation=(0, 0, start_yaw_deg)
            )
            
        for _ in range(30): 
            sim.step(render=True)

        for name, robot in robots.items():
            if name in navigators:
                robot.save_calibration(data_roots[name])


        all_done = False
        latest_ready = {}
        robot_reached_flags = {name: False for name in robots.keys()}
        
        while not all_done:
            all_done = True 
            
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

                v, w, reached = nav.compute_command(curr_pos, curr_yaw)
                
                if not reached:
                    all_done = False
                    
                    if max_rem > 0:
                        speed_scale = max(0.3, rem_steps[name] / max_rem)
                        v *= speed_scale
                        
                else:
                    v, w = 0.0, 0.0
                    if not robot_reached_flags[name]:
                        print(f"[INFO] {name} has reached the destination! Waiting for others and continuing to record data...")
                        robot_reached_flags[name] = True

                robot.drive(v, w)

                ready, prepared_data, drop_reason = robot.prepare_record_data()
                if ready:
                    sample_time, sample_timestamp = robot._timestamp_str_from_timeline()
                    latest_ready[name] = {
                        "data": prepared_data,
                        "sim_time": sample_time,
                        "timestamp_str": sample_timestamp,
                    }

            sync_names = [name for name in navigators.keys() if name in latest_ready]
            if sync_names and all(name in latest_ready for name in navigators.keys()):
                reference_name = "Jackal_R1" if "Jackal_R1" in latest_ready else sorted(latest_ready)[0]
                reference_item = latest_ready[reference_name]
                for name in navigators.keys():
                    robots[name].save_prepared_data(
                        data_roots[name],
                        latest_ready[name]["data"],
                        timestamp_str=reference_item["timestamp_str"],
                        sim_time=reference_item["sim_time"],
                    )
                latest_ready.clear()

            sim.step(render=True)
            sleep(0.05) # Sleep for lidar data sync, or you will get empty lidar point clouds due to the current data retrieval implementation. This can be removed after we implement a better data retrieval method.
        
        print(f">> {case_name} Traversal Completed! Saving data and moving to next case...")

    sim_app.close()
    print(">> All simulations completed. Exiting.")

if __name__ == "__main__":
    main()
