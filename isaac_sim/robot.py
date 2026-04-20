import numpy as np
from scipy.spatial.transform import Rotation as R
import config
from data_utils import DataProcessor

class JackalRobot:
    def __init__(self, name, prim_path, translation=(0, 0, 0), orientation=(0, 0, 0)):
        import omni
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.storage.native import get_assets_root_path
        from isaacsim.core.prims import SingleXFormPrim, SingleArticulation
        from omni.isaac.sensor import IMUSensor
        from pxr import Gf, UsdPhysics

        self.name = name
        self.prim_path = prim_path
        
        # 1. 載入機器人
        assets_root = get_assets_root_path()
        jackal_usd = assets_root + "/Isaac/Robots/Clearpath/Jackal/jackal.usd"

        r = R.from_euler('xyz', orientation, degrees=True) 
        q_scipy = r.as_quat()
        q_isaac = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

        create_prim(prim_path=self.prim_path, usd_path=jackal_usd, translation=translation, orientation=q_isaac)
        self.robot_prim = SingleArticulation(prim_path=self.prim_path, name=self.name)

        stage = omni.usd.get_context().get_stage()

        art_prim = stage.GetPrimAtPath(self.prim_path)
        if art_prim.HasAttribute("physxArticulation:solverVelocityIterationCount"):
            art_prim.GetAttribute("physxArticulation:solverVelocityIterationCount").Set(4)

        for prim in stage.Traverse():
            if prim.HasAttribute("drawLines"):
                prim.GetAttribute("drawLines").Set(False)
                
            if prim.HasAttribute("drawPoints"):
                prim.GetAttribute("drawPoints").Set(False)

        # 2. 掛載感測器
        mount_link = f"{self.prim_path}/base_link"
        base_link_prim = stage.GetPrimAtPath(mount_link)

        if not base_link_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(base_link_prim)
        
        # 相機
        self.camera = rep.create.camera(
            parent=mount_link, 
            position=(0.25, 0, 0.25),
            rotation=(0, 0, 180),
            focal_length=1.93,
            horizontal_aperture=3.86, 
            f_stop=0.0, 
            clipping_range=(0.1, 1000000.0)
        )

        self.cam_rp = rep.create.render_product(self.camera, (1280, 800))
        cam_prim_path = str(self.camera.node.get_attribute("outputs:prims").get()[0])
        self.camera_prim = SingleXFormPrim(prim_path=cam_prim_path)

        cam_prim = stage.GetPrimAtPath(cam_prim_path)
        if cam_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            cam_prim.RemoveAPI(UsdPhysics.RigidBodyAPI) 
        
        # RTX LiDAR
        _, sensor_obj = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar", 
            path=f"LiDAR_{self.name}",
            parent=mount_link,
            config="OS0", 
            variant="OS0_REV7_128ch10hz512res", 
            translation=Gf.Vec3d(0.13, 0, 0.37),
            orientation=Gf.Quatd(0.7071068, 0.0, 0.0, -0.7071068),
        )
        self.lidar_path = sensor_obj.GetPath().pathString

        self.lidar_rp = rep.create.render_product(self.lidar_path, [512, 128])
        self.lidar_prim = SingleXFormPrim(prim_path=self.lidar_path)
        
        self.imu = IMUSensor(prim_path=f"{mount_link}/IMU", name=f"{name}_imu", frequency=60)

        self.annos = self._init_annotators()

        self.lidar_buffer = []
        self.valid_buffer_count = 0
        self.last_pc_sum = 0

        self.meta_data = {
                "tight_bbox": {},
                "loose_bbox": {},
                "bbox_3d": {},
                "instance": {},
                "semantic": {}
        }
    
    def initialize_physics(self):
        self.robot_prim.initialize()

    def reset_state(self, translation, orientation):
        from scipy.spatial.transform import Rotation as R
        import numpy as np

        # 1. 瞬間移動 (Teleport)
        r = R.from_euler('xyz', orientation, degrees=True) 
        q_scipy = r.as_quat()
        # Isaac Sim 接受的四元數格式為 [w, x, y, z]
        q_isaac = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        self.robot_prim.set_world_pose(position=translation, orientation=q_isaac)

        # 2. 強制重置關節速度與本體速度為 0，避免從上一個 case 帶著慣性飛出去
        velocities = np.zeros(self.robot_prim.num_dof)
        self.robot_prim.set_joint_velocities(velocities)
        self.robot_prim.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.robot_prim.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        # 3. 清空暫存的資料緩衝區
        self.lidar_buffer = []
        self.valid_buffer_count = 0
        self.last_pc_sum = 0

    def drive(self, v, omega):
        # 🚀 引入 Isaac Sim 官方的 ArticulationAction
        from isaacsim.core.utils.types import ArticulationAction
        
        v_left = v - (omega * config.WHEEL_BASE / 2.0)
        v_right = v + (omega * config.WHEEL_BASE / 2.0)
        w_left = v_left / config.WHEEL_RADIUS
        w_right = v_right / config.WHEEL_RADIUS

        velocities = np.zeros(self.robot_prim.num_dof)
        for i, dof in enumerate(self.robot_prim.dof_names):
            if "left" in dof and "wheel" in dof:
                velocities[i] = w_left
            elif "right" in dof and "wheel" in dof:
                velocities[i] = w_right

        # 🚀 關鍵修正 1：打包成 ArticulationAction，交由系統的 PD 控制器平滑驅動/煞車
        action = ArticulationAction(joint_velocities=velocities)
        self.robot_prim.apply_action(action)

        # 🚀 關鍵修正 2：如果收到靜止指令，直接把本體的線性與角速度慣性清零，徹底鎖死
        if v == 0.0 and omega == 0.0:
            self.robot_prim.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            self.robot_prim.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    def _init_annotators(self):
        import omni.replicator.core as rep
        # lidar_anno = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
        lidar_anno = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
        lidar_anno.attach([self.lidar_rp])
        
        visual_annos = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            "semantic": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
            "instance": rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
            "tight_bbox": rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
            "loose_bbox": rep.AnnotatorRegistry.get_annotator("bounding_box_2d_loose"),
            "bbox_3d": rep.AnnotatorRegistry.get_annotator("bounding_box_3d"),
            "lidar_raw": lidar_anno
        }
        for a in list(visual_annos.values())[:-1]:
            a.attach([self.cam_rp])
        return visual_annos

    def get_data(self):
        pos, orient_w_first = self.robot_prim.get_world_pose()
        orient = np.array([orient_w_first[1], orient_w_first[2], orient_w_first[3], orient_w_first[0]])

        return {
            "pose": (pos, orient),
            "imu": self.imu.get_current_frame(),
            "lidar": self.annos["lidar_raw"].get_data(),
            "rgb": self.annos["rgb"].get_data(),
            "depth": self.annos["depth"].get_data(),
            "semantic": self.annos["semantic"].get_data(),
            "instance": self.annos["instance"].get_data(),
            "tight_bbox": self.annos["tight_bbox"].get_data(),
            "loose_bbox": self.annos["loose_bbox"].get_data(),
            "bbox_3d": self.annos["bbox_3d"].get_data()
        }

    def record_data(self, save_root):
        import omni.timeline
        
        data = self.get_data()

        current_pc = data['lidar']['data'].reshape(-1, 3)
        current_pc_sum = np.sum(current_pc)

        # if current_pc_sum == self.last_pc_sum:
        #     return False
        
        self.last_pc_sum = current_pc_sum

        self.lidar_buffer.append(current_pc)
        
        if self.valid_buffer_count < 5:
            self.valid_buffer_count += 1
        else:
            pc_data = np.vstack(self.lidar_buffer)

            data['lidar']['data'] = pc_data

            timeline = omni.timeline.get_timeline_interface()
            sim_time = timeline.get_current_time()
            
            timestamp_us = int(sim_time * 1_000)
            
            timestamp_str = f"{timestamp_us:07d}" 
            
            data["timestamp"] = sim_time
            
            occ3d, occ2d = DataProcessor.compute_occupancy(pc_data, config.GRID_CFG)
            
            DataProcessor.save_frame(data, occ3d, occ2d, save_root, timestamp_str)
            
            self.valid_buffer_count = 0
            self.lidar_buffer = []
            return True 
                
        return False
        
    def save_calibration(self, save_root):
        import omni.usd
        from pxr import UsdGeom
        from scipy.spatial.transform import Rotation as R
        import numpy as np
        import os
        import yaml

        # --- 1. 取得 Base (Robot) to World 的 4x4 轉換矩陣 ---
        base_pos, base_orient = self.robot_prim.get_world_pose()
        base_quat_scipy = [base_orient[1], base_orient[2], base_orient[3], base_orient[0]]
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(base_quat_scipy).as_matrix()
        T_world_to_base[:3, 3] = base_pos

        # --- 2. 取得 Camera (USD) to World 的 4x4 轉換矩陣 ---
        cam_pos, cam_orient = self.camera_prim.get_world_pose()
        cam_quat_scipy = [cam_orient[1], cam_orient[2], cam_orient[3], cam_orient[0]]
        T_world_to_cam_usd = np.eye(4)
        T_world_to_cam_usd[:3, :3] = R.from_quat(cam_quat_scipy).as_matrix()
        T_world_to_cam_usd[:3, 3] = cam_pos

        # --- 3. 處理 USD 到 ROS Optical Frame 的旋轉 (繞 X 軸旋轉 180 度) ---
        T_usd_to_ros_optical = np.eye(4)
        T_usd_to_ros_optical[:3, :3] = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        
        # 結合得到 Camera (ROS Optical) to World 矩陣
        T_world_to_cam_ros = T_world_to_cam_usd @ T_usd_to_ros_optical

        # --- 4. 核心：計算 Camera to Base 的「靜態相對外參」 ---
        # T_base_to_cam = (T_world_to_base)^-1 * T_world_to_cam
        T_base_to_cam = np.linalg.inv(T_world_to_base) @ T_world_to_cam_ros

        # --- 5. 處理內參 (K Matrix) ---
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(self.camera_prim.prim_path)
        
        if not cam_prim.IsA(UsdGeom.Camera):
            for child in cam_prim.GetChildren():
                if child.IsA(UsdGeom.Camera):
                    cam_prim = child
                    break
                    
        usd_camera = UsdGeom.Camera(cam_prim)
        focal_length = usd_camera.GetFocalLengthAttr().Get() or 1.93
        horiz_aperture = usd_camera.GetHorizontalApertureAttr().Get() or 3.8
        
        width, height = 1280, 800 
        fx = width * (focal_length / horiz_aperture)
        fy = fx  
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx,  0.0, cx],
            [0.0,  fy, cy],
            [0.0, 0.0, 1.0]
        ])
        P = np.hstack((K, np.zeros((3, 1))))

        # --- 6. 組裝成 ROS 標準 YAML 格式 ---
        calib_data = {
            "image_width": width,
            "image_height": height,
            "camera_name": self.name + "_camera",
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": K.flatten().tolist()
            },
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {
                "rows": 1,
                "cols": 5,
                "data": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": np.eye(3).flatten().tolist()
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": P.flatten().tolist()
            },
            # 🚀 這裡不再是 camera_to_world，而是純淨的 camera_to_base
            "camera_to_base": {
                "rows": 4,
                "cols": 4,
                "data": T_base_to_cam.flatten().tolist()
            }
        }
        
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "camera_info.yaml")
        with open(save_path, 'w') as f:
            yaml.dump(calib_data, f, default_flow_style=None, sort_keys=False)
        print(f"[{self.name}] 相機靜態校正檔已儲存至: {save_path}")

        # --- 1. 取得 Base (Robot) to World ---
        base_pos, base_orient = self.robot_prim.get_world_pose()
        base_quat_scipy = [base_orient[1], base_orient[2], base_orient[3], base_orient[0]]
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(base_quat_scipy).as_matrix()
        T_world_to_base[:3, 3] = base_pos

        # --- 2. 取得 LiDAR to World ---
        lidar_pos, lidar_orient = self.lidar_prim.get_world_pose()
        lidar_quat_scipy = [lidar_orient[1], lidar_orient[2], lidar_orient[3], lidar_orient[0]]
        T_world_to_lidar = np.eye(4)
        T_world_to_lidar[:3, :3] = R.from_quat(lidar_quat_scipy).as_matrix()
        T_world_to_lidar[:3, 3] = lidar_pos

        # --- 3. 計算 LiDAR to Base 的「靜態相對外參」 ---
        T_base_to_lidar = np.linalg.inv(T_world_to_base) @ T_world_to_lidar

        # --- 4. 處理 LiDAR 內參/規格 ---
        channels = 128
        horizontal_resolution = 512
        rotation_rate_hz = 10.0

        calib_data = {
            "sensor_name": self.name + "_lidar",
            "sensor_model": "Ouster OS1_REV7",
            "hardware_config": {
                "channels": channels,
                "horizontal_resolution": horizontal_resolution,
                "rotation_rate_hz": rotation_rate_hz,
            },
            # 🚀 這裡改為 lidar_to_base
            "lidar_to_base": {
                "rows": 4,
                "cols": 4,
                "data": T_base_to_lidar.flatten().tolist()
            }
        }
        
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "lidar_info.yaml")
        with open(save_path, 'w') as f:
            yaml.dump(calib_data, f, default_flow_style=None, sort_keys=False)
        print(f"[{self.name}] LiDAR 靜態校正檔已儲存至: {save_path}")