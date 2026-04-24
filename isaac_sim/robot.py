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

        mount_link = f"{self.prim_path}/base_link"
        base_link_prim = stage.GetPrimAtPath(mount_link)

        if not base_link_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(base_link_prim)
        
        self.camera = rep.create.camera(
            parent=mount_link, 
            position=(0.25, 0, 0.25),
            rotation=(0, 0, 180),
            focal_length=1.93,
            horizontal_aperture=3.86, 
            f_stop=0.0, 
            clipping_range=(0.01, 100.0)
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

        self.meta_data = {
                "tight_bbox": {},
                "loose_bbox": {},
                "bbox_3d": {},
                "semantic": {},
        }
    
    def initialize_physics(self):
        self.robot_prim.initialize()

    def reset_state(self, translation, orientation):
        from scipy.spatial.transform import Rotation as R
        import numpy as np

        r = R.from_euler('xyz', orientation, degrees=True) 
        q_scipy = r.as_quat()

        q_isaac = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        self.robot_prim.set_world_pose(position=translation, orientation=q_isaac)

        velocities = np.zeros(self.robot_prim.num_dof)
        self.robot_prim.set_joint_velocities(velocities)
        self.robot_prim.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.robot_prim.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        self.lidar_buffer = []
        self.valid_buffer_count = 0
        self.last_pc_sum = 0

    def drive(self, v, omega):
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

        action = ArticulationAction(joint_velocities=velocities)
        self.robot_prim.apply_action(action)

        if v == 0.0 and omega == 0.0:
            self.robot_prim.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            self.robot_prim.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    def _init_annotators(self):
        import omni.replicator.core as rep
        lidar_anno = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
        lidar_anno.attach([self.lidar_rp])
        
        visual_annos = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            "semantic": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
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
            "tight_bbox": self.annos["tight_bbox"].get_data(),
            "loose_bbox": self.annos["loose_bbox"].get_data(),
            "bbox_3d": self.annos["bbox_3d"].get_data()
        }

    @staticmethod
    def _finite_xyz(pc):
        if pc is None:
            return np.empty((0, 3), dtype=np.float32)
        pc = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
        return pc[np.all(np.isfinite(pc), axis=1)]

    @staticmethod
    def _timestamp_str_from_timeline():
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        sim_time = timeline.get_current_time()
        return sim_time, f"{int(sim_time * 1_000):07d}"

    def _discard_current_timestamp(self, save_root, reason):
        sim_time, timestamp_str = self._timestamp_str_from_timeline()
        removed = DataProcessor.delete_frame(save_root, timestamp_str)
        print(f"[{self.name}] Drop timestamp {timestamp_str} ({sim_time:.3f}s): {reason}; removed {removed} files/rows")

    def _lidar_to_base_matrix(self):
        lidar_pos, lidar_orient = self.lidar_prim.get_world_pose()
        lidar_quat = [lidar_orient[1], lidar_orient[2], lidar_orient[3], lidar_orient[0]]
        T_world_to_lidar = np.eye(4)
        T_world_to_lidar[:3, :3] = R.from_quat(lidar_quat).as_matrix()
        T_world_to_lidar[:3, 3] = lidar_pos

        base_pos, base_orient = self.robot_prim.get_world_pose()
        base_quat = [base_orient[1], base_orient[2], base_orient[3], base_orient[0]]
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(base_quat).as_matrix()
        T_world_to_base[:3, 3] = base_pos

        return np.linalg.inv(T_world_to_base) @ T_world_to_lidar

    def _front_lidar_fov(self, pc, fov_deg=None):
        pc = self._finite_xyz(pc)
        if pc.size == 0:
            return pc, np.empty(0, dtype=np.float32)

        fov = float(fov_deg if fov_deg is not None else getattr(config, "LIDAR_RECORD_FOV_DEG", 90.0))
        half_fov = np.deg2rad(fov / 2.0)

        T_lidar_to_base = self._lidar_to_base_matrix()
        pts_lidar = np.hstack((pc, np.ones((pc.shape[0], 1), dtype=pc.dtype)))
        pts_base = (T_lidar_to_base @ pts_lidar.T).T[:, :3]
        yaw_base = np.arctan2(pts_base[:, 1], pts_base[:, 0])
        forward = (pts_base[:, 0] > 0.05) & (np.abs(yaw_base) <= half_fov)
        return pc[forward], yaw_base[forward]

    def _front_lidar_fov_is_complete(self, yaw):
        yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
        if yaw.size == 0:
            return False, f"no points inside {config.LIDAR_RECORD_FOV_DEG:g} deg front FOV"

        fov = float(getattr(config, "LIDAR_RECORD_FOV_DEG", 90.0))
        half_fov = np.deg2rad(fov / 2.0)
        bins = int(getattr(config, "LIDAR_FOV_COMPLETENESS_BINS", 9))
        min_points = int(getattr(config, "LIDAR_FOV_COMPLETENESS_MIN_POINTS", 8))
        min_ratio = float(getattr(config, "LIDAR_FOV_COMPLETENESS_MIN_RATIO", 0.20))

        counts, _ = np.histogram(yaw, bins=bins, range=(-half_fov, half_fov))
        typical = np.median(counts[counts > 0]) if np.any(counts > 0) else 0
        expected = max(min_points, int(typical * min_ratio))

        if np.any(counts < expected):
            return False, f"uneven front FOV lidar counts={counts.tolist()} threshold={expected}"
        return True, f"front FOV lidar counts={counts.tolist()}"

    def prepare_record_data(self):
        data = self.get_data()

        lidar_data = data.get("lidar", {})
        lidar_points = lidar_data.get("data") if isinstance(lidar_data, dict) else lidar_data
        current_pc = self._finite_xyz(lidar_points)
        if current_pc.size == 0:
            return False, None, "empty point cloud"

        pc_data, yaw_fov = self._front_lidar_fov(current_pc)
        is_complete, reason = self._front_lidar_fov_is_complete(yaw_fov)
        if not is_complete:
            return False, None, reason

        data["lidar"]["data"] = pc_data

        return True, data, None

    def save_prepared_data(self, save_root, data, timestamp_str=None, sim_time=None):
        if sim_time is None or timestamp_str is None:
            sim_time, timestamp_str = self._timestamp_str_from_timeline()
        data["timestamp"] = sim_time

        DataProcessor.save_frame(data, save_root, timestamp_str)
        return True
        
    def save_calibration(self, save_root):
        import omni.usd
        from pxr import UsdGeom
        from scipy.spatial.transform import Rotation as R
        import numpy as np
        import os
        import yaml
        from frame_utils import matrix_to_yaml_dict

        def transform_to_yaml(parent_frame, child_frame, transform_matrix):
            quat_xyzw = R.from_matrix(transform_matrix[:3, :3]).as_quat()
            trans = transform_matrix[:3, 3]
            return {
                "header": {
                    "frame_id": parent_frame
                },
                "child_frame_id": child_frame,
                "transform": {
                    "translation": {
                        "x": float(trans[0]),
                        "y": float(trans[1]),
                        "z": float(trans[2])
                    },
                    "rotation": {
                        "x": float(quat_xyzw[0]),
                        "y": float(quat_xyzw[1]),
                        "z": float(quat_xyzw[2]),
                        "w": float(quat_xyzw[3])
                    }
                },
                "matrix": {
                    "rows": 4,
                    "cols": 4,
                    "data": [float(v) for v in transform_matrix.flatten().tolist()]
                }
            }

        base_pos, base_orient = self.robot_prim.get_world_pose()
        base_quat_scipy = [base_orient[1], base_orient[2], base_orient[3], base_orient[0]]
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(base_quat_scipy).as_matrix()
        T_world_to_base[:3, 3] = base_pos

        cam_pos, cam_orient = self.camera_prim.get_world_pose()
        cam_quat_scipy = [cam_orient[1], cam_orient[2], cam_orient[3], cam_orient[0]]
        T_world_to_cam_usd = np.eye(4)
        T_world_to_cam_usd[:3, :3] = R.from_quat(cam_quat_scipy).as_matrix()
        T_world_to_cam_usd[:3, 3] = cam_pos

        # --- Convert Isaac/Replicator camera local axes to camera optical axes ---
        # Replicator's camera pose is not already in the pinhole optical frame.
        # The exported intrinsics and RGB/depth projection use optical coordinates:
        # x right, y down, z forward.
        T_usd_to_ros_optical = np.eye(4)
        T_usd_to_ros_optical[:3, :3] = np.array([
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
        ])
        
        T_world_to_cam_ros = T_world_to_cam_usd @ T_usd_to_ros_optical

        T_camera_optical_to_base = np.linalg.inv(T_world_to_base) @ T_world_to_cam_ros

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

        camera_to_base_yaml = matrix_to_yaml_dict(T_camera_optical_to_base)
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
            # Camera stays in optical frame. Write both the preferred key and the legacy alias.
            "camera_optical_to_base": camera_to_base_yaml,
            "camera_to_base": camera_to_base_yaml,
        }
        
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "camera_info.yaml")
        with open(save_path, 'w') as f:
            yaml.dump(calib_data, f, default_flow_style=None, sort_keys=False)
        print(f"[{self.name}] Saved to: {save_path}")

        base_pos, base_orient = self.robot_prim.get_world_pose()
        base_quat_scipy = [base_orient[1], base_orient[2], base_orient[3], base_orient[0]]
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(base_quat_scipy).as_matrix()
        T_world_to_base[:3, 3] = base_pos

        lidar_pos, lidar_orient = self.lidar_prim.get_world_pose()
        lidar_quat_scipy = [lidar_orient[1], lidar_orient[2], lidar_orient[3], lidar_orient[0]]
        T_world_to_lidar = np.eye(4)
        T_world_to_lidar[:3, :3] = R.from_quat(lidar_quat_scipy).as_matrix()
        T_world_to_lidar[:3, 3] = lidar_pos

        T_base_to_lidar = np.linalg.inv(T_world_to_base) @ T_world_to_lidar

        channels = 128
        horizontal_resolution = 512
        rotation_rate_hz = 10.0

        lidar_to_base_yaml = matrix_to_yaml_dict(T_base_to_lidar)
        calib_data = {
            "sensor_name": self.name + "_lidar",
            "sensor_model": "Ouster OS1_REV7",
            "hardware_config": {
                "channels": channels,
                "horizontal_resolution": horizontal_resolution,
                "rotation_rate_hz": rotation_rate_hz,
            },
            # LiDAR is saved in its native Isaac Sim sensor frame.
            "lidar_sensor_to_base": lidar_to_base_yaml,
            "lidar_to_base": lidar_to_base_yaml,
        }
        
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "lidar_info.yaml")
        with open(save_path, 'w') as f:
            yaml.dump(calib_data, f, default_flow_style=None, sort_keys=False)
        print(f"[{self.name}] LiDAR static calibration saved to: {save_path}")

        T_base_to_imu = np.eye(4)
        tf_static_data = {
            "transforms": [
                transform_to_yaml("base_link", "camera_color_optical_frame", T_camera_optical_to_base),
                transform_to_yaml("base_link", "lidar_link", T_base_to_lidar),
                transform_to_yaml("base_link", "imu_link", T_base_to_imu),
            ]
        }

        tf_path = os.path.join(save_root, "tf_static.yaml")
        with open(tf_path, 'w') as f:
            yaml.dump(tf_static_data, f, default_flow_style=None, sort_keys=False)
        print(f"[{self.name}] TF static file saved to: {tf_path}")
