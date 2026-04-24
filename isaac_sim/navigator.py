import numpy as np
import pickle

class SequentialTracker:
    def __init__(self, path_waypoints, max_v=1.5, max_w=3.0, lookahead_dist=0.5):
        self.path = np.array(path_waypoints)[:, :2] 
        self.max_v = max_v
        self.max_w = max_w
        self.Ld = lookahead_dist  
        self.kp_w = 3.0
        
        self.current_node_idx = 0 
        self.path_length = len(self.path)

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_command(self, current_pos, current_yaw):
        dist_to_goal = np.hypot(self.path[-1][0] - current_pos[0], self.path[-1][1] - current_pos[1])

        if dist_to_goal < 0.35:  
            return 0.0, 0.0, True

        search_range = min(self.path_length, self.current_node_idx + 40)
        min_dist = float('inf')
        closest_idx = self.current_node_idx
        
        for i in range(self.current_node_idx, search_range):
            dist = np.hypot(self.path[i][0] - current_pos[0], self.path[i][1] - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        self.current_node_idx = closest_idx 

        target_idx = self.current_node_idx
        for i in range(self.current_node_idx, self.path_length):
            dist = np.hypot(self.path[i][0] - current_pos[0], self.path[i][1] - current_pos[1])
            if dist >= self.Ld:
                target_idx = i
                break
        
        if target_idx == self.current_node_idx and self.current_node_idx == self.path_length - 1:
             target_idx = self.path_length - 1

        target_pos = np.copy(self.path[target_idx])

        if target_idx == self.path_length - 1 and self.path_length >= 2:
            dx_last = self.path[-1][0] - self.path[-2][0]
            dy_last = self.path[-1][1] - self.path[-2][1]
            norm = np.hypot(dx_last, dy_last)
            
            if norm > 1e-6:
                dir_x = dx_last / norm
                dir_y = dy_last / norm
                target_pos[0] += dir_x * self.Ld
                target_pos[1] += dir_y * self.Ld

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        target_angle = np.arctan2(dy, dx)
        alpha = self.normalize_angle(target_angle - current_yaw)

        if abs(alpha) > (np.pi / 2): 
            if dist_to_goal < self.Ld:
                v_cmd = self.max_v * 0.2
                w_cmd = np.clip(self.kp_w * alpha, -self.max_w, self.max_w)
            else:
                v_cmd = 0.0
                w_cmd = np.sign(alpha) * self.max_w
        else:
            speed_factor = max(0.2, 1.0 - (abs(alpha) / (np.pi / 3))) 
            
            if dist_to_goal < 1.5:
                speed_factor *= max(0.3, dist_to_goal / 1.5)
                
            v_cmd = self.max_v * speed_factor
            w_cmd = np.clip(self.kp_w * alpha, -self.max_w, self.max_w) 

        return v_cmd, w_cmd, False

def load_roadmap_scenario(file_path):
    with open(file_path, 'rb') as f:
        roadmap_data = pickle.load(f)

    robot_paths = {}
    scenario_type = "unknown"

    if 'R1_waypoints' in roadmap_data and 'R2_waypoints' in roadmap_data:
        robot_paths['Jackal_R1'] = roadmap_data['R1_waypoints']
        robot_paths['Jackal_R2'] = roadmap_data['R2_waypoints']
        scenario_type = "dual"
    elif 'waypoint_path' in roadmap_data:
        robot_paths['Jackal_Single'] = roadmap_data['waypoint_path']
        scenario_type = "single"
    else:
        raise ValueError("無法辨識的 Roadmap 格式")

    return robot_paths, scenario_type