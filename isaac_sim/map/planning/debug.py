import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle, cv2, os, networkx as nx, random
import yaml
from scipy.spatial import KDTree
from matplotlib.patches import Polygon

class CoverageSolver:
    def __init__(self, 
                 png_path, 
                 yaml_path, 
                 roadmap_path, 
                 max_dist_m=10.0, 
                 fov_deg=90.0,
                 boundary_step_m=0.1,
                 obs_interval_m=0.1,
                 min_grazing_deg=15.0):
        """
        Initialize the Coverage Solver with tunable parameters.
        """
        # 1. Basic setup and output directory preparation
        self.scene_name = os.path.splitext(os.path.basename(png_path))[0]
        self.output_dir = os.path.join("trajectory", self.scene_name)
        self.debug_dir = os.path.join(self.output_dir, "debug") 
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True) # 統一存在此資料夾
        
        # 2. Load map metadata
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.res, self.org = cfg['resolution'], cfg['origin']
        
        # 3. Load map and build collision mask (0: Free, 1: Obstacle)
        raw_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        self.flipped_img = cv2.flip(raw_img, 0)
        self.h, self.w = self.flipped_img.shape[:2]
        self.obs_mask = (self.flipped_img < 200).astype(np.uint8)
        
        # 4. Load Roadmap, RELABEL NODES, and repair missing weights
        with open(roadmap_path, 'rb') as f:
            raw_G = pickle.load(f)
            
        # [修改點] 強制將 Node 重新命名為 0, 1, 2, ..., n-1，並保留原本的屬性 (如 pos)
        self.G = nx.convert_node_labels_to_integers(raw_G, first_label=0)
        print(f">> Graph loaded and relabeled. Total nodes: {self.G.number_of_nodes()} (0 to {self.G.number_of_nodes()-1})")
        
        self._repair_graph_weights()
        
        # 5. Store algorithm parameters
        self.max_dist_m = max_dist_m
        self.fov_deg = fov_deg
        self.boundary_step_m = boundary_step_m
        self.obs_interval_m = obs_interval_m
        
        self.min_grazing_deg = min_grazing_deg
        self.parallel_tol_cos = -np.sin(np.radians(self.min_grazing_deg))
        
        # 6. Extract target boundary points using the specified density
        self.boundary_pts = self.extract_boundary_points(step_m=self.boundary_step_m)
        self.target_tree = KDTree(self.boundary_pts)
        
        self.edge_coverage = {} 

    def _repair_graph_weights(self):
        added_weight_count = 0
        for u, v in self.G.edges():
            if 'weight' not in self.G[u][v]:
                p1 = np.array(self.G.nodes[u]['pos'])
                p2 = np.array(self.G.nodes[v]['pos'])
                self.G[u][v]['weight'] = np.linalg.norm(p1 - p2)
                added_weight_count += 1
        if added_weight_count > 0:
            print(f">> Graph repaired: Added weight to {added_weight_count} edges.")

    def g2w(self, r, c): 
        return [self.org[0] + c * self.res, self.org[1] + r * self.res]
    
    def w2g(self, x, y): 
        return int((y - self.org[1]) / self.res), int((x - self.org[0]) / self.res)

    def extract_boundary_points(self, step_m=0.1):
        pad_size = 1
        padded_mask = np.pad(self.obs_mask, pad_width=pad_size, mode='constant', constant_values=1)
        
        contours, _ = cv2.findContours(
            padded_mask, 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_NONE 
        )
        
        all_pts = []
        step_px = max(1, int(step_m / self.res))
        
        for cnt in contours:
            for i in range(0, len(cnt), step_px):
                c_pad, r_pad = cnt[i][0]
                c = c_pad - pad_size
                r = r_pad - pad_size
                
                if c < 0 or r < 0 or c >= self.w or r >= self.h:
                    continue
                    
                all_pts.append(self.g2w(r, c))
                
        print(f">> Extracted {len(all_pts)} fine-grained boundary points (step: {step_m}m).")
        return np.array(all_pts)

    def _check_visibility(self, start_world, end_world):
        r1, c1 = self.w2g(*start_world)
        r2, c2 = self.w2g(*end_world)
        
        # 1. Grazing Angle Filtering
        r_up = max(r2 - 1, 0); r_dn = min(r2 + 1, self.h - 1)
        c_lt = max(c2 - 1, 0); c_rt = min(c2 + 1, self.w - 1)

        dy = int(self.obs_mask[r_dn, c2]) - int(self.obs_mask[r_up, c2])
        dx = int(self.obs_mask[r2, c_rt]) - int(self.obs_mask[r2, c_lt])
        N_x, N_y = -dx, -dy 
        R_x, R_y = c2 - c1, r2 - r1
        
        norm_N, norm_R = np.hypot(N_x, N_y), np.hypot(R_x, R_y)
        if norm_N > 0 and norm_R > 0:
            cos_theta = (R_x * N_x + R_y * N_y) / (norm_R * norm_N)
            if cos_theta > self.parallel_tol_cos:
                return False 

        dist_px = np.hypot(r2 - r1, c2 - c1)
        if dist_px < 2.0: return True 
        
        # 2. High-Fidelity Raycasting
        num_samples = int(dist_px * 3)
        rows = np.linspace(r1, r2, num_samples)
        cols = np.linspace(c1, c2, num_samples)
        
        check_rows = np.clip(np.round(rows).astype(int), 0, self.h - 1)
        check_cols = np.clip(np.round(cols).astype(int), 0, self.w - 1)
        
        margin = max(5, int(num_samples * 0.05)) 
        if len(check_rows) > margin * 2:
            check_rows = check_rows[margin:-margin]
            check_cols = check_cols[margin:-margin]

        hit_count = np.sum(self.obs_mask[check_rows, check_cols] == 1)
        
        allowed_hits = 2 
        
        if hit_count > allowed_hits: 
            return False
            
        return True

    def _cast_ray(self, start_world, angle_deg):
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad) * self.res, np.sin(rad) * self.res
        
        curr_x, curr_y = start_world[0], start_world[1]
        max_steps = int(self.max_dist_m / self.res)
        
        for _ in range(max_steps):
            r, c = self.w2g(curr_x, curr_y)
            if not (0 <= r < self.h and 0 <= c < self.w) or self.obs_mask[r, c] == 1:
                break
            curr_x += dx
            curr_y += dy
            
        return [curr_x, curr_y]

    def _draw_fov(self, ax, pos, heading):
        ray_pts = [pos] 
        num_rays = 100 
        start_angle = heading - (self.fov_deg / 2.0)
        end_angle = heading + (self.fov_deg / 2.0)
        
        for angle in np.linspace(start_angle, end_angle, num_rays):
            hit_pt = self._cast_ray(pos, angle)
            ray_pts.append(hit_pt)
            
        poly = Polygon(ray_pts, closed=True, facecolor='yellow', edgecolor='orange', alpha=0.2)
        ax.add_patch(poly)

    def _evaluate_path_coverage(self, path):
        """Core logic to compute coverage for a specific directional path."""
        edge_len = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
        num_samples = max(2, int(edge_len / self.obs_interval_m))
        
        indices = np.linspace(0, len(path) - 1, num_samples).astype(int)
        sample_pts = path[indices]

        seen_indices = set()
        fov_data = []

        for i in range(len(sample_pts)-1):
            pos = sample_pts[i]
            r_idx, c_idx = self.w2g(*pos)
            if self.obs_mask[np.clip(r_idx, 0, self.h-1), np.clip(c_idx, 0, self.w-1)] == 1:
                continue 

            delta = sample_pts[i+1] - sample_pts[i]
            heading = np.degrees(np.arctan2(delta[1], delta[0]))
            fov_data.append((pos, heading)) 
            
            # 1. Spatial filtering
            near_idx = self.target_tree.query_ball_point(pos, self.max_dist_m)
            if not near_idx: continue
            
            # 2. FOV filtering
            rel_pts = self.boundary_pts[near_idx] - pos
            angles = np.degrees(np.arctan2(rel_pts[:, 1], rel_pts[:, 0]))
            diff = (angles - heading + 180) % 360 - 180
            in_fov = np.abs(diff) < (self.fov_deg / 2.0)
            
            # 3. Ray-casting wall collision check
            for orig_idx, is_in_fov in zip(near_idx, in_fov):
                if is_in_fov and orig_idx not in seen_indices:
                    if self._check_visibility(pos, self.boundary_pts[orig_idx]):
                        seen_indices.add(orig_idx)
                        
        return seen_indices, fov_data

    def _plot_debug_view(self, ax, path, seen_indices, fov_data, title_prefix):
        """Standardized drawing function for individual edges."""
        ext = [self.org[0], self.org[0]+self.w*self.res, self.org[1], self.org[1]+self.h*self.res]
        ax.imshow(self.flipped_img, cmap='gray', origin='lower', extent=ext, alpha=0.3)
        
        all_idx = set(range(len(self.boundary_pts)))
        missed_idx = list(all_idx - seen_indices)
        
        ax.scatter(self.boundary_pts[missed_idx, 0], self.boundary_pts[missed_idx, 1], c='red', s=1, label='Unseen (Red)')
        if seen_indices:
            ax.scatter(self.boundary_pts[list(seen_indices), 0], self.boundary_pts[list(seen_indices), 1], c='lime', s=1, zorder=5, label='Seen (Lime)')

        for pos, heading in fov_data:
            self._draw_fov(ax, pos, heading)
            ax.scatter(pos[0], pos[1], c='cyan', s=1, zorder=6) 

        ax.plot(path[:, 0], path[:, 1], color='blue', linewidth=4, label='Edge Path')
        ax.scatter(path[0, 0], path[0, 1], c='yellow', marker='*', s=100, edgecolors='black', zorder=10, label='Edge Start')

        ax.set_title(f"{title_prefix}\n(R={self.max_dist_m}m, int={self.obs_interval_m}m) - Seen: {len(seen_indices)} pts")
        ax.legend(loc='upper right')

    def plot_global_coverage(self):
        print(">> Computing Global Coverage (All Edges, Both Directions)...")
        global_seen = set()
        
        for u, v, data in self.G.edges(data=True):
            path = np.array(data['smooth_path']) if 'smooth_path' in data else \
                   np.array([self.G.nodes[u]['pos'], self.G.nodes[v]['pos']])
            
            if np.linalg.norm(path[0] - self.G.nodes[u]['pos']) > 0.5:
                path = path[::-1]
                
            seen_uv, _ = self._evaluate_path_coverage(path)
            global_seen.update(seen_uv)
            
            seen_vu, _ = self._evaluate_path_coverage(path[::-1])
            global_seen.update(seen_vu)
            
        fig, ax = plt.subplots(figsize=(20, 20))
        ext = [self.org[0], self.org[0]+self.w*self.res, self.org[1], self.org[1]+self.h*self.res]
        ax.imshow(self.flipped_img, cmap='gray', origin='lower', extent=ext, alpha=0.3)
        
        all_idx = set(range(len(self.boundary_pts)))
        missed_idx = list(all_idx - global_seen)
        ax.scatter(self.boundary_pts[missed_idx, 0], self.boundary_pts[missed_idx, 1], c='red', s=2, label='Unseen globally (Red)')
        if global_seen:
            ax.scatter(self.boundary_pts[list(global_seen), 0], self.boundary_pts[list(global_seen), 1], c='lime', s=2, zorder=5, label='Seen globally (Lime)')

        for u, v in self.G.edges():
            p1 = self.G.nodes[u]['pos']
            p2 = self.G.nodes[v]['pos']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=1, alpha=0.4)

        for node, data in self.G.nodes(data=True):
            pos = data['pos']
            ax.scatter(pos[0], pos[1], c='cyan', s=40, edgecolors='black', zorder=10)
            ax.text(pos[0], pos[1], str(node), color='white', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=11)
            
        ax.set_title(f"Global Combined Coverage (All Edges Forward & Backward)\nTotal Seen: {len(global_seen)} / {len(self.boundary_pts)} pts")
        ax.legend(loc='upper right')
        
        # [修改點] 儲存到 debug_dir
        save_path = os.path.join(self.debug_dir, "global_coverage.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f">> Global coverage map saved to {save_path}")

    def debug_specific_edge(self, u, v):
        if not self.G.has_edge(u, v):
            print(f">> ERROR: Edge ({u}, {v}) does not exist in the graph!")
            return

        print(f">> Generating debug image specifically for edge {u} -> {v}...")
        data = self.G[u][v]
        path = np.array(data['smooth_path']) if 'smooth_path' in data else \
               np.array([self.G.nodes[u]['pos'], self.G.nodes[v]['pos']])
        
        if np.linalg.norm(path[0] - self.G.nodes[u]['pos']) > 0.5:
            path = path[::-1]

        seen_indices, fov_data = self._evaluate_path_coverage(path)
        
        fig, ax = plt.subplots(figsize=(12, 16))
        self._plot_debug_view(ax, path, seen_indices, fov_data, f"Specific Edge Debug: {u} -> {v}")
        
        # [修改點] 儲存到 debug_dir，並把檔名加上 u, v
        save_path = os.path.join(self.debug_dir, f"debug_specific_edge_{u}_to_{v}.png")
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved specific debug image to {save_path}")

    def debug_precomputation(self, sample_size=5):
        print(f">> Starting Precomputation Debug with {sample_size} random edges...")
        
        all_edges = list(self.G.edges(data=True))
        random.shuffle(all_edges)
        test_edges = all_edges[:sample_size]
        
        for debug_id, (u, v, data) in enumerate(test_edges):
            path = np.array(data['smooth_path']) if 'smooth_path' in data else \
                   np.array([self.G.nodes[u]['pos'], self.G.nodes[v]['pos']])
            
            if np.linalg.norm(path[0] - self.G.nodes[u]['pos']) > 0.5:
                path = path[::-1]

            seen_indices, fov_data = self._evaluate_path_coverage(path)
            
            fig, ax = plt.subplots(figsize=(12, 16))
            self._plot_debug_view(ax, path, seen_indices, fov_data, f"Random Precompute Debug Edge {debug_id+1}/{sample_size} ({u}->{v})")
            
            # [修改點] 儲存到 debug_dir
            save_path = os.path.join(self.debug_dir, f"debug_edge_random_{debug_id+1}.png")
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved random debug image for edge {debug_id+1} ({u}->{v}) to {save_path}")
            
        print(">> Random Debug completed.")

if __name__ == "__main__":
    solver = CoverageSolver(
        png_path='../cropped_maps/hospital.png', 
        yaml_path='../cropped_maps/hospital.yaml', 
        roadmap_path='trajectory/hospital/planner/roadmap.pkl',
        max_dist_m=15.0, 
        fov_deg=89.0, 
        boundary_step_m=0.1, 
        obs_interval_m=0.3, 
        min_grazing_deg=5.0 
    )
    
    # -----------------------------------------------------
    # 1. 輸出包含所有 Edge 與全域 Nodes 編號的地圖
    # (此時所有 Node 一定是從 0 到 n-1)
    # -----------------------------------------------------
    solver.plot_global_coverage()
    
    # -----------------------------------------------------
    # 2. (可選) 輸出隨機幾條邊的 Debug 圖
    # -----------------------------------------------------
    solver.debug_precomputation(sample_size=5)
    
    # -----------------------------------------------------
    # 3. 輸出使用者「指定 Edge」的專屬 Debug 圖
    # -----------------------------------------------------
    try:
        nodes_list = list(solver.G.nodes())
        if len(nodes_list) >= 2:
            # 你現在可以直接用 0, 1, 2... 來指定 Node
            example_u = 16
            example_v = 41
            
            solver.debug_specific_edge(u=example_u, v=example_v)
    except Exception as e:
        print("Graph contains insufficient nodes/edges for specific debug.")