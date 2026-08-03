import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import json
import pickle, cv2, os, networkx as nx
import yaml
from scipy.spatial import KDTree
from shapely.geometry import Polygon as ShapelyPolygon
import concurrent.futures
import multiprocessing
import random

class DualCoveragePlanner:
    def __init__(self, 
                 png_path, 
                 yaml_path, 
                 roadmap_path, 
                 high_overlap_roadmap_path=None,
                 num_workers=None,
                 max_dist_m=20.0, 
                 fov_deg=89.0,
                 boundary_step_m=0.1,
                 obs_interval_m=0.3,
                 min_grazing_deg=5.0,
                 cav_lidar_range=(0.0, -11.2, -1.0, 22.4, 11.2, 3.0),
                 output_scene_name=None,
                 output_root=None,
                 max_world_y=None):
        # --- Basic Initialization ---
        self.scene_name = output_scene_name or os.path.splitext(os.path.basename(png_path))[0]
        if output_root is None:
            output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory")
        self.output_dir = os.path.join(output_root, self.scene_name, "dual")
        self.debug_dir = os.path.join(self.output_dir, "debug_shadow")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.res, self.org = cfg['resolution'], cfg['origin']
        
        raw_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        self.flipped_img = cv2.flip(raw_img, 0)
        self.h, self.w = self.flipped_img.shape[:2]
        self.obs_mask = (self.flipped_img < 200).astype(np.uint8)
        
        with open(roadmap_path, 'rb') as f:
            self.G = pickle.load(f)
        self.high_overlap_G = None
        if high_overlap_roadmap_path is not None:
            if not os.path.isfile(high_overlap_roadmap_path):
                raise FileNotFoundError(
                    f"High-overlap roadmap not found: {high_overlap_roadmap_path}"
                )
            with open(high_overlap_roadmap_path, 'rb') as f:
                self.high_overlap_G = pickle.load(f)
        self.max_world_y = max_world_y
        self._filter_graph_by_world_y()
        self._repair_graph_weights()
        
        self.max_dist_m = max_dist_m
        self.fov_deg = fov_deg
        self.boundary_step_m = boundary_step_m
        self.obs_interval_m = obs_interval_m
        self.parallel_tol_cos = -np.sin(np.radians(min_grazing_deg))
        self.cav_lidar_range = cav_lidar_range
        self.cav_forward_min = float(cav_lidar_range[0])
        self.cav_lateral_min = float(cav_lidar_range[1])
        self.cav_forward_max = float(cav_lidar_range[3])
        self.cav_lateral_max = float(cav_lidar_range[4])
        self.cav_query_radius = float(np.hypot(
            max(abs(self.cav_forward_min), abs(self.cav_forward_max)),
            max(abs(self.cav_lateral_min), abs(self.cav_lateral_max)),
        ))
        self.boundary_pts = self.extract_boundary_points(step_m=self.boundary_step_m)
        self.boundary_pts = self._filter_boundary_points_by_world_y(self.boundary_pts)
        if len(self.boundary_pts) == 0:
            raise ValueError("No boundary points remain after applying max_world_y filter.")
        self.target_tree = KDTree(self.boundary_pts)
        
        self.dir_edge_states = {} 
        cpu_count = multiprocessing.cpu_count()
        self.num_workers = min(
            cpu_count, num_workers if num_workers is not None else 2
        )

    # ================= 1. Graph & Environment Setup =================

    def _filter_graph_by_world_y(self):
        if self.max_world_y is None:
            return

        edges_to_remove = []
        for u, v, data in self.G.edges(data=True):
            if 'smooth_path' in data:
                pts = np.array(data['smooth_path'], dtype=float)
            else:
                pts = np.array([self.G.nodes[u]['pos'], self.G.nodes[v]['pos']], dtype=float)
            if len(pts) == 0 or np.any(pts[:, 1] > self.max_world_y):
                edges_to_remove.append((u, v))

        self.G.remove_edges_from(edges_to_remove)
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)
        print(
            f">> Applied max_world_y={self.max_world_y:.2f}m filter: "
            f"removed {len(edges_to_remove)} edges and {len(isolated_nodes)} isolated nodes."
        )

    def _filter_boundary_points_by_world_y(self, pts):
        if self.max_world_y is None:
            return pts
        filtered = pts[pts[:, 1] <= self.max_world_y]
        print(
            f">> Applied boundary max_world_y={self.max_world_y:.2f}m filter: "
            f"kept {len(filtered)}/{len(pts)} target points."
        )
        return filtered

    def _repair_graph_weights(self):
        for u, v in self.G.edges():
            if 'weight' not in self.G[u][v]:
                p1, p2 = np.array(self.G.nodes[u]['pos']), np.array(self.G.nodes[v]['pos'])
                self.G[u][v]['weight'] = np.linalg.norm(p1 - p2)

    def g2w(self, r, c): return [self.org[0] + c * self.res, self.org[1] + r * self.res]
    def w2g(self, x, y): return int((y - self.org[1]) / self.res), int((x - self.org[0]) / self.res)

    def extract_boundary_points(self, step_m=0.1):
        padded_mask = np.pad(self.obs_mask, pad_width=1, mode='constant', constant_values=1)
        contours, _ = cv2.findContours(padded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        all_pts = []
        step_px = max(1, int(step_m / self.res))
        for cnt in contours:
            for i in range(0, len(cnt), step_px):
                c, r = cnt[i][0][0] - 1, cnt[i][0][1] - 1
                if 0 <= c < self.w and 0 <= r < self.h:
                    all_pts.append(self.g2w(r, c))
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
        
    def _get_fov_polygon(self, pos, heading):
        num_rays = 90
        start_angle = heading - (self.fov_deg / 2.0)
        end_angle = heading + (self.fov_deg / 2.0)
        pts = [pos]
        for angle in np.linspace(start_angle, end_angle, num_rays):
            rad = np.radians(angle)
            dx = np.cos(rad) * self.res * 0.5
            dy = np.sin(rad) * self.res * 0.5
            cx, cy = pos[0], pos[1]
            for _ in range(int(self.max_dist_m / (self.res * 0.5))):
                r, c = self.w2g(cx, cy)
                if not (0 <= r < self.h and 0 <= c < self.w) or self.obs_mask[r, c] == 1:
                    break
                cx += dx
                cy += dy
            pts.append([cx, cy])
        return pts

    def _get_cav_range_indices(self, pos, heading):
        near_idx = self.target_tree.query_ball_point(pos, self.cav_query_radius)
        if not near_idx:
            return []

        rel_pts = self.boundary_pts[near_idx] - pos
        theta = np.radians(heading)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        forward = rel_pts[:, 0] * cos_t + rel_pts[:, 1] * sin_t
        lateral = -rel_pts[:, 0] * sin_t + rel_pts[:, 1] * cos_t
        in_range = (
            (forward >= self.cav_forward_min) &
            (forward <= self.cav_forward_max) &
            (lateral >= self.cav_lateral_min) &
            (lateral <= self.cav_lateral_max)
        )
        return [near_idx[j] for j, is_in in enumerate(in_range) if is_in]

    def _make_valid_polygon(self, pts):
        if len(pts) < 3:
            return ShapelyPolygon()
        poly = ShapelyPolygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    def _get_cav_range_polygon(self, pos, heading):
        theta = np.radians(heading)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        local_corners = np.array([
            [self.cav_forward_min, self.cav_lateral_min],
            [self.cav_forward_max, self.cav_lateral_min],
            [self.cav_forward_max, self.cav_lateral_max],
            [self.cav_forward_min, self.cav_lateral_max],
        ], dtype=float)
        world = np.column_stack([
            pos[0] + local_corners[:, 0] * cos_t - local_corners[:, 1] * sin_t,
            pos[1] + local_corners[:, 0] * sin_t + local_corners[:, 1] * cos_t,
        ])
        return self._make_valid_polygon(world)

    def _ray_distance_to_cav_rect(self, local_angle_rad):
        dx, dy = np.cos(local_angle_rad), np.sin(local_angle_rad)
        if dx <= 1e-6:
            return None

        candidates = [self.cav_forward_max / dx]
        if dy > 1e-6:
            candidates.append(self.cav_lateral_max / dy)
        elif dy < -1e-6:
            candidates.append(self.cav_lateral_min / dy)

        candidates = [dist for dist in candidates if dist > 0]
        if not candidates:
            return None
        return min(candidates)

    def _get_cav_visible_polygon(self, pos, heading):
        endpoints = []
        world_heading = np.radians(heading)
        ray_step = max(self.res * 0.5, 0.05)
        for local_angle in np.linspace(-89.5, 89.5, 360):
            local_rad = np.radians(local_angle)
            max_dist = self._ray_distance_to_cav_rect(local_rad)
            if max_dist is None:
                continue

            ray_angle = world_heading + local_rad
            dx, dy = np.cos(ray_angle), np.sin(ray_angle)
            last_free = np.array(pos, dtype=float)
            for dist in np.arange(0.0, max_dist + ray_step, ray_step):
                pt = np.array([pos[0] + dx * dist, pos[1] + dy * dist], dtype=float)
                r, c = self.w2g(pt[0], pt[1])
                if not (0 <= r < self.h and 0 <= c < self.w):
                    break
                if self.obs_mask[r, c] == 1:
                    break
                last_free = pt
            endpoints.append(last_free)

        if len(endpoints) < 3:
            return ShapelyPolygon()
        return self._make_valid_polygon(np.vstack([np.array(pos, dtype=float), np.array(endpoints)]))

    # ================= 2. Synergy Precomputation =================

    def _calc_edge_states(self, start_n, end_n, data):
        states = []
        num_frames = 10 
        path = np.array(data['smooth_path']) if data and 'smooth_path' in data else np.array([self.G.nodes[start_n]['pos'], self.G.nodes[end_n]['pos']])
        if np.linalg.norm(path[0] - self.G.nodes[start_n]['pos']) > 0.5: path = path[::-1]

        indices = np.linspace(0, len(path) - 1, num_frames).astype(int)
        for i in range(len(indices)-1):
            pos = path[indices[i]]
            r_idx, c_idx = self.w2g(*pos)
            if self.obs_mask[np.clip(r_idx, 0, self.h-1), np.clip(c_idx, 0, self.w-1)] == 1: continue 

            delta = path[indices[i+1]] - pos
            heading = np.degrees(np.arctan2(delta[1], delta[0]))
            
            near_idx = self.target_tree.query_ball_point(pos, self.max_dist_m)
            if near_idx:
                rel_pts = self.boundary_pts[near_idx] - pos
                angles = np.degrees(np.arctan2(rel_pts[:, 1], rel_pts[:, 0]))
                diff = (angles - heading + 180) % 360 - 180
                in_fov = np.abs(diff) < (self.fov_deg / 2.0)
                fov_idx = [near_idx[j] for j, is_in in enumerate(in_fov) if is_in]
            else:
                fov_idx = []

            seen_idx = {idx for idx in fov_idx if self._check_visibility(pos, self.boundary_pts[idx])}
            shadow_idx = set(fov_idx) - seen_idx

            cav_idx = set(self._get_cav_range_indices(pos, heading))
            cav_seen_idx = {idx for idx in cav_idx if self._check_visibility(pos, self.boundary_pts[idx])}
            no_info_idx = cav_idx - cav_seen_idx

            range_poly = self._get_cav_range_polygon(pos, heading)
            visible_poly = self._get_cav_visible_polygon(pos, heading)
            if not visible_poly.is_empty:
                visible_poly = visible_poly.intersection(range_poly)
            no_info_poly = range_poly.difference(visible_poly) if not range_poly.is_empty else ShapelyPolygon()
            fov_poly = self._make_valid_polygon(
                self._get_fov_polygon(pos, heading)
            )

            states.append({
                'pos': pos,
                'heading': heading,
                'V': seen_idx,
                'Shadow': shadow_idx,
                'CavRange': cav_idx,
                'CavV': cav_seen_idx,
                'NoInfo': no_info_idx,
                'RangePoly': range_poly,
                'VisiblePoly': visible_poly,
                'NoInfoPoly': no_info_poly,
                'FovPoly': fov_poly,
                'RangeArea': float(range_poly.area),
                'VisibleArea': float(visible_poly.area),
                'NoInfoArea': float(no_info_poly.area),
                'FovArea': float(fov_poly.area),
            })
        return states

    def _parallel_calc_edge(self, edge_info):
        u, v, data = edge_info
        return (u, v, self._calc_edge_states(u, v, data), self._calc_edge_states(v, u, data))

    def precompute_directed_edges(self):
        print(f">> [Parallel] Precomputing Synchronized States using {self.num_workers} cores...")
        self.dir_edge_states.clear()
        edges_list = list(self.G.edges(data=True))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._parallel_calc_edge, edges_list, chunksize=10))
            
        for u, v, states_uv, states_vu in results:
            self.dir_edge_states[(u, v)] = states_uv
            self.dir_edge_states[(v, u)] = states_vu
        print(">> Precomputation completed.")

    def _parallel_eval_pair(self, pair_info):
        e1, e2, min_overlap, min_score, min_valid_fraction, max_robot_dist_m = pair_info
        states1 = self.dir_edge_states[e1]
        states2 = self.dir_edge_states[e2]
        if not states1 or not states2: return None

        def shadow_frame_details(s1, s2):
            r1_area = s1.get('RangeArea', 0.0)
            r2_area = s2.get('RangeArea', 0.0)
            if r1_area <= 0.0 or r2_area <= 0.0:
                return None

            v1_area = s1.get('VisibleArea', 0.0)
            v2_area = s2.get('VisibleArea', 0.0)
            n1_area = s1.get('NoInfoArea', 0.0)
            n2_area = s2.get('NoInfoArea', 0.0)
            v1_poly = s1.get('VisiblePoly', ShapelyPolygon())
            v2_poly = s2.get('VisiblePoly', ShapelyPolygon())
            n1_poly = s1.get('NoInfoPoly', ShapelyPolygon())
            n2_poly = s2.get('NoInfoPoly', ShapelyPolygon())

            e1_effective_ratio = v1_area / r1_area
            e2_effective_ratio = v2_area / r2_area
            e1_no_info_ratio = n1_area / r1_area
            e2_no_info_ratio = n2_area / r2_area

            comp1 = float(n2_poly.intersection(v1_poly).area)  # e1 fills e2's no-info area.
            comp2 = float(n1_poly.intersection(v2_poly).area)  # e2 fills e1's no-info area.
            overlap = float(v1_poly.intersection(v2_poly).area)

            if e1_effective_ratio <= e2_effective_ratio:
                ego_source = 'e1'
                collaborative_source = 'e2'
                ego_effective_ratio = e1_effective_ratio
                collaborative_effective_ratio = e2_effective_ratio
                ego_no_info_ratio = e1_no_info_ratio
                ego_no_info_area = n1_area
                collaborative_fill_area = comp2
            else:
                ego_source = 'e2'
                collaborative_source = 'e1'
                ego_effective_ratio = e2_effective_ratio
                collaborative_effective_ratio = e1_effective_ratio
                ego_no_info_ratio = e2_no_info_ratio
                ego_no_info_area = n2_area
                collaborative_fill_area = comp1

            fill_ratio = collaborative_fill_area / max(1e-6, ego_no_info_area)
            score = (
                collaborative_fill_area
                + (fill_ratio * 10.0)
                + (ego_no_info_ratio * 5.0)
                + (overlap * 0.05)
            )
            return {
                'score': score,
                'overlap': overlap,
                'comp1': comp1,
                'comp2': comp2,
                'ego_source': ego_source,
                'collaborative_source': collaborative_source,
                'ego_effective_ratio': ego_effective_ratio,
                'collaborative_effective_ratio': collaborative_effective_ratio,
                'ego_no_info_ratio': ego_no_info_ratio,
                'ego_no_info_area_m2': ego_no_info_area,
                'collaborative_fill_area_m2': collaborative_fill_area,
                'fill_ratio': fill_ratio,
            }
        
        frame_details = []
        valid_frame_indices = []
        for idx, (s1, s2) in enumerate(zip(states1, states2)):
            if np.linalg.norm(np.array(s1['pos']) - np.array(s2['pos'])) > max_robot_dist_m:
                return None

            details = shadow_frame_details(s1, s2)
            if details is None or details['overlap'] < min_overlap:
                frame_details.append(None)
                continue

            frame_details.append(details)
            if details['collaborative_fill_area_m2'] >= min_score:
                valid_frame_indices.append(idx)

        total_frames = len(frame_details)
        if total_frames == 0:
            return None
        valid_fraction = len(valid_frame_indices) / float(total_frames)
        if valid_fraction < min_valid_fraction:
            return None

        best_idx = max(valid_frame_indices, key=lambda idx: frame_details[idx]['score'])
        best_details = frame_details[best_idx]
        
        return {
            'e1': e1, 'e2': e2, 'score': best_details['score'], 'best_idx': best_idx,
            'overlap': best_details['overlap'],
            'comp1': best_details['comp1'],
            'comp2': best_details['comp2'],
            'ego_source': best_details['ego_source'],
            'collaborative_source': best_details['collaborative_source'],
            'ego_effective_ratio': best_details['ego_effective_ratio'],
            'collaborative_effective_ratio': best_details['collaborative_effective_ratio'],
            'ego_no_info_ratio': best_details['ego_no_info_ratio'],
            'ego_no_info_area_m2': best_details['ego_no_info_area_m2'],
            'collaborative_fill_area_m2': best_details['collaborative_fill_area_m2'],
            'fill_ratio': best_details['fill_ratio'],
            'valid_fraction': valid_fraction, 'valid_frames': len(valid_frame_indices),
            'total_frames': total_frames
        }

    def get_all_useful_pairs(self, min_overlap=15, min_score=5.0, min_valid_fraction=0.25, max_robot_dist_m=20.0):
        print(f">> Extracting ALL valuable synchronization pairs...")
        edges = list(self.dir_edge_states.keys())
        pairs_to_eval = []
        for i in range(len(edges)):
            e1 = edges[i]
            for e2 in edges[i+1:]:
                # Evaluate the pair only when the two edges share no nodes.
                if set(e1).isdisjoint(set(e2)): 
                    pairs_to_eval.append((e1, e2, min_overlap, min_score, min_valid_fraction, max_robot_dist_m))
        
        reward_map = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._parallel_eval_pair, pairs_to_eval, chunksize=100))
            
        for res in results:
            if res is not None:
                reward_map[(res['e1'], res['e2'])] = res
                reward_map[(res['e2'], res['e1'])] = res
                
        print(f">> Found {len(reward_map)//2} highly valuable synergy combinations with >= {min_valid_fraction:.0%} valid frames and robot distance <= {max_robot_dist_m:.1f}m.")
        return reward_map

    def extract_filtered_pairs(self, reward_map, min_new_info=None):
        """Return unique shadow pairs sorted by score. New-info filtering is disabled."""
        unique_pairs = {}
        for (e1, e2), data in reward_map.items():
            key = frozenset([e1, e2])
            if key not in unique_pairs:
                unique_pairs[key] = data

        selected_pairs = sorted(unique_pairs.values(), key=lambda x: x['score'], reverse=True)
        print(f">> New-info filtering disabled. Keeping {len(selected_pairs)} unique shadow pairs.")
        return selected_pairs

    def _parallel_eval_random_pair(self, pair_info):
        (
            e1,
            e2,
            min_bev_fov_iou,
            min_overlap_area_m2,
            min_valid_fraction,
            min_robot_dist_m,
            max_robot_dist_m,
        ) = pair_info
        states1 = self.dir_edge_states.get(e1)
        states2 = self.dir_edge_states.get(e2)
        if not states1 or not states2:
            return None

        frame_details = []
        valid_indices = []
        for idx, (s1, s2) in enumerate(zip(states1, states2)):
            robot_dist = float(np.linalg.norm(
                np.asarray(s1['pos']) - np.asarray(s2['pos'])
            ))
            if (
                robot_dist < min_robot_dist_m or
                robot_dist > max_robot_dist_m
            ):
                frame_details.append(None)
                continue

            fov1 = s1.get('FovPoly', ShapelyPolygon())
            fov2 = s2.get('FovPoly', ShapelyPolygon())
            if fov1.is_empty or fov2.is_empty:
                frame_details.append(None)
                continue

            intersection_area = float(fov1.intersection(fov2).area)
            union_area = float(fov1.union(fov2).area)
            fov_iou = intersection_area / max(union_area, 1e-6)
            details = {
                'bev_fov_iou': fov_iou,
                'bev_fov_overlap_area_m2': intersection_area,
                'bev_fov_union_area_m2': union_area,
                'robot_distance_m': robot_dist,
            }
            frame_details.append(details)
            if (
                fov_iou >= min_bev_fov_iou and
                intersection_area >= min_overlap_area_m2
            ):
                valid_indices.append(idx)

        total_frames = len(frame_details)
        if total_frames == 0:
            return None
        valid_fraction = len(valid_indices) / float(total_frames)
        if valid_fraction < min_valid_fraction:
            return None

        best_idx = max(
            valid_indices,
            key=lambda idx: (
                frame_details[idx]['bev_fov_iou'],
                frame_details[idx]['bev_fov_overlap_area_m2'],
            ),
        )
        best = frame_details[best_idx]
        return {
            'e1': e1,
            'e2': e2,
            'score': best['bev_fov_iou'],
            'best_idx': best_idx,
            'bev_fov_iou': best['bev_fov_iou'],
            'bev_fov_overlap_area_m2': best['bev_fov_overlap_area_m2'],
            'bev_fov_union_area_m2': best['bev_fov_union_area_m2'],
            'robot_distance_m': best['robot_distance_m'],
            'valid_fraction': valid_fraction,
            'valid_frames': len(valid_indices),
            'total_frames': total_frames,
            'generation_mode': 'random',
        }

    def get_random_overlap_pairs(
            self,
            available_edges,
            min_bev_fov_iou=0.15,
            min_overlap_area_m2=10.0,
            min_valid_fraction=0.25,
            min_robot_dist_m=1.0,
            max_robot_dist_m=20.0):
        """Find geometry-only random candidates with enough BEV FOV overlap."""
        edges = list(available_edges)
        pairs_to_eval = []
        for i, e1 in enumerate(edges):
            for e2 in edges[i + 1:]:
                if set(e1).isdisjoint(set(e2)):
                    pairs_to_eval.append((
                        e1,
                        e2,
                        min_bev_fov_iou,
                        min_overlap_area_m2,
                        min_valid_fraction,
                        min_robot_dist_m,
                        max_robot_dist_m,
                    ))

        print(
            f">> Evaluating {len(pairs_to_eval)} remaining-path random pairs "
            f"(FOV IoU >= {min_bev_fov_iou:.2f}, "
            f"overlap >= {min_overlap_area_m2:.1f}m2)..."
        )
        reward_map = {}
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers) as executor:
            results = list(executor.map(
                self._parallel_eval_random_pair,
                pairs_to_eval,
                chunksize=100,
            ))

        for result in results:
            if result is None:
                continue
            reward_map[(result['e1'], result['e2'])] = result
            reward_map[(result['e2'], result['e1'])] = result

        print(
            f">> Found {len(reward_map) // 2} random geometry candidates "
            f"from {len(edges)} remaining directed paths."
        )
        return reward_map

    def extract_random_path_disjoint_split_pairs(
            self,
            reward_map,
            reserved_paths,
            ratios=None,
            directed=True,
            random_seed=42):
        """Randomly select remaining-path pairs without cross-split leakage."""
        if ratios is None:
            ratios = {'train': 0.75, 'validate': 0.10, 'test': 0.15}

        unique_pairs = {}
        for (e1, e2), data in reward_map.items():
            key = frozenset([e1, e2])
            if key not in unique_pairs:
                unique_pairs[key] = data

        candidates = list(unique_pairs.values())
        candidates.sort(key=lambda item: repr((item['e1'], item['e2'])))
        rng = random.Random(random_seed)
        rng.shuffle(candidates)

        split_pairs = {name: [] for name in ratios}
        used_paths = set(reserved_paths)
        for pair_data in candidates:
            pair_paths = self._pair_path_keys(
                pair_data, directed=directed
            )
            if pair_paths & used_paths:
                continue

            split_name = min(
                ratios,
                key=lambda name: (
                    len(split_pairs[name]) / float(ratios[name]),
                    len(split_pairs[name]),
                ),
            )
            split_pairs[split_name].append(pair_data)
            used_paths |= pair_paths

        print(
            ">> Random remaining-path split: "
            + ", ".join(
                f"{name}={len(split_pairs[name])}" for name in ratios
            )
            + f" | selected={sum(len(v) for v in split_pairs.values())}, "
            + f"reserved paths={len(reserved_paths)}, seed={random_seed}"
        )
        return split_pairs


    def get_random_pairs_for_primary_egos(
            self,
            primary_split_pairs,
            min_bev_fov_iou=0.05,
            min_overlap_area_m2=5.0,
            min_valid_fraction=0.25,
            min_robot_dist_m=1.0,
            max_robot_dist_m=20.0):
        """Evaluate same-split random partners for every fixed primary ego."""
        split_names = ('train', 'validate', 'test')
        split_paths = {name: set() for name in split_names}
        primary_combinations = set()
        ego_records = []

        for source_case_type, source_splits in primary_split_pairs.items():
            for split_name in split_names:
                for case_index, source_pair in enumerate(
                        source_splits.get(split_name, [])):
                    source_e1 = source_pair['e1']
                    source_e2 = source_pair['e2']
                    split_paths[split_name].update((source_e1, source_e2))
                    primary_combinations.add(
                        frozenset((source_e1, source_e2))
                    )
                    ego_edge, original_partner, _ = self._ordered_role_edges(
                        source_pair, source_case_type
                    )
                    ego_records.append({
                        'source_case_id': (
                            f'{source_case_type}:{split_name}:{case_index}'
                        ),
                        'source_case_type': source_case_type,
                        'source_split': split_name,
                        'source_e1': source_e1,
                        'source_e2': source_e2,
                        'source_ego_edge': ego_edge,
                        'source_original_partner': original_partner,
                    })

        task_records = []
        tasks_per_split = {name: 0 for name in split_names}
        for ego_record in ego_records:
            split_name = ego_record['source_split']
            ego_edge = ego_record['source_ego_edge']
            for partner_edge in sorted(split_paths[split_name], key=repr):
                if partner_edge == ego_edge:
                    continue
                if frozenset((ego_edge, partner_edge)) in primary_combinations:
                    continue
                task_records.append({
                    'pair_info': (
                        ego_edge, partner_edge, min_bev_fov_iou,
                        min_overlap_area_m2, min_valid_fraction,
                        min_robot_dist_m, max_robot_dist_m,
                    ),
                    'metadata': ego_record,
                })
                tasks_per_split[split_name] += 1

        print(
            f">> Evaluating {len(task_records)} fixed-ego same-split "
            "random candidates ("
            f"FOV IoU >= {min_bev_fov_iou:.2f}, "
            f"overlap >= {min_overlap_area_m2:.1f}m2, "
            f"valid frames >= {min_valid_fraction:.0%})..."
        )
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers) as executor:
            results = list(executor.map(
                self._parallel_eval_random_pair,
                [record['pair_info'] for record in task_records],
                chunksize=100,
            ))

        candidates = []
        valid_per_split = {name: 0 for name in split_names}
        for record, result in zip(task_records, results):
            if result is None:
                continue
            result.update(record['metadata'])
            candidates.append(result)
            valid_per_split[result['source_split']] += 1

        print(
            ">> Valid fixed-ego random candidates: "
            + ", ".join(
                f"{name}={valid_per_split[name]}/{tasks_per_split[name]}"
                for name in split_names
            )
        )
        return candidates

    def select_fixed_ego_random_split_pairs(
            self, candidates, max_partner_usage=3, random_seed=42):
        """Select at most one random partner per primary ego case."""
        split_names = ('train', 'validate', 'test')
        split_pairs = {name: [] for name in split_names}
        candidates_by_ego = {}
        for candidate in candidates:
            candidates_by_ego.setdefault(
                candidate['source_case_id'], []
            ).append(candidate)

        rng = random.Random(random_seed)
        ego_ids = sorted(candidates_by_ego)
        rng.shuffle(ego_ids)
        partner_usage = {}
        selected_pairs = set()

        for ego_id in ego_ids:
            eligible = []
            for candidate in candidates_by_ego[ego_id]:
                partner_key = self._path_key(candidate['e2'], directed=True)
                pair_key = (candidate['e1'], candidate['e2'])
                if partner_usage.get(partner_key, 0) >= max_partner_usage:
                    continue
                if pair_key in selected_pairs:
                    continue
                eligible.append(candidate)

            if not eligible:
                continue
            minimum_usage = min(
                partner_usage.get(
                    self._path_key(item['e2'], directed=True), 0
                )
                for item in eligible
            )
            least_used = [
                item for item in eligible
                if partner_usage.get(
                    self._path_key(item['e2'], directed=True), 0
                ) == minimum_usage
            ]
            selected = rng.choice(least_used).copy()
            partner_key = self._path_key(selected['e2'], directed=True)
            selected['partner_usage_before_selection'] = minimum_usage
            selected['max_partner_usage'] = max_partner_usage
            partner_usage[partner_key] = minimum_usage + 1
            selected_pairs.add((selected['e1'], selected['e2']))
            split_pairs[selected['source_split']].append(selected)

        print(
            ">> Fixed-ego random split: "
            + ", ".join(
                f"{name}={len(split_pairs[name])}" for name in split_names
            )
            + f" | selected={sum(len(v) for v in split_pairs.values())}/"
            + f"{len(candidates_by_ego)} eligible ego cases, "
            + f"partner max usage={max(partner_usage.values(), default=0)}/"
            + f"{max_partner_usage}, seed={random_seed}"
        )
        return split_pairs


    def _path_key(self, edge, directed=True):
        """Return a split identity for a directed or undirected robot path."""
        edge_key = tuple(tuple(node) for node in edge)
        if directed:
            return edge_key
        return min(edge_key, tuple(reversed(edge_key)))

    def _pair_path_keys(self, pair_data, directed=True):
        return {
            self._path_key(pair_data['e1'], directed=directed),
            self._path_key(pair_data['e2'], directed=directed),
        }

    def _split_targets(self, total, ratios):
        ratio_sum = sum(ratios.values())
        if ratio_sum <= 0:
            raise ValueError("Split ratios must be positive.")
        normalized = {name: value / ratio_sum for name, value in ratios.items()}
        return {name: max(1, int(round(total * normalized[name]))) for name in ratios}

    def extract_path_disjoint_split_pairs(self,
                                          reward_map,
                                          min_new_info=15,
                                          ratios=None,
                                          directed=True,
                                          target_total=None,
                                          candidate_pairs=None):
        """
        Select pair cases and assign them to train/validate/test while preventing
        any directed path from appearing in more than one split.

        Bridge pairs that would connect two already-owned split path pools are
        skipped. This intentionally trades a few high-score cases for cleaner
        train/validation/test separation.
        """
        if ratios is None:
            ratios = {'train': 0.75, 'validate': 0.10, 'test': 0.15}

        baseline_pairs = candidate_pairs
        if baseline_pairs is None:
            baseline_pairs = self.extract_filtered_pairs(reward_map, min_new_info=min_new_info)
        if target_total is None:
            target_total = len(baseline_pairs)
        targets = self._split_targets(target_total, ratios)

        candidates = list(baseline_pairs)
        path_freq = {}
        for pair_data in candidates:
            for path in self._pair_path_keys(pair_data, directed=directed):
                path_freq[path] = path_freq.get(path, 0) + 1

        candidates.sort(
            key=lambda pair_data: (
                -pair_data.get('score', 0),
                -max(path_freq[path] for path in self._pair_path_keys(pair_data, directed=directed)),
            )
        )

        split_pairs = {name: [] for name in ratios}
        split_seen = {name: set() for name in ratios}
        path_owner = {}
        skipped_bridges = 0
        skipped_low_info = 0

        for pair_data in candidates:
            pair_paths = self._pair_path_keys(pair_data, directed=directed)
            owned_splits = {path_owner[path] for path in pair_paths if path in path_owner}
            if len(owned_splits) > 1:
                skipped_bridges += 1
                continue

            if owned_splits:
                split_name = next(iter(owned_splits))
                if len(split_pairs[split_name]) >= targets[split_name]:
                    continue
            else:
                split_name = min(
                    ratios,
                    key=lambda name: (
                        len(split_pairs[name]) / float(targets[name]),
                        len(split_pairs[name]),
                    )
                )

            s1 = self.dir_edge_states[pair_data['e1']][pair_data['best_idx']]
            s2 = self.dir_edge_states[pair_data['e2']][pair_data['best_idx']]
            visible_union = s1['V'] | s2['V']
            new_info = len(visible_union - split_seen[split_name])
            if new_info <= min_new_info:
                skipped_low_info += 1
                continue

            split_pairs[split_name].append(pair_data)
            split_seen[split_name] |= visible_union
            for path in pair_paths:
                path_owner[path] = split_name

            if all(len(split_pairs[name]) >= targets[name] for name in ratios):
                break

        print(
            ">> Path-disjoint split selection: "
            + ", ".join(f"{name}={len(split_pairs[name])}/{targets[name]}" for name in ratios)
            + f" | skipped bridge={skipped_bridges}, low-info={skipped_low_info}, "
            + f"direction={'directed' if directed else 'undirected'}"
        )
        return split_pairs

    def extract_joint_path_disjoint_split_pairs(self,
                                                case_groups,
                                                min_new_info_by_group=None,
                                                ratios=None,
                                                directed=True,
                                                random_seed=42,
                                                max_path_usage=1):
        """
        Assign multiple case groups, such as shadow and distance, with a shared
        path owner table. This keeps train/validate/test path-disjoint after
        groups are merged for training.

        Ratios are soft targets. Cases are assigned greedily with a shared path
        owner table; bridge cases that would connect two already-separated
        splits are skipped. This intentionally drops a small number of cases so
        train/validate/test can all exist without path leakage.
        """
        if ratios is None:
            ratios = {'train': 0.75, 'validate': 0.10, 'test': 0.15}
        if min_new_info_by_group is None:
            min_new_info_by_group = {}

        all_candidates = []
        for group_name, pairs in case_groups.items():
            for pair_data in pairs:
                all_candidates.append((len(all_candidates), group_name, pair_data))

        targets = self._split_targets(len(all_candidates), ratios)
        path_freq = {}
        for _, _, pair_data in all_candidates:
            for path in self._pair_path_keys(pair_data, directed=directed):
                path_freq[path] = path_freq.get(path, 0) + 1

        def pair_spatial_key(pair_data, bin_size_m=5.0):
            s1 = self.dir_edge_states[pair_data['e1']][pair_data['best_idx']]
            s2 = self.dir_edge_states[pair_data['e2']][pair_data['best_idx']]
            center = (np.array(s1['pos']) + np.array(s2['pos'])) / 2.0
            return (int(np.floor(center[0] / bin_size_m)), int(np.floor(center[1] / bin_size_m)))

        for _, _, pair_data in all_candidates:
            pair_data['spatial_key'] = pair_spatial_key(pair_data)

        rng = random.Random(random_seed)
        rng.shuffle(all_candidates)
        all_candidates.sort(
            key=lambda item: (
                -item[2].get('score', 0),
                item[1],
                item[0],
            )
        )

        split_pairs = {
            group_name: {split_name: [] for split_name in ratios}
            for group_name in case_groups
        }
        split_counts = {split_name: 0 for split_name in ratios}
        split_seen = {split_name: set() for split_name in ratios}
        split_spatial_counts = {split_name: {} for split_name in ratios}
        path_owner = {}
        path_usage = {}
        added_candidate_ids = set()
        skipped_bridges = {1: 0, 2: 0}
        skipped_full = {1: 0, 2: 0}
        skipped_usage = {1: 0, 2: 0}
        skipped_low_info = 0
        added_by_pass = {1: 0, 2: 0}

        def split_has_capacity(split_name):
            return split_counts[split_name] < targets[split_name]

        def choose_split(pair_paths, spatial_key):
            owned_splits = {path_owner[path] for path in pair_paths if path in path_owner}
            if len(owned_splits) > 1:
                return None
            if owned_splits:
                split_name = next(iter(owned_splits))
                if split_has_capacity(split_name):
                    return split_name
                return None

            available_splits = [name for name in ratios if split_has_capacity(name)]
            if not available_splits:
                return None

            return min(
                available_splits,
                key=lambda name: (
                    split_spatial_counts[name].get(spatial_key, 0),
                    split_counts[name] / float(targets[name]),
                    split_counts[name],
                )
            )

        def add_candidate(candidate_id, group_name, pair_data, split_name):
            split_pairs[group_name][split_name].append(pair_data)
            split_counts[split_name] += 1
            added_candidate_ids.add(candidate_id)

            s1 = self.dir_edge_states[pair_data['e1']][pair_data['best_idx']]
            s2 = self.dir_edge_states[pair_data['e2']][pair_data['best_idx']]
            split_seen[split_name] |= (s1['V'] | s2['V'])
            spatial_key = pair_data.get('spatial_key')
            split_spatial_counts[split_name][spatial_key] = split_spatial_counts[split_name].get(spatial_key, 0) + 1

            for path in self._pair_path_keys(pair_data, directed=directed):
                path_owner[path] = split_name
                path_usage[path] = path_usage.get(path, 0) + 1

        def split_distance():
            return (
                sum(abs(split_counts[name] - targets[name]) for name in ratios),
                max(split_counts.values()) - min(split_counts.values()),
            )

        def try_rebalance_new_case(candidate_id, group_name, pair_data):
            pair_paths = self._pair_path_keys(pair_data, directed=directed)
            owned_splits = {path_owner[path] for path in pair_paths if path in path_owner}
            if owned_splits:
                return None

            before = split_distance()
            best_split = min(
                ratios,
                key=lambda name: (
                    abs((split_counts[name] + 1) - targets[name]),
                    split_counts[name],
                )
            )
            split_counts[best_split] += 1
            after = split_distance()
            split_counts[best_split] -= 1
            if after <= before:
                return best_split
            return None

        for candidate_id, group_name, pair_data in all_candidates:
            pair_paths = self._pair_path_keys(pair_data, directed=directed)
            if any(path_usage.get(path, 0) >= max_path_usage for path in pair_paths):
                skipped_usage[1] += 1
                continue

            split_name = choose_split(pair_paths, pair_data['spatial_key'])
            if split_name is None:
                owned_splits = {path_owner[path] for path in pair_paths if path in path_owner}
                if len(owned_splits) > 1:
                    skipped_bridges[1] += 1
                else:
                    skipped_full[1] += 1
                continue

            add_candidate(candidate_id, group_name, pair_data, split_name)
            added_by_pass[1] += 1

        for candidate_id, group_name, pair_data in all_candidates:
            if candidate_id in added_candidate_ids:
                continue

            pair_paths = self._pair_path_keys(pair_data, directed=directed)
            if any(path_usage.get(path, 0) >= max_path_usage for path in pair_paths):
                skipped_usage[2] += 1
                continue

            split_name = choose_split(pair_paths, pair_data['spatial_key'])
            if split_name is None:
                owned_splits = {path_owner[path] for path in pair_paths if path in path_owner}
                if len(owned_splits) > 1:
                    skipped_bridges[2] += 1
                else:
                    skipped_full[2] += 1
                continue

            rebalance_split = try_rebalance_new_case(candidate_id, group_name, pair_data)
            if rebalance_split is not None:
                split_name = rebalance_split

            add_candidate(candidate_id, group_name, pair_data, split_name)
            added_by_pass[2] += 1

        print(
            ">> Joint path-disjoint split selection: "
            + ", ".join(f"{name}={split_counts[name]}/{targets[name]}" for name in ratios)
            + f" | selected={sum(split_counts.values())}/{len(all_candidates)}, "
            + f"added pass1={added_by_pass[1]}, pass2={added_by_pass[2]}, "
            + f"skipped bridge pass1={skipped_bridges[1]}, pass2={skipped_bridges[2]}, "
            + f"skipped full pass1={skipped_full[1]}, pass2={skipped_full[2]}, "
            + f"skipped usage pass1={skipped_usage[1]}, pass2={skipped_usage[2]}, "
            + f"max path usage={max_path_usage}, "
            + f"new-info filtering=disabled, candidate order=score-desc(seeded tie-break seed={random_seed}), "
            + f"direction={'directed' if directed else 'undirected'}"
        )
        for group_name in case_groups:
            print(
                f">>   {group_name}: "
                + ", ".join(f"{name}={len(split_pairs[group_name][name])}" for name in ratios)
            )
        self._verify_joint_path_disjoint_split(split_pairs, directed=directed, max_path_usage=max_path_usage)
        return split_pairs

    def _verify_joint_path_disjoint_split(self, split_pairs_by_group, directed=True, max_path_usage=None):
        path_owner = {}
        path_usage = {}
        for group_name, split_pairs in split_pairs_by_group.items():
            for split_name, pairs in split_pairs.items():
                for pair_data in pairs:
                    for path in self._pair_path_keys(pair_data, directed=directed):
                        previous_owner = path_owner.get(path)
                        if previous_owner is not None and previous_owner != split_name:
                            raise ValueError(
                                "Path leakage detected: "
                                f"path={path}, previous_split={previous_owner}, "
                                f"current_split={split_name}, group={group_name}"
                            )
                        path_owner[path] = split_name
                        path_usage[path] = path_usage.get(path, 0) + 1

        if max_path_usage is not None:
            overused = {path: count for path, count in path_usage.items() if count > max_path_usage}
            if overused:
                raise ValueError(f"Path usage cap exceeded: {overused}")

        print(
            ">> Verified joint split path-disjoint: "
            f"{len(path_owner)} unique {'directed' if directed else 'undirected'} paths, "
            f"max usage={max(path_usage.values()) if path_usage else 0}."
        )

    def _oriented_high_overlap_paths(self, edge):
        if self.high_overlap_G is None:
            return None

        u, v = edge
        data = self.high_overlap_G.get_edge_data(u, v)
        if not data or not data.get('high_overlap_valid'):
            return None
        if data.get('high_overlap_path') is None:
            return None

        ego_path = np.asarray(data.get('smooth_path', []), dtype=float)
        partner_path = np.asarray(data['high_overlap_path'], dtype=float)
        if len(ego_path) < 2 or len(ego_path) != len(partner_path):
            return None

        start_pos = np.asarray(
            self.high_overlap_G.nodes[u]['pos'], dtype=float
        )
        if np.linalg.norm(ego_path[0] - start_pos) > 0.5:
            ego_path = ego_path[::-1]
            partner_path = partner_path[::-1]
        return ego_path, partner_path, data

    def build_high_overlap_split_pairs(self, primary_split_pairs):
        """Derive one high-overlap partner path from each selected ego path."""
        if self.high_overlap_G is None:
            raise ValueError(
                "high_overlap_G is required to build high-overlap cases."
            )

        split_names = ['train', 'validate', 'test']
        output = {name: [] for name in split_names}
        skipped = {}

        for source_case_type, source_splits in primary_split_pairs.items():
            for split_name in split_names:
                for source_pair in source_splits.get(split_name, []):
                    ego_edge, _, source_role_info = self._ordered_role_edges(
                        source_pair, source_case_type
                    )
                    paths = self._oriented_high_overlap_paths(ego_edge)
                    if paths is None:
                        skipped['no_partner_path'] = (
                            skipped.get('no_partner_path', 0) + 1
                        )
                        continue

                    ego_path, partner_path, edge_data = paths
                    if (
                        self.max_world_y is not None and
                        np.any(partner_path[:, 1] > self.max_world_y)
                    ):
                        skipped['max_world_y'] = (
                            skipped.get('max_world_y', 0) + 1
                        )
                        continue

                    distances = np.linalg.norm(
                        partner_path - ego_path, axis=1
                    )
                    case_data = source_pair.copy()
                    case_data.update({
                        'generation_mode': 'highoverlap',
                        'source_case_type': source_case_type,
                        'source_split': split_name,
                        'source_score': source_pair.get('score', 0.0),
                        'ego_edge': ego_edge,
                        'ego_path': ego_path,
                        'high_overlap_path': partner_path,
                        'score': edge_data.get(
                            'high_overlap_mean_nominal_fov_iou', 0.0
                        ),
                        'high_overlap_mean_nominal_fov_iou': edge_data.get(
                            'high_overlap_mean_nominal_fov_iou', 0.0
                        ),
                        'high_overlap_distance_m': edge_data.get(
                            'high_overlap_distance_m',
                            float(np.mean(distances)),
                        ),
                        'high_overlap_translation': edge_data.get(
                            'high_overlap_translation'
                        ),
                        'high_overlap_translation_angle_deg': edge_data.get(
                            'high_overlap_translation_angle_deg'
                        ),
                        'high_overlap_min_distance_m': float(
                            np.min(distances)
                        ),
                        'high_overlap_max_distance_m': float(
                            np.max(distances)
                        ),
                        'role_info': {
                            'case_type': 'highoverlap',
                            'ego_reason': (
                                'inherited_from_'
                                f'{source_case_type}_case'
                            ),
                            'source_role_info': source_role_info,
                            'partner_reason': (
                                'translated_path_with_high_nominal_fov_overlap'
                            ),
                        },
                    })
                    output[split_name].append(case_data)

        print(
            ">> High-overlap split inherited from Shadow/Distance: "
            + ", ".join(
                f"{name}={len(output[name])}" for name in split_names
            )
            + f" | skipped={skipped}"
        )
        return output

    def collect_reserved_split_paths(
            self, primary_split_pairs, directed=True):
        reserved = set()
        for group_splits in primary_split_pairs.values():
            for pairs in group_splits.values():
                for pair_data in pairs:
                    reserved |= self._pair_path_keys(
                        pair_data, directed=directed
                    )
        return reserved

    def verify_control_case_splits(
            self,
            primary_split_pairs,
            high_overlap_split_pairs,
            random_split_pairs,
            directed=True,
            max_random_partner_usage=3):
        path_owner = {}
        primary_combinations = set()
        for group_name, group_splits in primary_split_pairs.items():
            for split_name, pairs in group_splits.items():
                for pair_data in pairs:
                    primary_combinations.add(
                        frozenset((pair_data['e1'], pair_data['e2']))
                    )
                    for path in self._pair_path_keys(
                            pair_data, directed=directed):
                        previous = path_owner.get(path)
                        if previous is not None and previous != split_name:
                            raise ValueError(
                                f"Primary path leakage: {path} "
                                f"{previous}->{split_name} ({group_name})"
                            )
                        path_owner[path] = split_name

        for split_name, pairs in high_overlap_split_pairs.items():
            for pair_data in pairs:
                ego_path_key = self._path_key(
                    pair_data['ego_edge'], directed=directed
                )
                if path_owner.get(ego_path_key) != split_name:
                    raise ValueError(
                        "High-overlap case did not inherit its ego split: "
                        f"path={ego_path_key}, split={split_name}, "
                        f"owner={path_owner.get(ego_path_key)}"
                    )

        random_pairs = set()
        random_egos = set()
        partner_usage = {}
        for split_name, pairs in random_split_pairs.items():
            for pair_data in pairs:
                ego_path = self._path_key(
                    pair_data['e1'], directed=directed
                )
                partner_path = self._path_key(
                    pair_data['e2'], directed=directed
                )
                if path_owner.get(ego_path) != split_name:
                    raise ValueError(
                        f"Random ego is not owned by {split_name}: {ego_path}"
                    )
                if path_owner.get(partner_path) != split_name:
                    raise ValueError(
                        "Random partner crosses split or is not a primary "
                        f"path: {partner_path}, split={split_name}, "
                        f"owner={path_owner.get(partner_path)}"
                    )

                combination = frozenset((pair_data['e1'], pair_data['e2']))
                if combination in primary_combinations:
                    raise ValueError(
                        f"Random duplicates a primary pair: {combination}"
                    )
                pair_key = (pair_data['e1'], pair_data['e2'])
                if pair_key in random_pairs:
                    raise ValueError(
                        f"Random directed pair is duplicated: {pair_key}"
                    )
                if ego_path in random_egos:
                    raise ValueError(
                        f"Random ego is used by multiple cases: {ego_path}"
                    )

                random_pairs.add(pair_key)
                random_egos.add(ego_path)
                partner_usage[partner_path] = (
                    partner_usage.get(partner_path, 0) + 1
                )
                if partner_usage[partner_path] > max_random_partner_usage:
                    raise ValueError(
                        "Random partner usage exceeds limit: "
                        f"{partner_path} -> {partner_usage[partner_path]}"
                    )

        print(
            ">> Verified control-case splits: "
            f"primary path owners={len(path_owner)}, "
            f"random cases={len(random_pairs)}, "
            f"partner max usage={max(partner_usage.values(), default=0)}/"
            f"{max_random_partner_usage}, no cross-split leakage or "
            "primary-pair duplication."
        )


    # ================= 3. Smooth Path Generation =================

    def _check_collision_w(self, pts):
        for pt in pts:
            r, c = self.w2g(pt[0], pt[1])
            if 0 <= r < self.h and 0 <= c < self.w:
                if self.obs_mask[r, c] == 1: return True
            else: return True
        return False

    def _get_fillet_info(self, la, lb, node_pos, radius):
        def find_idx(line, d, reverse=False):
            cur = 0
            rng = range(len(line)-1, 0, -1) if reverse else range(len(line)-1)
            for i in rng:
                cur += np.linalg.norm(line[i if reverse else i+1] - line[i-1 if reverse else i])
                if cur >= d: return i
            return 0 if reverse else len(line)-1
            
        idx_in = find_idx(la, radius, reverse=True)
        idx_out = find_idx(lb, radius, reverse=False)
        p1, p2 = la[idx_in], lb[idx_out]
        
        t = np.linspace(0, 1, 25)
        curve = ((1-t)**2)[:, None] * p1 + (2*(1-t)*t)[:, None] * node_pos + (t**2)[:, None] * p2
        return curve, idx_in, idx_out

    def _build_smooth_continuous_path(self, path_sequence, max_fillet_r=0.9):
        if len(path_sequence) < 2: return []
        segments = []
        for i in range(len(path_sequence) - 1):
            u, v = path_sequence[i], path_sequence[i+1]
            seg = np.array(self.G[u][v].get('smooth_path', [self.G.nodes[u]['pos'], self.G.nodes[v]['pos']]))
            if np.linalg.norm(seg[0] - self.G.nodes[u]['pos']) > 0.5: seg = seg[::-1]
            segments.append(seg)
            
        if len(segments) == 1: return segments[0].tolist()
        final_path, current_seg = [], segments[0]
        
        for i in range(len(segments) - 1):
            node_pos = np.array(self.G.nodes[path_sequence[i+1]]['pos'])
            next_seg = segments[i+1]
            
            if path_sequence[i] == path_sequence[i+2]: 
                final_path.extend(current_seg.tolist())
                current_seg = next_seg
                continue
            
            r, valid_curve = max_fillet_r, None
            best_in, best_out = len(current_seg) - 1, 0
            while r >= 0.15:
                curve, i_in, i_out = self._get_fillet_info(current_seg, next_seg, node_pos, r)
                if not self._check_collision_w(curve):
                    valid_curve, best_in, best_out = curve, i_in, i_out
                    break
                r -= 0.1
                
            if valid_curve is not None:
                final_path.extend(current_seg[:best_in].tolist())
                final_path.extend(valid_curve.tolist())
            else:
                final_path.extend(current_seg.tolist()) 
            current_seg = next_seg[best_out:]
            
        final_path.extend(current_seg.tolist())
        return final_path

    # ================= 5. Plotting & Exporting =================
    def _infer_case_type(self, folder_name):
        if 'highoverlap' in folder_name:
            return 'highoverlap'
        if 'random' in folder_name:
            return 'random'
        if 'distance' in folder_name:
            return 'distance'
        if 'shadow' in folder_name:
            return 'shadow'
        return 'unknown'

    def _ordered_role_edges(self, pair_data, case_type):
        e1, e2 = pair_data['e1'], pair_data['e2']
        role_info = {'case_type': case_type}

        if case_type == 'shadow':
            ego_source = pair_data.get('ego_source', 'e1')
            if ego_source == 'e2':
                ego_edge, collab_edge = e2, e1
            else:
                ego_edge, collab_edge = e1, e2

            role_info.update({
                'ego_reason': 'lower_effective_cav_sensing_ratio',
                'ego_source': ego_source,
                'collaborative_source': pair_data.get('collaborative_source', 'e2' if ego_source == 'e1' else 'e1'),
                'ego_effective_ratio': pair_data.get('ego_effective_ratio', 0.0),
                'collaborative_effective_ratio': pair_data.get('collaborative_effective_ratio', 0.0),
                'ego_no_info_ratio': pair_data.get('ego_no_info_ratio', 0.0),
                'ego_no_info_area_m2': pair_data.get('ego_no_info_area_m2', 0.0),
                'collaborative_fill_area_m2': pair_data.get('collaborative_fill_area_m2', 0.0),
                'fill_ratio': pair_data.get('fill_ratio', 0.0),
                'comp1_e1_helps_e2': pair_data.get('comp1', 0),
                'comp2_e2_helps_e1': pair_data.get('comp2', 0),
            })
            return ego_edge, collab_edge, role_info

        if case_type == 'distance':
            best_idx = pair_data['best_idx']
            s1 = self.dir_edge_states[e1][best_idx]
            s2 = self.dir_edge_states[e2][best_idx]
            gap_indices = pair_data.get('gap_indices', set())
            if gap_indices:
                pts = self.boundary_pts[list(gap_indices)]
                e1_avg_dist = float(np.linalg.norm(pts - s1['pos'], axis=1).mean())
                e2_avg_dist = float(np.linalg.norm(pts - s2['pos'], axis=1).mean())
            else:
                e1_avg_dist = 0.0
                e2_avg_dist = 0.0

            if e1_avg_dist >= e2_avg_dist:
                ego_edge, collab_edge = e1, e2
                ego_avg_dist, collab_avg_dist = e1_avg_dist, e2_avg_dist
            else:
                ego_edge, collab_edge = e2, e1
                ego_avg_dist, collab_avg_dist = e2_avg_dist, e1_avg_dist

            role_info.update({
                'ego_reason': 'farther_from_distance_gap_boundary',
                'ego_avg_gap_dist_m': ego_avg_dist,
                'collaborative_avg_gap_dist_m': collab_avg_dist,
                'e1_avg_gap_dist_m': e1_avg_dist,
                'e2_avg_gap_dist_m': e2_avg_dist,
            })
            return ego_edge, collab_edge, role_info

        if case_type == 'random':
            role_info.update({
                'ego_reason': 'random_geometry_only_assignment',
                'partner_reason': (
                    'random_remaining_path_with_bev_fov_overlap'
                ),
                'bev_fov_iou': pair_data.get('bev_fov_iou', 0.0),
                'bev_fov_overlap_area_m2': pair_data.get(
                    'bev_fov_overlap_area_m2', 0.0
                ),
            })
            return e1, e2, role_info


        return e1, e2, {'case_type': case_type, 'ego_reason': 'unmodified_unknown_case_type'}

    def save_valuable_pairs(self, golden_pairs, folder_name="path_case"):
        print(f">> Saving {len(golden_pairs)} filtered pair cases to {folder_name}/ ...")
        case_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(case_dir, exist_ok=True)
        case_type = self._infer_case_type(folder_name)

        for filename in os.listdir(case_dir):
            if filename.startswith("case_") and filename.endswith(".pkl"):
                os.remove(os.path.join(case_dir, filename))
        
        for idx, pair_data in enumerate(golden_pairs):
            ego_edge, collab_edge, role_info = self._ordered_role_edges(pair_data, case_type)
            path1 = [ego_edge[0], ego_edge[1]]
            path2 = [collab_edge[0], collab_edge[1]]
            
            r1_waypoints = self._build_smooth_continuous_path(path1)
            r2_waypoints = self._build_smooth_continuous_path(path2)
            
            output_data = pair_data.copy()
            output_data['original_e1'] = pair_data['e1']
            output_data['original_e2'] = pair_data['e2']
            output_data['ego_edge'] = ego_edge
            output_data['collaborative_edge'] = collab_edge
            output_data['R1_role'] = 'ego'
            output_data['R2_role'] = 'collaborative'
            output_data['roles_ok'] = True
            output_data['rule_ok'] = True
            output_data['role_info'] = role_info
            output_data['R1_node_sequence'] = path1
            output_data['R2_node_sequence'] = path2
            output_data['R1_waypoints'] = r1_waypoints
            output_data['R2_waypoints'] = r2_waypoints
            
            file_path = os.path.join(case_dir, f"case_{idx}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(output_data, f)
                
        print(f">> Successfully saved {len(golden_pairs)} pair cases to {case_dir}/")

    def save_high_overlap_pairs(
            self,
            high_overlap_pairs,
            folder_name='path_case_highoverlap'):
        """Save translated partner paths using the same case contract as Shadow."""
        print(
            f">> Saving {len(high_overlap_pairs)} high-overlap pair cases "
            f"to {folder_name}/ ..."
        )
        case_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(case_dir, exist_ok=True)
        for filename in os.listdir(case_dir):
            if filename.startswith('case_') and filename.endswith('.pkl'):
                os.remove(os.path.join(case_dir, filename))

        for idx, pair_data in enumerate(high_overlap_pairs):
            ego_edge = pair_data['ego_edge']
            ego_path = np.asarray(pair_data['ego_path'], dtype=float)
            partner_path = np.asarray(
                pair_data['high_overlap_path'], dtype=float
            )

            output_data = pair_data.copy()
            output_data['original_e1'] = pair_data['e1']
            output_data['original_e2'] = pair_data['e2']
            output_data['collaborative_edge'] = None
            output_data['R1_role'] = 'ego'
            output_data['R2_role'] = 'collaborative'
            output_data['roles_ok'] = True
            output_data['rule_ok'] = True
            output_data['R1_node_sequence'] = [
                ego_edge[0], ego_edge[1]
            ]
            output_data['R2_node_sequence'] = []
            output_data['R2_source_edge'] = ego_edge
            output_data['R1_waypoints'] = ego_path.tolist()
            output_data['R2_waypoints'] = partner_path.tolist()

            file_path = os.path.join(case_dir, f'case_{idx}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(output_data, f)

        print(
            f">> Successfully saved {len(high_overlap_pairs)} "
            f"high-overlap cases to {case_dir}/"
        )


    def _flatten_split_pairs(self, split_pairs, split_order=None):
        if split_order is None:
            split_order = ['train', 'validate', 'test']

        selected_pairs = []
        for split_name in split_order:
            selected_pairs.extend(split_pairs.get(split_name, []))
        return selected_pairs

    def save_split_manifest(self,
                            split_pairs,
                            all_pairs,
                            folder_name="path_case",
                            filename=None,
                            directed=True,
                            ratios=None,
                            ordered_pairs=False):
        def make_pair_key(pair_data):
            if ordered_pairs:
                return (pair_data['e1'], pair_data['e2'])
            return frozenset([pair_data['e1'], pair_data['e2']])

        pair_to_index = {
            make_pair_key(pair_data): idx
            for idx, pair_data in enumerate(all_pairs)
        }
        manifest = {
            'meta': {
                'folder': folder_name,
                'path_direction': 'directed' if directed else 'undirected',
                'ratios': ratios or {'train': 0.75, 'validate': 0.10, 'test': 0.15},
                'split_unit': 'path_identity',
                'pair_order': 'ordered' if ordered_pairs else 'unordered',
            },
            'counts': {},
            'train': [],
            'validate': [],
            'test': [],
        }

        for split_name, pairs in split_pairs.items():
            case_names = []
            for pair_data in pairs:
                key = make_pair_key(pair_data)
                if key not in pair_to_index:
                    continue
                case_names.append(f"case_{pair_to_index[key]}.pkl")
            manifest[split_name] = case_names
            manifest['counts'][split_name] = len(case_names)

        if filename is None:
            filename = f"{folder_name}_split.json"
        save_path = os.path.join(self.output_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f">> Split manifest saved to: {save_path}")

        
    def _draw_edge_with_arrow(self, ax, e, color, label=None, marker='o'):
        u, v = e
        p_u = np.array(self.G.nodes[u]['pos'])
        edge_data = self.G.get_edge_data(u, v, default={})
        
        if 'smooth_path' in edge_data:
            segment = np.array(edge_data['smooth_path'])
            if np.linalg.norm(segment[0] - p_u) > 0.5: 
                segment = segment[::-1]
        else:
            segment = np.array([p_u, self.G.nodes[v]['pos']])
            
        ax.plot(segment[:, 0], segment[:, 1], color=color, lw=3, alpha=0.85, zorder=4, label=label)
        ax.scatter(segment[0, 0], segment[0, 1], color=color, s=70, marker=marker, edgecolors='white', linewidths=1.2, zorder=5)
        
        mid_idx = len(segment) // 2
        if len(segment) >= 2:
            idx1 = max(0, mid_idx - 3)
            idx2 = min(len(segment) - 1, mid_idx + 3)
            dx = segment[idx2, 0] - segment[idx1, 0]
            dy = segment[idx2, 1] - segment[idx1, 1]
            if np.hypot(dx, dy) > 1e-3:
                mid_x, mid_y = segment[mid_idx, 0], segment[mid_idx, 1]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                            xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                            arrowprops=dict(arrowstyle="->", color=color, lw=3), zorder=6)
        return segment[mid_idx]

    def _plot_base_map(self, ax, ext):
        ax.imshow(self.flipped_img, cmap='gray', origin='lower', extent=ext, alpha=0.3)
        for u, v, data in self.G.edges(data=True):
            if 'smooth_path' in data:
                seg = np.array(data['smooth_path'])
                ax.plot(seg[:, 0], seg[:, 1], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
            else:
                p1, p2 = self.G.nodes[u]['pos'], self.G.nodes[v]['pos']
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    def visualize_golden_pairs(self, golden_pairs, filename="golden_pairs_overview.png"):
        save_path = os.path.join(self.output_dir, filename)
        print(f">> Plotting global Golden Pairs map to: {save_path}")
        
        fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
        ext = [self.org[0], self.org[0] + self.w * self.res, self.org[1], self.org[1] + self.h * self.res]
        self._plot_base_map(ax, ext)

        cmap = matplotlib.colormaps['tab20'].resampled(len(golden_pairs))
        for i, gp in enumerate(golden_pairs):
            color = cmap(i)
            e1, e2 = gp['e1'], gp['e2']
            
            mid1 = self._draw_edge_with_arrow(ax, e1, color)
            mid2 = self._draw_edge_with_arrow(ax, e2, color)
            
            ax.plot([mid1[0], mid2[0]], [mid1[1], mid2[1]], color=color, linestyle=':', linewidth=2, alpha=0.7, zorder=4)
            mid_center = (mid1 + mid2) / 2.0
            ax.text(mid_center[0], mid_center[1], str(i+1), color='white', fontsize=12, fontweight='bold',
                    ha='center', va='center', bbox=dict(facecolor=color, alpha=0.9, edgecolor='black', boxstyle='circle,pad=0.3'), zorder=10)

        ax.set_title(f"Golden Pairs Global Overview Map", fontsize=18, fontweight='bold')
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def _clear_debug_batch_images(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for filename in os.listdir(out_dir):
            if filename.endswith(".png") and (filename.startswith("pair_") or filename.startswith("batch_")):
                os.remove(os.path.join(out_dir, filename))

    def _edge_midpoint(self, edge):
        u, v = edge
        p_u = np.array(self.G.nodes[u]['pos'])
        edge_data = self.G.get_edge_data(u, v, default={})
        if 'smooth_path' in edge_data:
            segment = np.array(edge_data['smooth_path'])
            if np.linalg.norm(segment[0] - p_u) > 0.5:
                segment = segment[::-1]
        else:
            segment = np.array([p_u, self.G.nodes[v]['pos']])
        return segment[len(segment) // 2]

    def _draw_shapely_geometry(self, ax, geom, color, alpha, zorder):
        if geom is None or geom.is_empty:
            return
        geoms = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for poly in geoms:
            if poly.is_empty or not hasattr(poly, 'exterior'):
                continue
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha, linewidth=0, zorder=zorder)

    def _draw_boundary_targets(self, ax, pair_data, label_prefix):
        e1, e2 = pair_data['e1'], pair_data['e2']
        best_idx = pair_data['best_idx']
        s1 = self.dir_edge_states.get(e1, [])[best_idx]
        s2 = self.dir_edge_states.get(e2, [])[best_idx]

        if label_prefix == 'Shadow':
            overlap_geom = s1.get('VisiblePoly', ShapelyPolygon()).intersection(
                s2.get('VisiblePoly', ShapelyPolygon())
            )
            self._draw_shapely_geometry(ax, overlap_geom, '#ff9f9f', alpha=0.09, zorder=7)

            if pair_data.get('ego_source', 'e1') == 'e2':
                ego_state, collab_state = s2, s1
            else:
                ego_state, collab_state = s1, s2
            shadow_geom = ego_state.get('NoInfoPoly', ShapelyPolygon()).intersection(
                collab_state.get('VisiblePoly', ShapelyPolygon())
            )
            self._draw_shapely_geometry(ax, shadow_geom, '#8fe88a', alpha=0.12, zorder=8)
        elif label_prefix == 'Distance':
            overlap_idx = s1['V'] & s2['V']
            if overlap_idx:
                pts = self.boundary_pts[list(overlap_idx)]
                ax.scatter(pts[:, 0], pts[:, 1], c='#d62728', s=7, alpha=0.85, zorder=7)

            distance_idx = pair_data.get('gap_indices', set())
            if distance_idx:
                pts = self.boundary_pts[list(distance_idx)]
                ax.scatter(pts[:, 0], pts[:, 1], c='#1f77b4', s=10, alpha=0.9, zorder=8)

    def _draw_waypoint_path(
            self, ax, waypoints, color, marker='^', label=None):
        segment = np.asarray(waypoints, dtype=float)
        ax.plot(
            segment[:, 0], segment[:, 1],
            color=color, lw=3, alpha=0.85, zorder=4, label=label
        )
        ax.scatter(
            segment[0, 0], segment[0, 1],
            color=color, s=70, marker=marker,
            edgecolors='white', linewidths=1.2, zorder=5
        )
        mid_idx = len(segment) // 2
        if len(segment) >= 2:
            idx1 = max(0, mid_idx - 3)
            idx2 = min(len(segment) - 1, mid_idx + 3)
            delta = segment[idx2] - segment[idx1]
            if np.linalg.norm(delta) > 1e-3:
                ax.annotate(
                    '',
                    xy=segment[mid_idx] + delta * 0.1,
                    xytext=segment[mid_idx] - delta * 0.1,
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=3
                    ),
                    zorder=6,
                )
        return segment[mid_idx]


    def _draw_pair_overview(self, ax, pair_data, pair_idx, color, label_prefix):
        self._draw_boundary_targets(ax, pair_data, label_prefix)

        case_type = label_prefix.lower()
        if label_prefix == 'HighOverlap':
            ego_edge = pair_data['ego_edge']
            mid1 = self._draw_edge_with_arrow(
                ax, ego_edge, color, marker='s'
            )
            mid2 = self._draw_waypoint_path(
                ax, pair_data['high_overlap_path'], color, marker='^'
            )
        else:
            ego_edge, collab_edge, _ = self._ordered_role_edges(
                pair_data, case_type
            )
            mid1 = self._draw_edge_with_arrow(ax, ego_edge, color, marker='s')
            mid2 = self._draw_edge_with_arrow(ax, collab_edge, color, marker='^')
        ax.plot(
            [mid1[0], mid2[0]],
            [mid1[1], mid2[1]],
            color=color,
            linestyle=':',
            linewidth=1.8,
            alpha=0.75,
            zorder=4,
        )

        label_pos = (mid1 + mid2) / 2.0
        ax.text(
            label_pos[0],
            label_pos[1],
            str(pair_idx),
            color='white',
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(facecolor=color, alpha=0.95, edgecolor='white', boxstyle='circle,pad=0.25'),
            zorder=12,
        )

        score = pair_data.get('score', 0)
        if label_prefix == 'Distance':
            detail = f"#{pair_idx} score={score:.1f} gap={pair_data.get('gap_pts', 0)}"
        else:
            detail = f"#{pair_idx} score={score:.1f} fill_area={pair_data.get('collaborative_fill_area_m2', 0.0):.2f}m2"
        return detail

    def _save_pair_batches(self, pairs, out_dir, title_prefix, batch_size=10):
        self._clear_debug_batch_images(out_dir)
        if not pairs:
            print(f">> No {title_prefix} debug pairs for {out_dir}/")
            return

        ext = [self.org[0], self.org[0] + self.w * self.res, self.org[1], self.org[1] + self.h * self.res]
        num_batches = int(np.ceil(len(pairs) / float(batch_size)))
        print(f">> Plotting {len(pairs)} {title_prefix} pair(s), up to {batch_size} per map: {out_dir}/")

        cmap = matplotlib.colormaps['tab10']
        for batch_idx in range(num_batches):
            batch_pairs = pairs[batch_idx * batch_size:min(len(pairs), (batch_idx + 1) * batch_size)]
            fig, ax = plt.subplots(figsize=(14, 14), dpi=180)
            self._plot_base_map(ax, ext)

            for local_idx, pair_data in enumerate(batch_pairs):
                pair_idx = batch_idx * batch_size + local_idx
                color = cmap(local_idx % 10)
                self._draw_pair_overview(ax, pair_data, pair_idx, color, title_prefix)

            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=8)

            plt.tight_layout()
            save_path = os.path.join(out_dir, f"batch_{batch_idx:03d}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

    def visualize_debug_shadow_pairs(self, golden_pairs, folder_name="debug_shadow", batch_size=10):
        out_dir = os.path.join(self.output_dir, folder_name)
        self._save_pair_batches(golden_pairs, out_dir, "Shadow", batch_size=batch_size)

    def visualize_debug_distance_pairs(self, res_comp_pairs, folder_name="debug_distance", batch_size=10):
        out_dir = os.path.join(self.output_dir, folder_name)
        self._save_pair_batches(res_comp_pairs, out_dir, "Distance", batch_size=batch_size)


    def visualize_debug_high_overlap_pairs(
            self, pairs, folder_name='debug_highoverlap', batch_size=10):
        out_dir = os.path.join(self.output_dir, folder_name)
        self._save_pair_batches(
            pairs, out_dir, 'HighOverlap', batch_size=batch_size
        )

    def visualize_debug_random_pairs(
            self, pairs, folder_name='debug_random', batch_size=10):
        out_dir = os.path.join(self.output_dir, folder_name)
        self._save_pair_batches(
            pairs, out_dir, 'Random', batch_size=batch_size
        )

    # ------------------ 6. dist gap -----------------
    def _parallel_eval_pair_dist_gap(self, pair_info):
        """Worker for scoring resolution-compensation (distance-gap) pairs."""
        e1, e2, min_overlap, min_dist_gap, min_score, min_valid_fraction, max_robot_dist_m, min_ego_target_dist_m, max_ego_target_dist_m, min_robot_dist_m = pair_info
        states1 = self.dir_edge_states.get(e1)
        states2 = self.dir_edge_states.get(e2)
        if not states1 or not states2: return None
        
        best_score = -1
        best_idx = -1
        best_details = None
        valid_frame_count = 0
        total_frames = 0
        
        for idx, (s1, s2) in enumerate(zip(states1, states2)):
            total_frames += 1
            robot_dist = np.linalg.norm(np.array(s1['pos']) - np.array(s2['pos']))
            if robot_dist > max_robot_dist_m or robot_dist < min_robot_dist_m:
                return None

            overlap_idx = s1['V'] & s2['V']
            if len(overlap_idx) < min_overlap:
                continue
                
            # Compute each overlapping point's distance to both robots.
            pts = self.boundary_pts[list(overlap_idx)]
            d1 = np.linalg.norm(pts - s1['pos'], axis=1)
            d2 = np.linalg.norm(pts - s2['pos'], axis=1)
            
            # Select feature points whose distance difference is large enough.
            dist_diffs = np.abs(d1 - d2)
            gap_mask = dist_diffs >= min_dist_gap
            valid_pts_count = np.sum(gap_mask)
            if valid_pts_count == 0:
                continue

            gap_d1 = d1[gap_mask]
            gap_d2 = d2[gap_mask]
            e1_avg_gap_dist = float(np.mean(gap_d1))
            e2_avg_gap_dist = float(np.mean(gap_d2))
            ego_avg_gap_dist = max(e1_avg_gap_dist, e2_avg_gap_dist)
            if ego_avg_gap_dist < min_ego_target_dist_m or ego_avg_gap_dist > max_ego_target_dist_m:
                continue

            if valid_pts_count >= min_score:
                valid_frame_count += 1
            
            # Score the pair by the number of overlap points with a clear
            # distance gap.
            if valid_pts_count > best_score:
                best_score = valid_pts_count
                best_idx = idx
                gap_indices = set(np.array(list(overlap_idx))[gap_mask])
                best_details = {
                    'overlap': len(overlap_idx),
                    'gap_pts': int(valid_pts_count),
                    'gap_indices': gap_indices,
                    'avg_gap': float(np.mean(dist_diffs)) if len(dist_diffs) > 0 else 0.0,
                    'e1_avg_gap_dist_m': e1_avg_gap_dist,
                    'e2_avg_gap_dist_m': e2_avg_gap_dist,
                    'ego_avg_gap_dist_m': ego_avg_gap_dist,
                }

        if total_frames == 0:
            return None
        valid_fraction = valid_frame_count / float(total_frames)
        if valid_fraction < min_valid_fraction or best_score < min_score:
            return None

        return {
            'e1': e1, 'e2': e2, 'score': best_score, 'best_idx': best_idx,
            'overlap': best_details['overlap'], 'gap_pts': best_details['gap_pts'], 
            'gap_indices': best_details['gap_indices'], 'avg_gap': best_details['avg_gap'],
            'e1_avg_gap_dist_m': best_details['e1_avg_gap_dist_m'],
            'e2_avg_gap_dist_m': best_details['e2_avg_gap_dist_m'],
            'ego_avg_gap_dist_m': best_details['ego_avg_gap_dist_m'],
            'min_ego_target_dist_m': min_ego_target_dist_m,
            'max_ego_target_dist_m': max_ego_target_dist_m,
            'min_robot_dist_m': min_robot_dist_m,
            'max_robot_dist_m': max_robot_dist_m,
            'valid_fraction': valid_fraction, 'valid_frames': valid_frame_count,
            'total_frames': total_frames
        }

    def get_resolution_comp_pairs(self, min_overlap=15, min_dist_gap=5.0, min_score=10, min_valid_fraction=0.25, max_robot_dist_m=20.0, min_ego_target_dist_m=7.5, max_ego_target_dist_m=22.4, min_robot_dist_m=5.0):
        """Find candidate pairs with complementary near and far viewpoints."""
        print(
            f">> Extracting Resolution Compensation pairs "
            f"(Distance gap > {min_dist_gap}m, ego target distance {min_ego_target_dist_m}-{max_ego_target_dist_m}m, "
            f"robot distance {min_robot_dist_m}-{max_robot_dist_m}m)..."
        )
        edges = list(self.dir_edge_states.keys())
        pairs_to_eval = []
        for i in range(len(edges)):
            e1 = edges[i]
            for e2 in edges[i+1:]:
                if set(e1).isdisjoint(set(e2)):
                    pairs_to_eval.append((e1, e2, min_overlap, min_dist_gap, min_score, min_valid_fraction, max_robot_dist_m, min_ego_target_dist_m, max_ego_target_dist_m, min_robot_dist_m))
                    
        reward_map = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._parallel_eval_pair_dist_gap, pairs_to_eval, chunksize=100))
            
        for res in results:
            if res is not None:
                reward_map[(res['e1'], res['e2'])] = res
                reward_map[(res['e2'], res['e1'])] = res
                
        print(
            f">> Found {len(reward_map)//2} resolution compensation combinations with "
            f">= {min_valid_fraction:.0%} valid frames, robot distance <= {max_robot_dist_m:.1f}m, "
            f"ego target distance {min_ego_target_dist_m:.1f}-{max_ego_target_dist_m:.1f}m, "
            f"and robot distance >= {min_robot_dist_m:.1f}m."
        )
        return reward_map

    def extract_filtered_res_comp_pairs(self, reward_map, min_new_gap_pts=None):
        """Return unique distance pairs sorted by score. New-info filtering is disabled."""
        unique_pairs = {}
        for (e1, e2), data in reward_map.items():
            key = frozenset([e1, e2])
            if key not in unique_pairs:
                unique_pairs[key] = data

        selected_pairs = sorted(unique_pairs.values(), key=lambda x: x['score'], reverse=True)
        print(f">> New-info filtering disabled. Keeping {len(selected_pairs)} unique distance pairs.")
        return selected_pairs

def run_dual_scene(raw_scene_name, output_scene_name, max_world_y=None, random_seed=42):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_map_dir = os.path.join(repo_root, "map", "raw_maps")
    trajectory_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory")

    print(f"\n{'=' * 60}")
    print(f">> Building dual cases for raw map: {raw_scene_name} -> {output_scene_name}")
    print(f"{'=' * 60}")

    planner = DualCoveragePlanner(
        png_path=os.path.join(raw_map_dir, f"{raw_scene_name}.png"),
        yaml_path=os.path.join(raw_map_dir, f"{raw_scene_name}.yaml"),
        roadmap_path=os.path.join(trajectory_root, output_scene_name, "planner", "roadmap.pkl"),
        high_overlap_roadmap_path=os.path.join(
            trajectory_root, output_scene_name, "planner",
            "roadmap_highoverlap.pkl",
        ),
        num_workers=2,
        max_dist_m=20.0,
        fov_deg=89.0,
        boundary_step_m=0.1,
        obs_interval_m=0.3,
        min_grazing_deg=5.0,
        output_scene_name=output_scene_name,
        output_root=trajectory_root,
        max_world_y=max_world_y,
    )
    
    planner.precompute_directed_edges()
    
    print("\n--- Task A: Shadow Compensation ---")
    reward_map_shadow = planner.get_all_useful_pairs(
        min_overlap=15,
        min_score=30.0,
        min_valid_fraction=0.25,
        max_robot_dist_m=20.0,
    )
    filtered_shadow_pairs = planner.extract_filtered_pairs(reward_map_shadow)

    print("\n--- Task B: Resolution Compensation ---")
    reward_map_res = planner.get_resolution_comp_pairs(
        min_overlap=10,
        min_dist_gap=5.0,
        min_score=15,
        min_valid_fraction=0.25,
        max_robot_dist_m=20.0,
        min_ego_target_dist_m=7.5,
        max_ego_target_dist_m=22.4,
        min_robot_dist_m=5.0,
    )
    filtered_res_pairs = planner.extract_filtered_res_comp_pairs(reward_map_res)

    split_ratios = {'train': 0.75, 'validate': 0.10, 'test': 0.15}
    primary_split_pairs = planner.extract_joint_path_disjoint_split_pairs(
        {
            'shadow': filtered_shadow_pairs,
            'distance': filtered_res_pairs,
        },
        ratios=split_ratios,
        directed=True,
        random_seed=random_seed,
        max_path_usage=1,
    )

    print("\n--- Task C: High FOV Overlap from Primary Ego Paths ---")
    high_overlap_split_pairs = planner.build_high_overlap_split_pairs(
        primary_split_pairs
    )

    print("\n--- Task D: Fixed-Ego Random Same-Split FOV Overlap ---")
    random_candidates = planner.get_random_pairs_for_primary_egos(
        primary_split_pairs,
        min_bev_fov_iou=0.05,
        min_overlap_area_m2=5.0,
        min_valid_fraction=0.25,
        min_robot_dist_m=1.0,
        max_robot_dist_m=20.0,
    )
    random_split_pairs = planner.select_fixed_ego_random_split_pairs(
        random_candidates,
        max_partner_usage=3,
        random_seed=random_seed,
    )
    planner.verify_control_case_splits(
        primary_split_pairs,
        high_overlap_split_pairs,
        random_split_pairs,
        directed=True,
        max_random_partner_usage=3,
    )

    selected_shadow_pairs = planner._flatten_split_pairs(
        primary_split_pairs['shadow']
    )
    selected_res_pairs = planner._flatten_split_pairs(
        primary_split_pairs['distance']
    )
    selected_high_overlap_pairs = planner._flatten_split_pairs(
        high_overlap_split_pairs
    )
    selected_random_pairs = planner._flatten_split_pairs(
        random_split_pairs
    )

    for split_name, split_pairs in primary_split_pairs['shadow'].items():
        planner.visualize_debug_shadow_pairs(
            split_pairs,
            folder_name=os.path.join('debug_shadow', split_name),
            batch_size=10,
        )

    for split_name, split_pairs in primary_split_pairs['distance'].items():
        planner.visualize_debug_distance_pairs(
            split_pairs,
            folder_name=os.path.join('debug_distance', split_name),
            batch_size=10,
        )

    for split_name, split_pairs in high_overlap_split_pairs.items():
        planner.visualize_debug_high_overlap_pairs(
            split_pairs,
            folder_name=os.path.join('debug_highoverlap', split_name),
            batch_size=10,
        )

    for split_name, split_pairs in random_split_pairs.items():
        planner.visualize_debug_random_pairs(
            split_pairs,
            folder_name=os.path.join('debug_random', split_name),
            batch_size=10,
        )

    if selected_shadow_pairs:
        planner.save_valuable_pairs(
            selected_shadow_pairs, folder_name='path_case_shadow'
        )
        planner.save_split_manifest(
            primary_split_pairs['shadow'],
            selected_shadow_pairs,
            folder_name='path_case_shadow',
            directed=True,
            ratios=split_ratios,
        )

    if selected_res_pairs:
        planner.save_valuable_pairs(
            selected_res_pairs, folder_name='path_case_distance'
        )
        planner.save_split_manifest(
            primary_split_pairs['distance'],
            selected_res_pairs,
            folder_name='path_case_distance',
            directed=True,
            ratios=split_ratios,
        )

    planner.save_high_overlap_pairs(
        selected_high_overlap_pairs,
        folder_name='path_case_highoverlap',
    )
    planner.save_split_manifest(
        high_overlap_split_pairs,
        selected_high_overlap_pairs,
        folder_name='path_case_highoverlap',
        directed=True,
        ratios=split_ratios,
    )

    planner.save_valuable_pairs(
        selected_random_pairs,
        folder_name='path_case_random',
    )
    planner.save_split_manifest(
        random_split_pairs,
        selected_random_pairs,
        folder_name='path_case_random',
        directed=True,
        ratios=split_ratios,
        ordered_pairs=True,
    )


if __name__ == "__main__":
    scenes = [
        ("hospital", "hospital", None),
        ("warehouse", "warehouse", 10.0),
        ("office", "office", None),
    ]
    for raw_scene_name, output_scene_name, max_world_y in scenes:
        run_dual_scene(raw_scene_name, output_scene_name, max_world_y=max_world_y, random_seed=42)
