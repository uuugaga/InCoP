import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import json
import pickle, cv2, os, networkx as nx
import yaml
from scipy.spatial import KDTree
import concurrent.futures
import multiprocessing

class DualCoveragePlanner:
    def __init__(self, 
                 png_path, 
                 yaml_path, 
                 roadmap_path, 
                 max_dist_m=20.0, 
                 fov_deg=89.0,
                 boundary_step_m=0.1,
                 obs_interval_m=0.3,
                 min_grazing_deg=5.0):
        # --- Basic Initialization ---
        self.scene_name = os.path.splitext(os.path.basename(png_path))[0]
        self.output_dir = os.path.join("trajectory", self.scene_name, "dual")
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
        self._repair_graph_weights()
        
        self.max_dist_m = max_dist_m
        self.fov_deg = fov_deg
        self.boundary_step_m = boundary_step_m
        self.obs_interval_m = obs_interval_m
        self.parallel_tol_cos = -np.sin(np.radians(min_grazing_deg))
        
        self.boundary_pts = self.extract_boundary_points(step_m=self.boundary_step_m)
        self.target_tree = KDTree(self.boundary_pts)
        
        self.dir_edge_states = {} 
        self.num_workers = multiprocessing.cpu_count()

    # ================= 1. Graph & Environment Setup =================

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

    # ================= 2. Synergy Precomputation =================

    def _calc_edge_states(self, start_n, end_n, data, is_reverse=False):
        states = []
        num_frames = 10 
        path = np.array(data['smooth_path']) if data and 'smooth_path' in data else np.array([self.G.nodes[start_n]['pos'], self.G.nodes[end_n]['pos']])
        if np.linalg.norm(path[0] - self.G.nodes[start_n]['pos']) > 0.5: path = path[::-1]
        if is_reverse: path = path[::-1]

        indices = np.linspace(0, len(path) - 1, num_frames).astype(int)
        for i in range(len(indices)-1):
            pos = path[indices[i]]
            r_idx, c_idx = self.w2g(*pos)
            if self.obs_mask[np.clip(r_idx, 0, self.h-1), np.clip(c_idx, 0, self.w-1)] == 1: continue 

            delta = path[indices[i+1]] - pos
            heading = np.degrees(np.arctan2(delta[1], delta[0]))
            
            near_idx = self.target_tree.query_ball_point(pos, self.max_dist_m)
            if not near_idx: continue
            
            rel_pts = self.boundary_pts[near_idx] - pos
            angles = np.degrees(np.arctan2(rel_pts[:, 1], rel_pts[:, 0]))
            diff = (angles - heading + 180) % 360 - 180
            in_fov = np.abs(diff) < (self.fov_deg / 2.0)
            
            fov_idx = [near_idx[j] for j, is_in in enumerate(in_fov) if is_in]
            seen_idx = {idx for idx in fov_idx if self._check_visibility(pos, self.boundary_pts[idx])}
            shadow_idx = set(fov_idx) - seen_idx 
            states.append({'pos': pos, 'heading': heading, 'V': seen_idx, 'Shadow': shadow_idx})
        return states

    def _parallel_calc_edge(self, edge_info):
        u, v, data = edge_info
        return (u, v, self._calc_edge_states(u, v, data, False), self._calc_edge_states(v, u, data, True))

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
        e1, e2, min_overlap = pair_info
        states1 = self.dir_edge_states[e1]
        states2 = self.dir_edge_states[e2]
        if not states1 or not states2: return None
        
        comp_list = []
        for s1, s2 in zip(states1, states2):
            exact_overlap_pts = len(s1['V'] & s2['V'])
            if exact_overlap_pts < min_overlap:
                comp_list.append(0)
                continue 
            comp_list.append(len(s1['Shadow'] & s2['V']) + len(s2['Shadow'] & s1['V']))
        
        if any(c > 0 for c in comp_list):
            best_idx = int(np.argmax(comp_list))
            s1_b, s2_b = states1[best_idx], states2[best_idx]
            overlap = len(s1_b['V'] & s2_b['V'])
            comp1 = len(s2_b['Shadow'] & s1_b['V']) 
            comp2 = len(s1_b['Shadow'] & s2_b['V']) 
            score = comp1 + comp2 + (overlap * 0.1) 
            
            return {
                'e1': e1, 'e2': e2, 'score': score, 'best_idx': best_idx,
                'overlap': overlap, 'comp1': comp1, 'comp2': comp2
            }
        return None

    def get_all_useful_pairs(self, min_overlap=15, min_score=5.0):
        print(f">> Extracting ALL valuable synchronization pairs...")
        edges = list(self.dir_edge_states.keys())
        pairs_to_eval = []
        for i in range(len(edges)):
            e1 = edges[i]
            for e2 in edges[i+1:]:
                # set(e1) 和 set(e2) 若完全沒有交集 (isdisjoint == True)，才加入評估
                if set(e1).isdisjoint(set(e2)): 
                    pairs_to_eval.append((e1, e2, min_overlap))
        
        reward_map = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._parallel_eval_pair, pairs_to_eval, chunksize=100))
            
        for res in results:
            if res is not None and res['score'] >= min_score:
                reward_map[(res['e1'], res['e2'])] = res
                reward_map[(res['e2'], res['e1'])] = res
                
        print(f">> Found {len(reward_map)//2} highly valuable synergy combinations.")
        return reward_map

    def extract_filtered_pairs(self, reward_map, min_new_info=15):
        """萃取所有能提供足夠新視野 (new_info > min_new_info) 的不重複黃金組合"""
        unique_pairs = {}
        for (e1, e2), data in reward_map.items():
            key = frozenset([e1, e2])
            if key not in unique_pairs:
                unique_pairs[key] = data
                
        # 依分數從高到低排序
        sorted_pairs = sorted(unique_pairs.values(), key=lambda x: x['score'], reverse=True)
        
        selected_pairs = []
        covered_targets = set() # 用來追蹤「已經被前面更高分組合看過」的特徵點
        
        for gp in sorted_pairs:
            s1 = self.dir_edge_states[gp['e1']][gp['best_idx']]
            s2 = self.dir_edge_states[gp['e2']][gp['best_idx']]
            v_union = s1['V'] | s2['V'] # 這組 pair 總共能看到的點
            
            # 集合運算：扣除掉已經被看過的點，計算這組 pair 能帶來多少「新資訊」
            new_info = len(v_union - covered_targets)
            
            if new_info > min_new_info: 
                selected_pairs.append(gp)
                covered_targets |= v_union # 將這些新看到的點加入已覆蓋清單
                
        print(f">> 經過視野重複性過濾 (new_info > {min_new_info})，共保留 {len(selected_pairs)} 組精華組合。")
        return selected_pairs

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
            ratios = {'train': 0.70, 'validate': 0.15, 'test': 0.15}

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
                                                directed=True):
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
            ratios = {'train': 0.70, 'validate': 0.15, 'test': 0.15}
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

        all_candidates.sort(
            key=lambda item: (
                -item[2].get('score', 0),
                -max(path_freq[path] for path in self._pair_path_keys(item[2], directed=directed)),
            )
        )

        split_pairs = {
            group_name: {split_name: [] for split_name in ratios}
            for group_name in case_groups
        }
        split_counts = {split_name: 0 for split_name in ratios}
        split_seen = {split_name: set() for split_name in ratios}
        path_owner = {}
        added_candidate_ids = set()
        skipped_bridges = {1: 0, 2: 0}
        skipped_low_info = 0
        added_by_pass = {1: 0, 2: 0}

        def choose_split(pair_paths):
            owned_splits = {path_owner[path] for path in pair_paths if path in path_owner}
            if len(owned_splits) > 1:
                return None
            if owned_splits:
                return next(iter(owned_splits))
            return min(
                ratios,
                key=lambda name: (
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

            for path in self._pair_path_keys(pair_data, directed=directed):
                path_owner[path] = split_name

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
            split_name = choose_split(pair_paths)
            if split_name is None:
                skipped_bridges[1] += 1
                continue

            s1 = self.dir_edge_states[pair_data['e1']][pair_data['best_idx']]
            s2 = self.dir_edge_states[pair_data['e2']][pair_data['best_idx']]
            visible_union = s1['V'] | s2['V']
            min_new_info = min_new_info_by_group.get(group_name, 0)
            new_info = len(visible_union - split_seen[split_name])
            if new_info <= min_new_info:
                skipped_low_info += 1
                continue

            add_candidate(candidate_id, group_name, pair_data, split_name)
            added_by_pass[1] += 1

        for candidate_id, group_name, pair_data in all_candidates:
            if candidate_id in added_candidate_ids:
                continue

            pair_paths = self._pair_path_keys(pair_data, directed=directed)
            split_name = choose_split(pair_paths)
            if split_name is None:
                skipped_bridges[2] += 1
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
            + f"low-info pass1={skipped_low_info}, "
            + f"direction={'directed' if directed else 'undirected'}"
        )
        for group_name in case_groups:
            print(
                f">>   {group_name}: "
                + ", ".join(f"{name}={len(split_pairs[group_name][name])}" for name in ratios)
            )
        self._verify_joint_path_disjoint_split(split_pairs, directed=directed)
        return split_pairs

    def _verify_joint_path_disjoint_split(self, split_pairs_by_group, directed=True):
        path_owner = {}
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

        print(
            ">> Verified joint split path-disjoint: "
            f"{len(path_owner)} unique {'directed' if directed else 'undirected'} paths."
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
    def save_valuable_pairs(self, golden_pairs, folder_name="path_case"):
        print(f">> Saving {len(golden_pairs)} filtered pair cases to {folder_name}/ ...")
        case_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(case_dir, exist_ok=True)

        for filename in os.listdir(case_dir):
            if filename.startswith("case_") and filename.endswith(".pkl"):
                os.remove(os.path.join(case_dir, filename))
        
        for idx, pair_data in enumerate(golden_pairs):
            path1 = [pair_data['e1'][0], pair_data['e1'][1]]
            path2 = [pair_data['e2'][0], pair_data['e2'][1]]
            
            r1_waypoints = self._build_smooth_continuous_path(path1)
            r2_waypoints = self._build_smooth_continuous_path(path2)
            
            output_data = pair_data.copy()
            output_data['R1_node_sequence'] = path1
            output_data['R2_node_sequence'] = path2
            output_data['R1_waypoints'] = r1_waypoints
            output_data['R2_waypoints'] = r2_waypoints
            
            file_path = os.path.join(case_dir, f"case_{idx}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(output_data, f)
                
        print(f">> Successfully saved {len(golden_pairs)} pair cases to {case_dir}/")

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
                            ratios=None):
        pair_to_index = {
            frozenset([pair_data['e1'], pair_data['e2']]): idx
            for idx, pair_data in enumerate(all_pairs)
        }
        manifest = {
            'meta': {
                'folder': folder_name,
                'path_direction': 'directed' if directed else 'undirected',
                'ratios': ratios or {'train': 0.70, 'validate': 0.15, 'test': 0.15},
                'split_unit': 'path_identity',
            },
            'counts': {},
            'train': [],
            'validate': [],
            'test': [],
        }

        for split_name, pairs in split_pairs.items():
            case_names = []
            for pair_data in pairs:
                pair_key = frozenset([pair_data['e1'], pair_data['e2']])
                if pair_key not in pair_to_index:
                    continue
                case_names.append(f"case_{pair_to_index[pair_key]}.pkl")
            manifest[split_name] = case_names
            manifest['counts'][split_name] = len(case_names)

        if filename is None:
            filename = f"{folder_name}_split.json"
        save_path = os.path.join(self.output_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f">> Split manifest saved to: {save_path}")

        
    def _draw_edge_with_arrow(self, ax, e, color, label=None):
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
        ax.scatter(segment[0, 0], segment[0, 1], color=color, s=50, edgecolors='white', zorder=5)
        
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

    def visualize_debug_shadow_pairs(self, golden_pairs):
        print(f">> Plotting {len(golden_pairs)} debug snapshots to: {self.debug_dir}/")
        for filename in os.listdir(self.debug_dir):
            if filename.startswith("pair_") and filename.endswith(".png"):
                os.remove(os.path.join(self.debug_dir, filename))

        ext = [self.org[0], self.org[0] + self.w * self.res, self.org[1], self.org[1] + self.h * self.res]
        all_idx = set(range(len(self.boundary_pts)))
        
        for i, gp in enumerate(golden_pairs):
            fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
            self._plot_base_map(ax, ext)
            
            e1, e2 = gp['e1'], gp['e2']
            best_idx = gp['best_idx']
            s1 = self.dir_edge_states[e1][best_idx]
            s2 = self.dir_edge_states[e2][best_idx]
            
            fov_poly1 = Polygon(self._get_fov_polygon(s1['pos'], s1['heading']), closed=True, facecolor='#1f77b4', edgecolor='none', alpha=0.25)
            fov_poly2 = Polygon(self._get_fov_polygon(s2['pos'], s2['heading']), closed=True, facecolor='#ff7f0e', edgecolor='none', alpha=0.25)
            ax.add_patch(fov_poly1)
            ax.add_patch(fov_poly2)
            
            overlap_idx = s1['V'] & s2['V']
            comp1_idx = s2['Shadow'] & s1['V']   
            comp2_idx = s1['Shadow'] & s2['V']   
            highlight_idx = overlap_idx | comp1_idx | comp2_idx
            unseen_idx = all_idx - highlight_idx
            
            if unseen_idx: ax.scatter(self.boundary_pts[list(unseen_idx), 0], self.boundary_pts[list(unseen_idx), 1], c='#d62728', s=1, label='Unseen')
            if overlap_idx: ax.scatter(self.boundary_pts[list(overlap_idx), 0], self.boundary_pts[list(overlap_idx), 1], c='#2ca02c', s=1, zorder=6, label='Overlap (Anchor)')
            if comp1_idx: ax.scatter(self.boundary_pts[list(comp1_idx), 0], self.boundary_pts[list(comp1_idx), 1], c='#1f77b4', s=1, zorder=5, label='Comp 1 (R1 helps R2)')
            if comp2_idx: ax.scatter(self.boundary_pts[list(comp2_idx), 0], self.boundary_pts[list(comp2_idx), 1], c='#ff7f0e', s=1, zorder=5, label='Comp 2 (R2 helps R1)')
            
            self._draw_edge_with_arrow(ax, e1, '#1f77b4', label='R1 Path')
            self._draw_edge_with_arrow(ax, e2, '#ff7f0e', label='R2 Path')
            
            title_text = (f"Golden Pair #{i} - Best Instant Snapshot\n"
                          f"Final Score: {gp['score']:.2f} | Overlap: {len(overlap_idx)} | Comp1 (Blue): {len(comp1_idx)} | Comp2 (Orange): {len(comp2_idx)}")
            ax.set_title(title_text, fontsize=16, fontweight='bold', family='monospace')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.debug_dir, f"pair_{i}.png"), bbox_inches='tight')
            plt.close(fig)

    def visualize_debug_distance_pairs(self, res_comp_pairs, folder_name="debug_distance"):
        """繪製解析度補償的 Debug 圖片，以螢光色突顯一遠一近的補償點"""
        out_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f">> Plotting {len(res_comp_pairs)} Resolution Comp snapshots to: {out_dir}/")
        for filename in os.listdir(out_dir):
            if filename.startswith("pair_") and filename.endswith(".png"):
                os.remove(os.path.join(out_dir, filename))

        ext = [self.org[0], self.org[0] + self.w * self.res, self.org[1], self.org[1] + self.h * self.res]
        all_idx = set(range(len(self.boundary_pts)))
        
        for i, gp in enumerate(res_comp_pairs):
            fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
            self._plot_base_map(ax, ext)
            
            e1, e2, best_idx = gp['e1'], gp['e2'], gp['best_idx']
            s1, s2 = self.dir_edge_states[e1][best_idx], self.dir_edge_states[e2][best_idx]
            
            # 畫出 FOV 扇形
            ax.add_patch(Polygon(self._get_fov_polygon(s1['pos'], s1['heading']), closed=True, facecolor='#1f77b4', alpha=0.2))
            ax.add_patch(Polygon(self._get_fov_polygon(s2['pos'], s2['heading']), closed=True, facecolor='#ff7f0e', alpha=0.2))
            
            overlap_idx = s1['V'] & s2['V']
            gap_indices = gp['gap_indices']
            normal_overlap = overlap_idx - gap_indices
            unseen_idx = all_idx - overlap_idx
            
            # 繪製點雲
            if unseen_idx: ax.scatter(self.boundary_pts[list(unseen_idx), 0], self.boundary_pts[list(unseen_idx), 1], c='#d62728', s=1, alpha=0.3, label='Unseen/Irrelevant')
            if normal_overlap: ax.scatter(self.boundary_pts[list(normal_overlap), 0], self.boundary_pts[list(normal_overlap), 1], c='#2ca02c', s=3, zorder=5, label='Normal Overlap')
            
            # 突顯高距離落差的補償點 (螢光粉紅色)
            if gap_indices: ax.scatter(self.boundary_pts[list(gap_indices), 0], self.boundary_pts[list(gap_indices), 1], c='#e377c2', s=15, zorder=6, marker='*', label='High Distance Gap Pts')
            
            self._draw_edge_with_arrow(ax, e1, '#1f77b4', label='R1 Path')
            self._draw_edge_with_arrow(ax, e2, '#ff7f0e', label='R2 Path')
            
            title_text = (f"Res-Comp Pair #{i} | Gap Pts: {gp['gap_pts']} | Avg Gap Dist: {gp['avg_gap']:.2f}m\n"
                          f"Overlap: {len(overlap_idx)}")
            ax.set_title(title_text, fontsize=16, fontweight='bold', family='monospace')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pair_{i}.png"), bbox_inches='tight')
            plt.close(fig)

    # ------------------ 6. dist gap -----------------
    def _parallel_eval_pair_dist_gap(self, pair_info):
        """專用於計算「解析度補償 (Distance Gap)」的平行運算核心"""
        e1, e2, min_overlap, min_dist_gap = pair_info
        states1 = self.dir_edge_states.get(e1)
        states2 = self.dir_edge_states.get(e2)
        if not states1 or not states2: return None
        
        best_score = -1
        best_idx = -1
        best_details = None
        
        for idx, (s1, s2) in enumerate(zip(states1, states2)):
            overlap_idx = s1['V'] & s2['V']
            if len(overlap_idx) < min_overlap:
                continue
                
            # 計算重疊點分別到兩台機器人的距離
            pts = self.boundary_pts[list(overlap_idx)]
            d1 = np.linalg.norm(pts - s1['pos'], axis=1)
            d2 = np.linalg.norm(pts - s2['pos'], axis=1)
            
            # 計算距離差，並找出差值大於 min_dist_gap 的特徵點
            dist_diffs = np.abs(d1 - d2)
            gap_mask = dist_diffs >= min_dist_gap
            valid_pts_count = np.sum(gap_mask)
            
            # 分數定義為：符合「明顯距離差距」的重疊點數量
            if valid_pts_count > best_score:
                best_score = valid_pts_count
                best_idx = idx
                gap_indices = set(np.array(list(overlap_idx))[gap_mask])
                best_details = {
                    'overlap': len(overlap_idx),
                    'gap_pts': valid_pts_count,
                    'gap_indices': gap_indices,
                    'avg_gap': float(np.mean(dist_diffs)) if len(dist_diffs) > 0 else 0.0
                }
                
        if best_score > 0:
            return {
                'e1': e1, 'e2': e2, 'score': best_score, 'best_idx': best_idx,
                'overlap': best_details['overlap'], 'gap_pts': best_details['gap_pts'], 
                'gap_indices': best_details['gap_indices'], 'avg_gap': best_details['avg_gap']
            }
        return None

    def get_resolution_comp_pairs(self, min_overlap=15, min_dist_gap=5.0, min_score=10):
        """尋找所有具有「一遠一近」互補特性的候選組合"""
        print(f">> Extracting Resolution Compensation pairs (Distance gap > {min_dist_gap}m)...")
        edges = list(self.dir_edge_states.keys())
        pairs_to_eval = []
        for i in range(len(edges)):
            e1 = edges[i]
            for e2 in edges[i+1:]:
                if set(e1).isdisjoint(set(e2)):
                    pairs_to_eval.append((e1, e2, min_overlap, min_dist_gap))
                    
        reward_map = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._parallel_eval_pair_dist_gap, pairs_to_eval, chunksize=100))
            
        for res in results:
            if res is not None and res['score'] >= min_score:
                reward_map[(res['e1'], res['e2'])] = res
                reward_map[(res['e2'], res['e1'])] = res
                
        print(f">> Found {len(reward_map)//2} resolution compensation combinations.")
        return reward_map

    def extract_filtered_res_comp_pairs(self, reward_map, min_new_gap_pts=10):
        """從候選組合中萃取能提供足夠「新」遠近補償視角的不重複精華組合"""
        unique_pairs = {}
        for (e1, e2), data in reward_map.items():
            key = frozenset([e1, e2])
            if key not in unique_pairs:
                unique_pairs[key] = data
                
        # 依據符合距離差距的特徵點數量 (score) 排序
        sorted_pairs = sorted(unique_pairs.values(), key=lambda x: x['score'], reverse=True)
        
        selected_pairs = []
        covered_gap_targets = set() 
        
        for gp in sorted_pairs:
            # 集合運算：扣除掉已經被前面的組合補償過的點
            new_info = len(gp['gap_indices'] - covered_gap_targets)
            
            if new_info >= min_new_gap_pts:
                selected_pairs.append(gp)
                covered_gap_targets |= gp['gap_indices']
                
        print(f">> 經過視野重複性過濾，共保留 {len(selected_pairs)} 組「遠近補償」精華組合。")
        return selected_pairs

if __name__ == "__main__":
    planner = DualCoveragePlanner(
        png_path='../cropped_maps/hospital.png', 
        yaml_path='../cropped_maps/hospital.yaml', 
        roadmap_path='trajectory/hospital/planner/roadmap.pkl',
        max_dist_m=25.0, 
        fov_deg=80.0, 
        boundary_step_m=0.1, 
        obs_interval_m=0.3, 
        min_grazing_deg=5.0
    )
    
    # 1. 平行預計算
    planner.precompute_directed_edges()
    
    print("\n--- Task A: Shadow Compensation ---")
    reward_map_shadow = planner.get_all_useful_pairs(min_overlap=15, min_score=30.0)
    filtered_shadow_pairs = planner.extract_filtered_pairs(reward_map_shadow, min_new_info=5)

    print("\n--- Task B: Resolution Compensation ---")
    reward_map_res = planner.get_resolution_comp_pairs(min_overlap=10, min_dist_gap=5.0, min_score=5)
    filtered_res_pairs = planner.extract_filtered_res_comp_pairs(reward_map_res, min_new_gap_pts=1)

    joint_split_pairs = planner.extract_joint_path_disjoint_split_pairs(
        {
            'shadow': filtered_shadow_pairs,
            'distance': filtered_res_pairs,
        },
        ratios={'train': 0.70, 'validate': 0.15, 'test': 0.15},
        directed=True
    )

    selected_shadow_pairs = planner._flatten_split_pairs(joint_split_pairs['shadow'])
    selected_res_pairs = planner._flatten_split_pairs(joint_split_pairs['distance'])

    if selected_shadow_pairs:
        planner.visualize_debug_shadow_pairs(selected_shadow_pairs)
        planner.save_valuable_pairs(selected_shadow_pairs, folder_name="path_case_shadow")
        planner.save_split_manifest(
            joint_split_pairs['shadow'],
            selected_shadow_pairs,
            folder_name="path_case_shadow",
            directed=True,
            ratios={'train': 0.70, 'validate': 0.15, 'test': 0.15}
        )
    
    if selected_res_pairs:
        planner.visualize_debug_distance_pairs(selected_res_pairs, folder_name="debug_distance")
        planner.save_valuable_pairs(selected_res_pairs, folder_name="path_case_distance")
        planner.save_split_manifest(
            joint_split_pairs['distance'],
            selected_res_pairs,
            folder_name="path_case_distance",
            directed=True,
            ratios={'train': 0.70, 'validate': 0.15, 'test': 0.15}
        )
