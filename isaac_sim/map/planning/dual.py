import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
import numpy as np
import pickle, cv2, os, networkx as nx
import yaml
from scipy.spatial import KDTree
import concurrent.futures
import multiprocessing
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

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

    # ================= 3. Dual Route Planning =================

    def _get_path_with_no_uturn(self, start_node, target_node, prev_node):
        """Helper to find shortest path while strictly preventing U-Turns unless dead end."""
        temporarily_removed = None
        if prev_node is not None and self.G.has_edge(start_node, prev_node):
            edge_data = self.G.get_edge_data(start_node, prev_node)
            self.G.remove_edge(start_node, prev_node)
            temporarily_removed = (start_node, prev_node, edge_data)

        path, dist = None, float('inf')
        try:
            path = nx.dijkstra_path(self.G, start_node, target_node, weight='weight')
            dist = nx.path_weight(self.G, path, 'weight')
        except nx.NetworkXNoPath:
            if temporarily_removed:
                u, v, data = temporarily_removed
                self.G.add_edge(u, v, **data)
                temporarily_removed = None
                try:
                    path = nx.dijkstra_path(self.G, start_node, target_node, weight='weight')
                    dist = nx.path_weight(self.G, path, 'weight')
                except nx.NetworkXNoPath:
                    pass

        if temporarily_removed:
            u, v, data = temporarily_removed
            self.G.add_edge(u, v, **data)

        return path, dist
    
    def _build_dual_tsp_matrix(self, start_n1, start_n2, target_pairs):
        """
        為 OR-Tools 建立雙機轉移矩陣
        target_pairs: list of ((u1, v1), (u2, v2))
        """
        N = len(target_pairs)
        matrix = np.zeros((N + 1, N + 1), dtype=int)
        SCALE = 1000
        MAX_VAL = 999999999

        # Helper: 單機的 Dijkstra 最短距離
        def get_dist(n_start, n_end):
            try:
                return nx.dijkstra_path_length(self.G, n_start, n_end, weight='weight')
            except nx.NetworkXNoPath:
                return MAX_VAL / SCALE

        # 1. 計算起點 (Index 0) 到各 Pair 起點 (Index 1~N) 的距離
        for j, (e1, e2) in enumerate(target_pairs):
            d1 = get_dist(start_n1, e1[0])
            d2 = get_dist(start_n2, e2[0])
            # 以花費時間最長的機器人為成本 (因為要互相等待)
            matrix[0][j+1] = int(max(d1, d2) * SCALE)

        # 2. 計算各個 Pair 之間的轉移距離
        for i, (from_e1, from_e2) in enumerate(target_pairs):
            for j, (to_e1, to_e2) in enumerate(target_pairs):
                if i == j:
                    matrix[i+1][j+1] = MAX_VAL
                    continue
                
                # 從上一個 Pair 的結束點 (v)，走到下一個 Pair 的起始點 (u)
                d1 = get_dist(from_e1[1], to_e1[0])
                d2 = get_dist(from_e2[1], to_e2[0])
                
                # 同理，取 max 代表整體執行的步數/時間代價
                matrix[i+1][j+1] = int(max(d1, d2) * SCALE)
            
            # 回到起點的距離設為 0 (Open-ended TSP，不強制回起點)
            matrix[i+1][0] = 0

        return matrix.tolist()

    def _solve_dual_tsp_sequence(self, distance_matrix):
        """呼叫 Google OR-Tools 求解雙機全局最短序列"""
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 5  # 給予 5 秒最佳化時間

        print(">> Running OR-Tools Dual-TSP Solver...")
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            return route
        return None

    def _select_essential_pairs(self, reward_map):
        """從 reward_map 中挑選出能覆蓋全局的『最少必要黃金組合』(Set Cover)"""
        print(">> Selecting essential dual-pairs for maximum global coverage...")
        uncovered = set(range(len(self.boundary_pts)))
        essential_pairs = []
        
        # 1. 建立方便查詢的快取，並剔除重複方向的 Pair (e1, e2) 與 (e2, e1)
        pair_coverage_cache = {}
        for (e1, e2), data in reward_map.items():
            fs = frozenset([e1, e2])
            if fs in pair_coverage_cache: continue
            
            s1 = self.dir_edge_states[e1][data['best_idx']]
            s2 = self.dir_edge_states[e2][data['best_idx']]
            combined_v = s1['V'] | s2['V']
            
            pair_coverage_cache[fs] = {
                'e1': e1, 'e2': e2, 'V': combined_v, 'score': data['score']
            }

        # 2. Set Cover 貪婪挑選：每次挑選能看到「最多新東西」的組合
        while uncovered:
            best_pair_key = None
            best_new_cov = 0
            
            for fs, info in pair_coverage_cache.items():
                # 若已經選過則跳過
                if fs in [frozenset([p['e1'], p['e2']]) for p in essential_pairs]:
                    continue
                    
                new_cov = len(info['V'] & uncovered)
                
                # 優先選能看到最多新點的；若數量一樣，選 Synergy 分數較高的
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_pair_key = fs
                elif new_cov > 0 and new_cov == best_new_cov:
                    if info['score'] > pair_coverage_cache[best_pair_key]['score']:
                        best_pair_key = fs
                        
            # 如果剩下的組合都看不到新東西了，就提早結束
            if best_new_cov == 0:
                break
                
            best_info = pair_coverage_cache[best_pair_key]
            essential_pairs.append({'e1': best_info['e1'], 'e2': best_info['e2']})
            uncovered -= best_info['V']
            
        print(f">> Selected {len(essential_pairs)} essential pairs from {len(reward_map)//2} candidates.")
        return essential_pairs
    
    def _build_dual_tsp_matrix(self, start_n1, start_n2, target_pairs):
        """為 OR-Tools 建立雙機轉移成本矩陣 (以走得慢的那台車為成本)"""
        N = len(target_pairs)
        matrix = np.zeros((N + 1, N + 1), dtype=int)
        SCALE = 1000
        MAX_VAL = 999999999

        def get_dist(n_start, n_end):
            if n_start == n_end: return 0
            try: return nx.dijkstra_path_length(self.G, n_start, n_end, weight='weight')
            except nx.NetworkXNoPath: return MAX_VAL / SCALE

        # 1. 計算起點到各 Pair 的距離
        for j, (e1, e2) in enumerate(target_pairs):
            d1 = get_dist(start_n1, e1[0])
            d2 = get_dist(start_n2, e2[0])
            matrix[0][j+1] = int(max(d1, d2) * SCALE) # 取最大值，因為要互相等待

        # 2. 計算 Pair 之間的轉移距離
        for i, (from_e1, from_e2) in enumerate(target_pairs):
            for j, (to_e1, to_e2) in enumerate(target_pairs):
                if i == j:
                    matrix[i+1][j+1] = MAX_VAL
                    continue
                d1 = get_dist(from_e1[1], to_e1[0])
                d2 = get_dist(from_e2[1], to_e2[0])
                matrix[i+1][j+1] = int(max(d1, d2) * SCALE)
            matrix[i+1][0] = 0 # 允許不回起點

        return matrix.tolist()

    def _solve_dual_tsp_sequence(self, distance_matrix):
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 5  

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            return route
        return None

    def plan_dual_trajectory(self, reward_map, start_n1="auto", start_n2="auto"):
        """結合 Set Cover 與 OR-Tools 的終極雙機路由規劃"""
        
        # 1. 精煉目標：選出最少且能涵蓋全局的必要 Pair
        essential_pairs_info = self._select_essential_pairs(reward_map)
        target_pairs = [(info['e1'], info['e2']) for info in essential_pairs_info]
        
        if not target_pairs:
            print(">> Error: No valid synergistic pairs found to plan route.")
            return [], []

        # 2. 決定起點
        if start_n1 == "auto" or start_n2 == "auto":
            start_n1 = target_pairs[0][0][0]
            start_n2 = target_pairs[0][1][0]
            print(f">> Auto Start Selected -> R1: {start_n1}, R2: {start_n2}")

        print(f">> Formulating Dual TSP Sequence with {len(target_pairs)} Essential Pairs...")
        
        # 3. 建立矩陣並呼叫 OR-Tools 求解全局最短序列
        dist_matrix = self._build_dual_tsp_matrix(start_n1, start_n2, target_pairs)
        route_indices = self._solve_dual_tsp_sequence(dist_matrix)

        if not route_indices:
            print(">> OR-Tools TSP Failed. Returning start nodes only.")
            return [start_n1], [start_n2]

        # 4. 根據最佳序列生成實體物理路徑 (防 U-Turn 接軌)
        path1, path2 = [start_n1], [start_n2]
        
        for k in range(len(route_indices) - 1):
            from_idx = route_indices[k]
            to_idx = route_indices[k+1]
            if to_idx == 0: break 
            
            target_e1, target_e2 = target_pairs[to_idx - 1]
            
            # 使用你寫好的無 U-Turn 邏輯前往下一個點
            p1, _ = self._get_path_with_no_uturn(path1[-1], target_e1[0], path1[-2] if len(path1)>1 else None)
            p2, _ = self._get_path_with_no_uturn(path2[-1], target_e2[0], path2[-2] if len(path2)>1 else None)
            
            # 串接路徑 (避開首尾重複)
            if p1 and len(p1) > 1: path1.extend(p1[1:])
            elif not p1 and path1[-1] != target_e1[0]: path1.append(target_e1[0]) # 防呆硬接
                
            if p2 and len(p2) > 1: path2.extend(p2[1:])
            elif not p2 and path2[-1] != target_e2[0]: path2.append(target_e2[0])
            
            # 執行該目標特徵邊的掃描
            if path1[-1] != target_e1[1]: path1.append(target_e1[1])
            if path2[-1] != target_e2[1]: path2.append(target_e2[1])

        print(f">> Dual Global Plan Completed! Total nodes -> R1: {len(path1)}, R2: {len(path2)}")
        return path1, path2

    def calculate_full_path_statistics(self, path1, path2):
        print(">> Calculating comprehensive Full-Path Statistics & Global Coverage...")
        stats = []
        edges1 = [(path1[i], path1[i+1]) for i in range(len(path1)-1)] if len(path1) > 1 else []
        edges2 = [(path2[i], path2[i+1]) for i in range(len(path2)-1)] if len(path2) > 1 else []
        
        max_steps = max(len(edges1), len(edges2))
        total_seen = set() # 新增：用來追蹤全局不重複覆蓋點
        
        for i in range(max_steps):
            e1 = edges1[i] if i < len(edges1) else (path1[-1], path1[-1])
            e2 = edges2[i] if i < len(edges2) else (path2[-1], path2[-1])
            
            states1 = self.dir_edge_states.get(e1, [])
            states2 = self.dir_edge_states.get(e2, [])
            
            # 累加走過這條邊時，所有影格中兩台機器人能看到的所有特徵點
            for s in states1: total_seen.update(s['V'])
            for s in states2: total_seen.update(s['V'])
            
            if e1[0] == e1[1] or e2[0] == e2[1]:
                stats.append({'overlap': 0, 'comp1': 0, 'comp2': 0})
                continue
                
            if not states1 or not states2:
                stats.append({'overlap': 0, 'comp1': 0, 'comp2': 0})
                continue
                
            best_overlap, best_comp1, best_comp2 = 0, 0, 0
            best_score = -1
            
            for s1, s2 in zip(states1, states2):
                ov = len(s1['V'] & s2['V'])
                c1 = len(s2['Shadow'] & s1['V'])
                c2 = len(s1['Shadow'] & s2['V'])
                score = c1 + c2 + (ov * 0.1)
                
                if score > best_score:
                    best_score = score
                    best_overlap, best_comp1, best_comp2 = ov, c1, c2
                    
            stats.append({'overlap': best_overlap, 'comp1': best_comp1, 'comp2': best_comp2})
            
        total_pts = len(self.boundary_pts)
        coverage_percent = (len(total_seen) / total_pts * 100) if total_pts > 0 else 0.0
        
        # 回傳更豐富的字典結構
        return {
            'step_stats': stats,
            'coverage_percent': coverage_percent,
            'max_steps': max_steps,
            'total_seen_count': len(total_seen),
            'total_pts': total_pts
        }

    # ================= 4. Smooth Path Generation =================

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

        
    def save_dual_trajectory(self, path1, path2, filename="dual_trajectory.pkl"):
        save_path = os.path.join(self.output_dir, filename)
        
        r1_waypoints = self._build_smooth_continuous_path(path1)
        r2_waypoints = self._build_smooth_continuous_path(path2)
        
        output_data = {
            'R1_node_sequence': path1,
            'R2_node_sequence': path2,
            'R1_waypoints': r1_waypoints,
            'R2_waypoints': r2_waypoints,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(output_data, f)
            
        print(f">> 軌跡已儲存至: {save_path}")

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

    def visualize_dual_trajectory(self, path1, path2, filename="dual_optimized_trajectory.png"):
        print(">> Generating continuous Dual Trajectory visualization...")
        fig, ax = plt.subplots(figsize=(16, 16), dpi=200)
        ext = [self.org[0], self.org[0] + self.w * self.res, self.org[1], self.org[1] + self.h * self.res]
        ax.imshow(self.flipped_img, cmap='gray', origin='lower', extent=ext, alpha=0.3)

        for u, v, data in self.G.edges(data=True):
            seg = np.array(data.get('smooth_path', [self.G.nodes[u]['pos'], self.G.nodes[v]['pos']]))
            ax.plot(seg[:, 0], seg[:, 1], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

        def plot_robot_path(path_seq, cmap_name, is_dashed, label_prefix):
            if len(path_seq) < 2: return
            cmap = plt.get_cmap(cmap_name)
            num_edges = len(path_seq) - 1

            for i in range(num_edges):
                u, v = path_seq[i], path_seq[i+1]
                
                seg = np.array(self.G[u][v].get('smooth_path', [self.G.nodes[u]['pos'], self.G.nodes[v]['pos']]))
                
                if np.linalg.norm(seg[0] - self.G.nodes[u]['pos']) > 0.5:
                    seg = seg[::-1]

                edge_color = cmap(i / max(1, num_edges))

                if is_dashed:
                    ax.plot(seg[:, 0], seg[:, 1], color=edge_color, 
                            linewidth=1.0, linestyle='--', zorder=4, alpha=1.0)
                else:
                    ax.plot(seg[:, 0], seg[:, 1], color=edge_color, 
                            linewidth=3.0, linestyle='-', zorder=3, alpha=0.3)

            for i in range(len(path_seq) - 1):
                u, v = path_seq[i], path_seq[i+1]
                seg = np.array(self.G[u][v].get('smooth_path', [self.G.nodes[u]['pos'], self.G.nodes[v]['pos']]))
                if np.linalg.norm(seg[0] - self.G.nodes[u]['pos']) > 0.5: seg = seg[::-1]
                if len(seg) >= 2:
                    mid = len(seg) // 2
                    i1, i2 = max(0, mid - 3), min(len(seg) - 1, mid + 3)
                    dx, dy = seg[i2, 0] - seg[i1, 0], seg[i2, 1] - seg[i1, 1]
                    if np.hypot(dx, dy) > 1e-3:
                        color = cmap(i / max(1, len(path_seq) - 2))
                        ax.annotate('', xy=(seg[mid,0]+dx*0.1, seg[mid,1]+dy*0.1), 
                                    xytext=(seg[mid,0]-dx*0.1, seg[mid,1]-dy*0.1),
                                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0), zorder=4)

        plot_robot_path(path1, 'viridis', is_dashed=False, label_prefix='R1')
        plot_robot_path(path2, 'viridis', is_dashed=True, label_prefix='R2')

        ax.set_title("Dual Robot Continuous Coverage Trajectory\nR1 (Solid) | R2 (Dashed)", fontsize=18, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f">> Map saved to: {save_path}")

    def visualize_path_statistics_boxplot(self, full_stats_data, filename="full_path_synergy_boxplot.png"):
        if not full_stats_data or not full_stats_data['step_stats']:
            print(">> No statistics generated. Skipping boxplot.")
            return
            
        full_stats = full_stats_data['step_stats']
        steps = full_stats_data['max_steps']
        cov_pct = full_stats_data['coverage_percent']
        seen = full_stats_data['total_seen_count']
        total = full_stats_data['total_pts']
            
        print(f">> Generating performance Boxplot for Full-Path operations...")
        overlap_data = [s['overlap'] for s in full_stats]
        comp1_data = [s['comp1'] for s in full_stats]
        comp2_data = [s['comp2'] for s in full_stats]

        data = [overlap_data, comp1_data, comp2_data]
        labels = ['Overlap\n(Shared View)', 'Comp 1\n(R1 Helps R2)', 'Comp 2\n(R2 Helps R1)']
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        
        try:
            bplot = ax.boxplot(data, patch_artist=True, tick_labels=labels,
                               boxprops=dict(linewidth=1.5), medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
        except TypeError: # 兼容舊版 Matplotlib
            bplot = ax.boxplot(data, patch_artist=True, labels=labels,
                               boxprops=dict(linewidth=1.5), medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))

        colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        # ================= 動態標題排版：顯示步數與覆蓋率 =================
        title_text = (f"Dual Robot Performance & Synergy Statistics\n"
                      f"Total Steps (Per Robot): {steps} | Global Coverage: {cov_pct:.2f}% ({seen}/{total})")
                      
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Number of Boundary Points", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f">> Statistics chart saved to: {save_path}")

    def visualize_debug_distance_pairs(self, res_comp_pairs, folder_name="debug_distance"):
        """繪製解析度補償的 Debug 圖片，以螢光色突顯一遠一近的補償點"""
        out_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f">> Plotting {len(res_comp_pairs)} Resolution Comp snapshots to: {out_dir}/")
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
    filtered_shadow_pairs = planner.extract_filtered_pairs(reward_map_shadow, min_new_info=15)
    
    if filtered_shadow_pairs:
        planner.visualize_debug_shadow_pairs(filtered_shadow_pairs)
        planner.save_valuable_pairs(filtered_shadow_pairs, folder_name="path_case_shadow")

    print("\n--- Task B: Resolution Compensation ---")
    reward_map_res = planner.get_resolution_comp_pairs(min_overlap=15, min_dist_gap=7.5, min_score=10)
    filtered_res_pairs = planner.extract_filtered_res_comp_pairs(reward_map_res, min_new_gap_pts=10)
    
    if filtered_res_pairs:
        planner.visualize_debug_distance_pairs(filtered_res_pairs, folder_name="debug_distance")
        planner.save_valuable_pairs(filtered_res_pairs, folder_name="path_case_distance")

    # 5. 雙機全局路徑最佳化與後續輸出...
    p1, p2 = planner.plan_dual_trajectory(reward_map_shadow, start_n1="auto", start_n2="auto")
    full_path_stats = planner.calculate_full_path_statistics(p1, p2)
    planner.visualize_dual_trajectory(p1, p2, filename="dual_continuous_path.png")
    planner.visualize_path_statistics_boxplot(full_path_stats, filename="full_path_synergy_stats.png")
    planner.save_dual_trajectory(p1, p2, filename="dual_trajectory.pkl")