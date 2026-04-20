import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pickle, cv2, os, networkx as nx, random
import yaml
import itertools
import concurrent.futures
import multiprocessing
from scipy.spatial import KDTree

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
        # --- 基本初始化 ---
        self.scene_name = os.path.splitext(os.path.basename(png_path))[0]
        self.output_dir = os.path.join("trajectory", self.scene_name)
        self.debug_dir = os.path.join(self.output_dir, "debug") 
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
        self.min_grazing_deg = min_grazing_deg
        self.parallel_tol_cos = -np.sin(np.radians(self.min_grazing_deg))
        
        self.boundary_pts = self.extract_boundary_points(step_m=self.boundary_step_m)
        self.target_tree = KDTree(self.boundary_pts)
        self.edge_coverage_cache = {}

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
        contours, _ = cv2.findContours(padded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        all_pts = []
        step_px = max(1, int(step_m / self.res))
        for cnt in contours:
            for i in range(0, len(cnt), step_px):
                c_pad, r_pad = cnt[i][0]
                c, r = c_pad - pad_size, r_pad - pad_size
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

    def _evaluate_path_coverage(self, path):
        """核心邏輯：計算特定方向路徑的覆蓋率"""
        cumulative_dist = np.zeros(len(path))
        for i in range(1, len(path)):
            cumulative_dist[i] = cumulative_dist[i-1] + np.linalg.norm(path[i] - path[i-1])

        total_len = cumulative_dist[-1]
        if total_len == 0:
            sample_pts = np.array([path[0]])
        else:
            num_samples = max(2, int(total_len / self.obs_interval_m))
            target_dists = np.linspace(0, total_len, num_samples)
            
            # 針對 X 和 Y 分別做線性插值
            sample_x = np.interp(target_dists, cumulative_dist, path[:, 0])
            sample_y = np.interp(target_dists, cumulative_dist, path[:, 1])
            sample_pts = np.column_stack((sample_x, sample_y))

        seen_indices = set()
        for i in range(len(sample_pts)-1):
            pos = sample_pts[i]
            r_idx, c_idx = self.w2g(*pos)
            if self.obs_mask[np.clip(r_idx, 0, self.h-1), np.clip(c_idx, 0, self.w-1)] == 1:
                continue 

            delta = sample_pts[i+1] - sample_pts[i]
            heading = np.degrees(np.arctan2(delta[1], delta[0]))
            
            near_idx = self.target_tree.query_ball_point(pos, self.max_dist_m)
            if not near_idx: continue
            
            rel_pts = self.boundary_pts[near_idx] - pos
            angles = np.degrees(np.arctan2(rel_pts[:, 1], rel_pts[:, 0]))
            diff = (angles - heading + 180) % 360 - 180
            in_fov = np.abs(diff) < (self.fov_deg / 2.0)
            
            for orig_idx, is_in_fov in zip(near_idx, in_fov):
                if is_in_fov and orig_idx not in seen_indices:
                    if self._check_visibility(pos, self.boundary_pts[orig_idx]):
                        seen_indices.add(orig_idx)
                        
        return seen_indices

    # ================= 新增：供平行運算使用的 Helper =================
    def _process_single_edge_task(self, task_data):
        """處理單一邊的雙向視角，設計給 ProcessPoolExecutor 呼叫"""
        u, v, path = task_data
        seen_uv = self._evaluate_path_coverage(path)
        seen_vu = self._evaluate_path_coverage(path[::-1])
        return u, v, seen_uv, seen_vu

    def _precompute_all_edges_coverage(self):
        print(">> Precomputing coverage for all edges (Directed & Parallelized)...")
        self.edge_coverage_cache.clear()
        
        tasks = []
        for u, v, data in self.G.edges(data=True):
            path = np.array(data['smooth_path']) if 'smooth_path' in data else \
                   np.array([self.G.nodes[u]['pos'], self.G.nodes[v]['pos']])
            
            # 確保 path 方向是 u -> v
            if np.linalg.norm(path[0] - self.G.nodes[u]['pos']) > 0.5:
                path = path[::-1]

            tasks.append((u, v, path))

        # 使用 CPU 核心數作為 worker 數量
        max_workers = multiprocessing.cpu_count()
        print(f"   => Spawning {max_workers} parallel workers for Raycasting...")

        # 啟動多進程池
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 透過 map 自動把 tasks 分配給多個核心運算
            for u, v, seen_uv, seen_vu in executor.map(self._process_single_edge_task, tasks):
                self.edge_coverage_cache[(u, v)] = seen_uv
                self.edge_coverage_cache[(v, u)] = seen_vu
            
        print(">> Precomputation completed.")

    # ================= 智慧起點選擇演算法 =================
    def _smart_select_start_node(self):
        """
        自動挑選最佳起點：
        優先順序： 1. 偶數分支(Degree % 2 == 0)優先  2. 分支數量最多  3. 連接的 Edge 中位數長度最長(代表空地最大)
        """
        candidates = []
        for node in self.G.nodes():
            degree = self.G.degree(node)
            edges = list(self.G.edges(node, data=True))
            if not edges: continue
            
            # 計算該節點相連的所有 Edge 的長度中位數
            lengths = [data.get('weight', 0) for _, _, data in edges]
            median_length = np.median(lengths)
            
            candidates.append({
                'node': node,
                'degree': degree,
                'is_even': (degree % 2 == 0),
                'median_len': median_length
            })
            
        if not candidates:
            return list(self.G.nodes())[0] # 防呆機制
            
        # 進行多條件排序 (True 在前, 數值大的在前)
        candidates.sort(key=lambda x: (x['is_even'], x['degree'], x['median_len']), reverse=True)
        
        best = candidates[0]
        print(f">> Smart Start Selected -> Node: {best['node']} | Degree: {best['degree']} (Even: {best['is_even']}) | Median Edge Length: {best['median_len']:.2f}m")
        return best['node']

    def plan_trajectory(self, start_node="auto", visit_all_edges=False):
        """
        統一的路由入口，支援 "auto" 智慧起點與手動指定起點。
        """
        # --- 處理起點選擇 ---
        if start_node == "auto" or start_node is None:
            actual_start = self._smart_select_start_node()
        elif start_node not in self.G:
            print(f">> Warning: Manual start node {start_node} not found in Roadmap! Falling back to 'auto'.")
            actual_start = self._smart_select_start_node()
        else:
            actual_start = start_node
            print(f">> Manual Start Selected -> Node: {actual_start}")

        if visit_all_edges:
            print(f">> Mode: Full Coverage (Strict No U-Turn & Route Cleaning). Start: {actual_start}")
            return self._plan_routing(actual_start, visit_all_edges=True)
        else:
            print(f">> Mode: Optimized Coverage (Strict No U-Turn & Route Cleaning). Start: {actual_start}")
            return self._plan_routing(actual_start, visit_all_edges=False)

    def _build_atsp_matrix_and_paths(self, start_node, required_edges):
        """建構非對稱 TSP 的距離矩陣，同時記錄點到點的具體路徑"""
        N = len(required_edges)
        matrix = np.zeros((N + 1, N + 1), dtype=int)
        paths = {} 
        SCALE = 1000  
        MAX_VAL = 999999999  

        def get_dijkstra(source, excluded_edge=None):
            if excluded_edge and self.G.has_edge(*excluded_edge):
                u, v = excluded_edge
                
                # 【修復 1】防呆機制：檢查是否為死胡同。
                # 如果該節點的連接數 > 1，代表有其他路可走，才允許「禁止倒車」
                degree = self.G.out_degree(u) if self.G.is_directed() else self.G.degree(u)
                if degree > 1:
                    data = self.G.get_edge_data(u, v)
                    self.G.remove_edge(u, v)
                    try:
                        l, p = nx.single_source_dijkstra(self.G, source, weight='weight')
                    except Exception:
                        l, p = {}, {}
                    self.G.add_edge(u, v, **data)
                    return l, p
            
            # 如果是死胡同，或是沒有排除條件，就跑正常的最短路徑
            try:
                l, p = nx.single_source_dijkstra(self.G, source, weight='weight')
            except Exception:
                l, p = {}, {}
            return l, p

        # 1. 計算從 start_node 出發到各必經 Edge 起點 (uj) 的距離
        l_start, p_start = get_dijkstra(start_node)
        for j, (uj, vj) in enumerate(required_edges):
            if uj in l_start:
                matrix[0][j+1] = int(l_start[uj] * SCALE)
                paths[(0, j+1)] = p_start[uj]
            else:
                matrix[0][j+1] = MAX_VAL
                paths[(0, j+1)] = []

        # 2. 計算各個必經 Edge 之間的轉移距離 (從 vi 走到 uj)
        for i, (ui, vi) in enumerate(required_edges):
            l_vi, p_vi = get_dijkstra(vi, excluded_edge=(vi, ui))
            
            for j, (uj, vj) in enumerate(required_edges):
                if i == j:
                    matrix[i+1][j+1] = MAX_VAL  
                    continue
                if uj in l_vi:
                    matrix[i+1][j+1] = int(l_vi[uj] * SCALE)
                    paths[(i+1, j+1)] = p_vi[uj]
                else:
                    matrix[i+1][j+1] = MAX_VAL
                    paths[(i+1, j+1)] = []
            
            matrix[i+1][0] = 0
            
        return matrix.tolist(), paths

    def _solve_tsp_sequence(self, distance_matrix, penalties=None):
        """呼叫 OR-Tools 求解帶有懲罰機制的全局最短 TSP 序列 (Prize-Collecting TSP)"""
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp

        num_nodes = len(distance_matrix)
        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 核心魔法：加入 Disjunction (允許放棄節點)
        if penalties:
            # node 0 是起點，不能放棄，所以從 1 開始
            for i in range(1, num_nodes):
                penalty_value = int(penalties[i])
                # AddDisjunction 允許求解器不經過該節點，但必須付出 penalty_value 的成本代價
                routing.AddDisjunction([manager.NodeToIndex(i)], penalty_value)

        # 設定進階 Metaheuristic 演算法
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 5  # 給予 5 秒的深度思考時間

        print(">> Running OR-Tools Prize-Collecting TSP Solver...")
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            return route
        return None

    def _plan_routing(self, start_node, visit_all_edges):
        """全新的 Prize-Collecting 路由：考量空間成本，允許果斷放棄低效益邊緣點"""
        
        if not self.edge_coverage_cache:
            self._precompute_all_edges_coverage()

        required_edges = []
        penalties = [0]  # index 0 是起點，懲罰為 0 (因為設為 Depot 不會被 drop)

        # 【參數調校區】
        # SCALE = 1000 (1公尺 = 1000成本)。
        # 如果 1 個覆蓋點值得機器人多走 5 公尺，則 PENALTY = 5 * 1000 = 5000
        # 你可以根據實際場地大小與想要覆蓋的細緻度來調整這個權重。
        PENALTY_PER_COVER_POINT = 5000 
        MIN_COV_THRESHOLD = 3  # 覆蓋點數低於 3 的直接視為雜訊，不納入規劃

        # --- 階段一：挑選出有潛力的目標 Edge 並計算它們的專屬懲罰值 ---
        if not visit_all_edges:
            uncovered = set(range(len(self.boundary_pts)))
            
            while uncovered:
                best_edge, best_cov_size = None, 0
                for edge_tuple, cov in self.edge_coverage_cache.items():
                    u, v = edge_tuple
                    # 如果這條邊的任何一個方向已經被選過了，就跳過
                    if (u, v) in required_edges or (v, u) in required_edges: 
                        continue
                    
                    new_cov = len(cov & uncovered)
                    if new_cov > best_cov_size:
                        best_cov_size = new_cov
                        best_edge = edge_tuple
                
                # 若剩下的邊能提供的額外覆蓋極低，直接停止挑選，拒絕為了一兩點跑全圖
                if best_cov_size < MIN_COV_THRESHOLD: 
                    break 
                    
                required_edges.append(best_edge)
                # 將覆蓋效益轉換為放棄該邊的「懲罰金」
                penalties.append(best_cov_size * PENALTY_PER_COVER_POINT)
                uncovered -= self.edge_coverage_cache[best_edge]
                
            print(f">> Optimal Mode: Selected {len(required_edges)} candidate edges with localized penalties.")
            
        else:
            # Full Mode: 每條實體邊只走一次
            for u, v in self.G.edges():
                cov_uv = len(self.edge_coverage_cache.get((u, v), set()))
                cov_vu = len(self.edge_coverage_cache.get((v, u), set()))
                
                if cov_uv >= cov_vu:
                    chosen_edge = (u, v)
                    cov_size = cov_uv
                else:
                    chosen_edge = (v, u)
                    cov_size = cov_vu
                    
                required_edges.append(chosen_edge)
                # 全覆蓋模式下賦予極高懲罰 (99999999)，強迫求解器盡可能走完，除非真的無路可走
                huge_penalty = 99999999 if cov_size > 0 else 0 
                penalties.append(huge_penalty)
                
            print(f">> Full Mode: Selected {len(required_edges)} directed edges.")

        if not required_edges:
            return [start_node]

        # --- 階段二：建立 TSP 矩陣並求解 ---
        dist_matrix, path_dict = self._build_atsp_matrix_and_paths(start_node, required_edges)
        route_indices = self._solve_tsp_sequence(dist_matrix, penalties)

        if not route_indices:
            print(">> Error: TSP Solver failed to find a valid route! Returning start node only.")
            return [start_node]

        # --- 階段三：根據最佳化序列，重組為連續的物理路徑 ---
        final_path = []
        
        for k in range(len(route_indices) - 1):
            from_idx = route_indices[k]
            to_idx = route_indices[k+1]
            
            if to_idx == 0:
                break
                
            trans_path = path_dict.get((from_idx, to_idx), [])
            
            # 強制備用接軌
            if not trans_path:
                from_node = start_node if from_idx == 0 else required_edges[from_idx - 1][1]
                to_node = required_edges[to_idx - 1][0]
                try:
                    _, fallback_path = nx.single_source_dijkstra(self.G, from_node, to_node, weight='weight')
                    trans_path = fallback_path
                except Exception:
                    print(f">> Critical Error: Graph completely disconnected between {from_node} and {to_node}.")
                    break
                    
            # 串接路徑 (避免首尾節點重複)
            if final_path and final_path[-1] == trans_path[0]:
                final_path.extend(trans_path[1:])
            else:
                final_path.extend(trans_path)
                
            # 執行必須走訪的目標 Edge (uj -> vj)
            to_edge_idx = to_idx - 1
            uj, vj = required_edges[to_edge_idx]
            
            if final_path[-1] != vj:
                final_path.append(vj)

        # 統計並印出被演算法判定為「不符成本效益」而捨棄的邊緣點數量
        dropped_count = len(required_edges) - (len(route_indices) - 1)
        if dropped_count > 0:
            print(f">> Smart Optimization: Dropped {dropped_count} inefficient edge(s) to save travel cost.")

        if not final_path:
            return [start_node]

        print(f">> Global Plan completed! Total nodes in sequence: {len(final_path)}")
        return final_path

    # ================= 平滑路徑生成與倒角計算邏輯 =================
    
    def _check_collision_w(self, pts):
        for pt in pts:
            r, c = self.w2g(pt[0], pt[1])
            if 0 <= r < self.h and 0 <= c < self.w:
                if self.obs_mask[r, c] == 1: return True
            else:
                return True
        return False

    def _get_fillet_info(self, la, lb, node_pos, radius):
        def find_idx_in(line, d):
            cur = 0
            for i in range(len(line)-1, 0, -1):
                cur += np.linalg.norm(line[i] - line[i-1])
                if cur >= d: return i
            return 0
            
        def find_idx_out(line, d):
            cur = 0
            for i in range(len(line)-1):
                cur += np.linalg.norm(line[i+1] - line[i])
                if cur >= d: return i
            return len(line)-1
            
        idx_in = find_idx_in(la, radius)
        idx_out = find_idx_out(lb, radius)
        p1, p2 = la[idx_in], lb[idx_out]
        
        t = np.linspace(0, 1, 25)
        curve = ((1-t)**2)[:, None] * p1 + (2*(1-t)*t)[:, None] * node_pos + (t**2)[:, None] * p2
        return curve, idx_in, idx_out

    def _build_smooth_continuous_path(self, path_sequence, max_fillet_r=0.9):
        if len(path_sequence) < 2: return []
            
        segments = []
        for i in range(len(path_sequence) - 1):
            u, v = path_sequence[i], path_sequence[i+1]
            edge_data = self.G[u][v]
            if 'smooth_path' in edge_data:
                seg = np.array(edge_data['smooth_path'])
                if np.linalg.norm(seg[0] - self.G.nodes[u]['pos']) > 0.5:
                    seg = seg[::-1]
            else:
                seg = np.array([self.G.nodes[u]['pos'], self.G.nodes[v]['pos']])
            segments.append(seg)
            
        if len(segments) == 1: return segments[0].tolist()
            
        final_path = []
        current_seg = segments[0]
        
        for i in range(len(segments) - 1):
            node_b = path_sequence[i+1]
            node_pos = np.array(self.G.nodes[node_b]['pos'])
            next_seg = segments[i+1]
            
            if path_sequence[i] == path_sequence[i+2]:
                final_path.extend(current_seg.tolist())
                current_seg = next_seg
                continue
            
            r = max_fillet_r
            valid_curve = None
            best_idx_in = len(current_seg) - 1
            best_idx_out = 0
            
            while r >= 0.15:
                curve, idx_in, idx_out = self._get_fillet_info(current_seg, next_seg, node_pos, r)
                if not self._check_collision_w(curve):
                    valid_curve = curve
                    best_idx_in, best_idx_out = idx_in, idx_out
                    break
                r -= 0.1
                
            if valid_curve is not None:
                final_path.extend(current_seg[:best_idx_in].tolist())
                final_path.extend(valid_curve.tolist())
            else:
                final_path.extend(current_seg.tolist()) 
                
            current_seg = next_seg[best_idx_out:]
            
        final_path.extend(current_seg.tolist())
        return final_path

    # ================= 匯出與繪圖 =================

    def save_trajectory(self, path_sequence, filename="trajectory.pkl"):
        single_dir = os.path.join(self.output_dir, "single")
        os.makedirs(single_dir, exist_ok=True)
        save_path = os.path.join(single_dir, filename)

        full_waypoints = self._build_smooth_continuous_path(path_sequence)

        output_data = {
            'node_sequence': path_sequence,
            'waypoint_path': full_waypoints,
            'total_waypoints': len(full_waypoints)
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(output_data, f)
            
        print(f">> 軌跡資料已成功匯出至: {save_path} (共 {len(full_waypoints)} 個平滑無縫航點)")

    def visualize_trajectory(self, path_sequence, filename="optimized_trajectory.png"):
        if not self.edge_coverage_cache:
            self._precompute_all_edges_coverage()

        single_dir = os.path.join(self.output_dir, "single")
        os.makedirs(single_dir, exist_ok=True)
        save_path = os.path.join(single_dir, filename)

        print(f">> Generating advanced continuous trajectory visualization...")
        fig, ax = plt.subplots(figsize=(14, 14))
        
        ext = [self.org[0], self.org[0] + self.w * self.res, self.org[1], self.org[1] + self.h * self.res]
        ax.imshow(self.flipped_img, cmap='gray', origin='lower', extent=ext, alpha=0.3)

        # 背景 Roadmap
        for u, v, data in self.G.edges(data=True):
            if 'smooth_path' in data:
                segment = np.array(data['smooth_path'])
                ax.plot(segment[:, 0], segment[:, 1], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            else:
                p1 = self.G.nodes[u]['pos']; p2 = self.G.nodes[v]['pos']
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # 覆蓋率計算與繪圖
        covered_indices = set()
        for i in range(len(path_sequence) - 1):
            u = path_sequence[i]; v = path_sequence[i+1]
            if (u, v) in self.edge_coverage_cache:
                covered_indices.update(self.edge_coverage_cache[(u, v)])

        total_pts = len(self.boundary_pts)
        coverage_percent = (len(covered_indices) / total_pts) * 100 if total_pts > 0 else 0.0

        all_idx = set(range(total_pts))
        uncovered_idx = list(all_idx - covered_indices)
        covered_idx = list(covered_indices)
        
        if uncovered_idx:
            ax.scatter(self.boundary_pts[uncovered_idx, 0], self.boundary_pts[uncovered_idx, 1], 
                       c='red', s=1, alpha=0.8, label='Uncovered', zorder=2)
        if covered_idx:
            ax.scatter(self.boundary_pts[covered_idx, 0], self.boundary_pts[covered_idx, 1], 
                       c='lime', s=1, alpha=0.8, label='Covered', zorder=2)

        # 畫無縫漸層軌跡 (LineCollection)
        full_path_pts = np.array(self._build_smooth_continuous_path(path_sequence))
        
        cmap = plt.get_cmap('inferno')
        if len(full_path_pts) > 1:
            points = full_path_pts.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, len(segments))
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.85, zorder=3)
            lc.set_array(np.arange(len(segments)))
            ax.add_collection(lc)

        # 完美貼合 Smooth Path 的箭頭
        for i in range(len(path_sequence) - 1):
            u = path_sequence[i]
            v = path_sequence[i+1]
            
            edge_data = self.G[u][v]
            if 'smooth_path' in edge_data:
                segment = np.array(edge_data['smooth_path'])
                if np.linalg.norm(segment[0] - self.G.nodes[u]['pos']) > 0.5:
                    segment = segment[::-1]
            else:
                segment = np.array([self.G.nodes[u]['pos'], self.G.nodes[v]['pos']])
            
            if len(segment) >= 2:
                mid_idx = len(segment) // 2
                idx1 = max(0, mid_idx - 3)
                idx2 = min(len(segment) - 1, mid_idx + 3)
                p1 = segment[idx1]
                p2 = segment[idx2]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                if np.hypot(dx, dy) > 1e-3:
                    mid_x = segment[mid_idx][0]
                    mid_y = segment[mid_idx][1]
                    color = cmap(i / max(1, len(path_sequence) - 2))
                    
                    ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                                xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                                arrowprops=dict(arrowstyle="->", color=color, lw=2.5), zorder=6)

        # 起終點標示
        if len(full_path_pts) > 0:
            ax.scatter(full_path_pts[0, 0], full_path_pts[0, 1], c='c', marker='*', s=150, edgecolors='black', zorder=7, label='Start')
            ax.scatter(full_path_pts[-1, 0], full_path_pts[-1, 1], c='r', marker='s', s=60, edgecolors='black', zorder=7, label='End')

        title_text = (f"Continuous Coverage Trajectory\n"
                      f"Total Coverage: {coverage_percent:.2f}% ({len(covered_indices)}/{total_pts})\n"
                      f"Edges Traversed: {len(path_sequence)-1}")
        ax.set_title(title_text, fontsize=16, fontweight='bold')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Continuous Progression (Start -> End)', rotation=270, labelpad=15)

        ax.legend(loc='upper right')
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f">> 成功！完美平滑連續軌跡圖已儲存至: {save_path}")

if __name__ == "__main__":
    solver = CoverageSolver(
        png_path='../cropped_maps/hospital.png', 
        yaml_path='../cropped_maps/hospital.yaml', 
        roadmap_path='trajectory/hospital/planner/roadmap.pkl',
        max_dist_m=20.0,           
        fov_deg=89.0,              
        boundary_step_m=0.1,       
        obs_interval_m=0.3,        
        min_grazing_deg=5.0       
    )
    
    print("\n--- Running Optimized Coverage Mode ---")
    optimized_path = solver.plan_trajectory(start_node="auto", visit_all_edges=False)
    solver.visualize_trajectory(optimized_path, filename="optimized_path.png")
    solver.save_trajectory(optimized_path, filename="roadmap_single_optimal.pkl")
    
    print("\n--- Running Full Coverage Mode ---")
    full_path = solver.plan_trajectory(start_node="auto", visit_all_edges=True)
    solver.visualize_trajectory(full_path, filename="full_coverage_path.png")
    solver.save_trajectory(full_path, filename="roadmap_single_full.pkl")