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
        
        self.scene_name = os.path.splitext(os.path.basename(png_path))[0]
        self.output_dir = os.path.join("trajectory", self.scene_name, "dual_shortest")
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        if hit_count > 2: return False
        return True

    # ================= 2. Precomputation (Fast Single Edges) =================

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
        print(f">> Precomputing Edge View States using {self.num_workers} cores...")
        self.dir_edge_states.clear()
        edges_list = list(self.G.edges(data=True))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._parallel_calc_edge, edges_list, chunksize=10))
            
        for u, v, states_uv, states_vu in results:
            self.dir_edge_states[(u, v)] = states_uv
            self.dir_edge_states[(v, u)] = states_vu
        print(">> Precomputation completed.")

    # ================= 3. Shortest Routing (Split & Solve) =================

    def _get_path_with_no_uturn(self, start_node, target_node, prev_node):
        temporarily_removed = None
        if prev_node is not None and self.G.has_edge(start_node, prev_node):
            edge_data = self.G.get_edge_data(start_node, prev_node)
            self.G.remove_edge(start_node, prev_node)
            temporarily_removed = (start_node, prev_node, edge_data)

        path = None
        try:
            path = nx.dijkstra_path(self.G, start_node, target_node, weight='weight')
        except nx.NetworkXNoPath:
            if temporarily_removed:
                u, v, data = temporarily_removed
                self.G.add_edge(u, v, **data)
                temporarily_removed = None
                try: path = nx.dijkstra_path(self.G, start_node, target_node, weight='weight')
                except: pass

        if temporarily_removed:
            u, v, data = temporarily_removed
            self.G.add_edge(u, v, **data)

        return path, 0

    def _select_and_split_workload(self, visit_all_edges=False):
        """用 Set Cover 或全圖掃描找出關鍵邊，記錄價值後，以地理中心線剖半分割任務"""
        print(f">> Selecting global essential edges (Mode: {'FULL' if visit_all_edges else 'OPTIMAL'})...")
        
        essential_edges = []
        edge_cov_sizes = {}  # 用來儲存每條邊的覆蓋點數，作為後續懲罰依據
        
        if not visit_all_edges:
            # --- Optimal Mode: 貪婪集合覆蓋與門檻過濾 ---
            edge_cov_union = {}
            for (u, v), states in self.dir_edge_states.items():
                combined_v = set()
                for s in states: combined_v.update(s['V'])
                edge_cov_union[(u, v)] = combined_v
                
            uncovered = set(range(len(self.boundary_pts)))
            selected_phys = set()
            MIN_COV_THRESHOLD = 3  # 覆蓋低於此數值的邊界點視為雜訊
            
            while uncovered:
                best_edge, best_cov = None, 0
                for (u, v), v_set in edge_cov_union.items():
                    fs = frozenset([u, v])
                    if fs in selected_phys: continue
                    
                    new_cov = len(v_set & uncovered)
                    if new_cov > best_cov:
                        best_cov = new_cov
                        best_edge = (u, v)
                        
                if best_cov < MIN_COV_THRESHOLD: break
                
                essential_edges.append(best_edge)
                edge_cov_sizes[best_edge] = best_cov  # 記錄覆蓋貢獻度
                selected_phys.add(frozenset(best_edge))
                uncovered -= edge_cov_union[best_edge]
        else:
            # --- Full Mode: 每條實體道路挑選視角最佳的單向 ---
            selected_phys = set()
            for (u, v), states in self.dir_edge_states.items():
                fs = frozenset([u, v])
                if fs in selected_phys: continue
                
                # 計算雙向的總覆蓋點數
                cov_uv = sum(len(s['V']) for s in self.dir_edge_states.get((u, v), []))
                cov_vu = sum(len(s['V']) for s in self.dir_edge_states.get((v, u), []))
                
                best_edge = (u, v) if cov_uv >= cov_vu else (v, u)
                best_cov = max(cov_uv, cov_vu)
                
                essential_edges.append(best_edge)
                edge_cov_sizes[best_edge] = best_cov
                selected_phys.add(fs)

        print(f">> Selected {len(essential_edges)} essential edges. Splitting workloads...")
        if not essential_edges: return [], [], {}

        # --- 地理切割法 (找尋邊的中心點進行長軸切割) ---
        midpoints = np.array([ (np.array(self.G.nodes[u]['pos']) + np.array(self.G.nodes[v]['pos']))/2 for u, v in essential_edges ])
        dx = np.ptp(midpoints[:, 0])
        dy = np.ptp(midpoints[:, 1])
        
        if dx > dy:
            median_val = np.median(midpoints[:, 0])
            idx1 = midpoints[:, 0] <= median_val
        else:
            median_val = np.median(midpoints[:, 1])
            idx1 = midpoints[:, 1] <= median_val
            
        e1 = [e for i, e in enumerate(essential_edges) if idx1[i]]
        e2 = [e for i, e in enumerate(essential_edges) if not idx1[i]]
        
        if not e1: e1, e2 = e2[:len(e2)//2], e2[len(e2)//2:]
        if not e2: e1, e2 = e1[:len(e1)//2], e1[len(e1)//2:]
        
        print(f">> Workload Split -> Area 1: {len(e1)} edges, Area 2: {len(e2)} edges")
        return e1, e2, edge_cov_sizes

    def _get_best_start(self, edges):
        if not edges: return list(self.G.nodes())[0]
        centroid = np.mean([(np.array(self.G.nodes[u]['pos']) + np.array(self.G.nodes[v]['pos']))/2 for u,v in edges], axis=0)
        best_node, min_dist = None, float('inf')
        for n in self.G.nodes():
            d = np.linalg.norm(np.array(self.G.nodes[n]['pos']) - centroid)
            if d < min_dist:
                min_dist, best_node = d, n
        return best_node

    def _solve_single_tsp(self, start_node, target_edges, edge_cov_sizes, visit_all_edges):
        """帶有懲罰機制的 Prize-Collecting TSP 求解 (針對單一子區域)"""
        if not target_edges: return [start_node]
        
        N = len(target_edges)
        matrix = np.zeros((N + 1, N + 1), dtype=int)
        SCALE, MAX_VAL = 1000, 999999999

        def get_dist(u, v):
            if u == v: return 0
            try: return nx.dijkstra_path_length(self.G, u, v, weight='weight')
            except: return MAX_VAL / SCALE

        # 建立 ATSP 距離矩陣
        for j, (uj, vj) in enumerate(target_edges):
            matrix[0][j+1] = int(get_dist(start_node, uj) * SCALE)

        for i, (ui, vi) in enumerate(target_edges):
            for j, (uj, vj) in enumerate(target_edges):
                if i == j: matrix[i+1][j+1] = MAX_VAL
                else: matrix[i+1][j+1] = int(get_dist(vi, uj) * SCALE)
            matrix[i+1][0] = 0

        dist_matrix = matrix.tolist()
        manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def dist_callback(f_idx, t_idx):
            return dist_matrix[manager.IndexToNode(f_idx)][manager.IndexToNode(t_idx)]

        transit_idx = routing.RegisterTransitCallback(dist_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # --- 核心魔法：加入 Disjunction 懲罰機制 ---
        PENALTY_PER_COVER_POINT = 5000 
        for i in range(1, N + 1):
            edge = target_edges[i-1]
            if visit_all_edges:
                # Full 模式：強迫盡量走完
                penalty = 99999999 if edge_cov_sizes.get(edge, 0) > 0 else 0
            else:
                # Optimal 模式：以價值換算懲罰金
                penalty = edge_cov_sizes.get(edge, 0) * PENALTY_PER_COVER_POINT
                
            routing.AddDisjunction([manager.NodeToIndex(i)], penalty)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 4  # 每台車給予 4 秒最佳化時間

        sol = routing.SolveWithParameters(params)
        if not sol: return [start_node]

        idx = routing.Start(0)
        route_indices = []
        while not routing.IsEnd(idx):
            route_indices.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))

        # 統計捨棄的邊數
        dropped_count = len(target_edges) - (len(route_indices) - 1)
        if dropped_count > 0:
            print(f"   => Smart Agent Dropped {dropped_count} inefficient edge(s).")

        # 重建實體路徑
        path = [start_node]
        for k in range(len(route_indices)-1):
            f_idx, t_idx = route_indices[k], route_indices[k+1]
            if t_idx == 0: break
            
            target_u, target_v = target_edges[t_idx - 1]
            p, _ = self._get_path_with_no_uturn(path[-1], target_u, path[-2] if len(path)>1 else None)
            
            if p and len(p) > 1: path.extend(p[1:])
            elif not p and path[-1] != target_u: path.append(target_u)
            if path[-1] != target_v: path.append(target_v)
            
        return path

    def plan_dual_shortest_trajectory(self, visit_all_edges=False):
        """極致短步數雙車策略主入口 (支援 Optimal/Full)"""
        # 1. 找出最少需要走的關鍵邊，並一分為二，同時取得價值表
        edges1, edges2, edge_cov_sizes = self._select_and_split_workload(visit_all_edges=visit_all_edges)
        
        # 2. 自動決定雙車起點 (降落在各自區域的中心)
        start1 = self._get_best_start(edges1)
        start2 = self._get_best_start(edges2)
        print(f">> Starting R1 at Region Center: {start1}")
        print(f">> Starting R2 at Region Center: {start2}")
        
        # 3. 分別求解各自的最佳 TSP 路徑 (帶入懲罰機制)
        print(">> Solving independent Prize-Collecting TSP for Robot 1...")
        path1 = self._solve_single_tsp(start1, edges1, edge_cov_sizes, visit_all_edges)
        
        print(">> Solving independent Prize-Collecting TSP for Robot 2...")
        path2 = self._solve_single_tsp(start2, edges2, edge_cov_sizes, visit_all_edges)
        
        max_time_steps = max(len(path1), len(path2))
        total_steps = len(path1) + len(path2)
        
        print(f">> Dual Plan Complete!")
        print(f"   - Max Time Steps (Bottleneck): {max_time_steps} (R1: {len(path1)}, R2: {len(path2)})")
        print(f"   - Total Combined Steps: {total_steps}")
        return path1, path2

    # ================= 4. Stats & Smooth Paths =================

    def calculate_full_path_statistics(self, path1, path2):
        print(">> Calculating Global Coverage and Step Statistics...")
        stats = []
        edges1 = [(path1[i], path1[i+1]) for i in range(len(path1)-1)] if len(path1) > 1 else []
        edges2 = [(path2[i], path2[i+1]) for i in range(len(path2)-1)] if len(path2) > 1 else []
        
        max_steps = max(len(edges1), len(edges2))
        print(f'edges1: {len(edges1)}')
        print(f'edges2: {len(edges2)}')
        total_seen = set() 
        
        for i in range(max_steps):
            e1 = edges1[i] if i < len(edges1) else (path1[-1], path1[-1])
            e2 = edges2[i] if i < len(edges2) else (path2[-1], path2[-1])
            
            states1 = self.dir_edge_states.get(e1, [])
            states2 = self.dir_edge_states.get(e2, [])
            
            for s in states1: total_seen.update(s['V'])
            for s in states2: total_seen.update(s['V'])
            
            if e1[0] == e1[1] or e2[0] == e2[1] or not states1 or not states2:
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
        
        return {
            'step_stats': stats,
            'coverage_percent': coverage_percent,
            'max_steps': max_steps,
            'total_seen_count': len(total_seen),
            'total_pts': total_pts
        }

    def visualize_path_statistics_boxplot(self, full_stats_data, filename="dual_shortest_stats.png"):
        if not full_stats_data or not full_stats_data['step_stats']: return
            
        full_stats = full_stats_data['step_stats']
        steps = full_stats_data['max_steps']
        cov_pct = full_stats_data['coverage_percent']
        seen = full_stats_data['total_seen_count']
        total = full_stats_data['total_pts']
            
        print(f">> Generating performance Boxplot...")
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
        except TypeError: 
            bplot = ax.boxplot(data, patch_artist=True, labels=labels,
                               boxprops=dict(linewidth=1.5), medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))

        colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        title_text = (f"Dual Robot Performance (Divide & Conquer)\n"
                      f"Total Execution Steps: {steps} | Global Coverage: {cov_pct:.2f}% ({seen}/{total})")
                      
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Number of Boundary Points", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f">> Statistics chart saved to: {save_path}")

    # ===== Smooth Path & Visualization (保持原樣，僅修改檔名與外觀邏輯) =====
    
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

    def visualize_dual_trajectory(self, path1, path2, filename="dual_shortest_path.png"):
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
                if np.linalg.norm(seg[0] - self.G.nodes[u]['pos']) > 0.5: seg = seg[::-1]
                
                edge_color = cmap(i / max(1, num_edges))
                if is_dashed:
                    ax.plot(seg[:, 0], seg[:, 1], color=edge_color, linewidth=2.0, linestyle='--', zorder=4, alpha=1.0)
                else:
                    ax.plot(seg[:, 0], seg[:, 1], color=edge_color, linewidth=3.0, linestyle='-', zorder=3, alpha=0.7)

            # Draw arrows
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
                                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0), zorder=5)
        
        # 由於是分工合作，用不同顏色區分視覺會更直觀
        plot_robot_path(path1, 'Blues', is_dashed=False, label_prefix='R1')
        plot_robot_path(path2, 'Oranges', is_dashed=False, label_prefix='R2')

        ax.set_title("Dual Robot Split Coverage (Shortest Total Time)\nBlue: Region 1 | Orange: Region 2", fontsize=18, fontweight='bold')
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f">> Map saved to: {save_path}")

    def save_dual_trajectory(self, path1, path2, filename="dual_shortest.pkl"):
        save_path = os.path.join(self.output_dir, filename)
        output_data = {
            'R1_node_sequence': path1,
            'R2_node_sequence': path2,
            'R1_waypoints': self._build_smooth_continuous_path(path1),
            'R2_waypoints': self._build_smooth_continuous_path(path2),
        }
        with open(save_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f">> Trajectory Saved to: {save_path}")

if __name__ == "__main__":
    planner = DualCoveragePlanner(
        png_path='warehouse.png', 
        yaml_path='warehouse.yaml', 
        roadmap_path='trajectory/warehouse/planner/roadmap.pkl',
        max_dist_m=15.0, 
        fov_deg=89.0, 
        boundary_step_m=0.1, 
        obs_interval_m=0.3, 
        min_grazing_deg=5.0
    )
    
    # 1. 前置計算邊的視野
    planner.precompute_directed_edges()
    
    # 2. 自動執行 區域切割 + 雙 TSP 路徑求解
    p1, p2 = planner.plan_dual_shortest_trajectory()
    
    # 3. 輸出包含 "極限總步數" 與 "全局覆蓋率" 的統計資料與圖表
    full_path_stats = planner.calculate_full_path_statistics(p1, p2)
    planner.visualize_path_statistics_boxplot(full_path_stats, filename="dual_shortest_stats.png")
    
    # 4. 輸出視覺化路線圖與 PKL
    planner.visualize_dual_trajectory(p1, p2, filename="dual_shortest_path.png")
    planner.save_dual_trajectory(p1, p2, filename="dual_shortest.pkl")