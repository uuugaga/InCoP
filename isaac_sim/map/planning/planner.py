import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml, cv2, sys, os, heapq, pickle
from skimage.morphology import skeletonize
import networkx as nx
from scipy.spatial import KDTree
from scipy import interpolate
from scipy.ndimage import distance_transform_edt

sys.setrecursionlimit(100000)

class ElasticBandPlanner:
    def __init__(self, 
                 png_path, 
                 yaml_path, 
                 inflation_m=0.35, 
                 fillet_r=0.9,
                 max_dist_factor=8.0, 
                 clearance_factor=1.2,
                 prune_dist_m=2.5,
                 eb_iterations=300,
                 eb_alpha=0.2,
                 eb_beta=0.1,
                 visibility_max_dist_m=20.0):
        
        # Define output directory: ./trajectory/{scene_name}/
        scene_name = os.path.splitext(os.path.basename(png_path))[0]
        self.output_dir = os.path.join(os.path.join("trajectory", scene_name), "planner")
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)

        # Store parameters
        self.inflation_m = inflation_m
        self.fillet_r = fillet_r
        self.max_dist_factor = max_dist_factor
        self.clearance_factor = clearance_factor
        self.prune_dist_m = prune_dist_m
        self.eb_iterations = eb_iterations
        self.eb_alpha = eb_alpha
        self.eb_beta = eb_beta
        self.visibility_dist = visibility_max_dist_m

        # 1. Load Map and Metadata
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.res, self.org = cfg['resolution'], cfg['origin']
        
        raw_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        self.flipped_img = cv2.flip(raw_img, 0)
        self.h, self.w = self.flipped_img.shape[:2]
        
        # 2. Collision Map and Potential Field Preparation
        obs_mask = (self.flipped_img < 200).astype(np.uint8)
        px_radius = int(self.inflation_m / self.res)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px_radius*2+1, px_radius*2+1))
        self.collision_map = cv2.dilate(obs_mask, kernel, iterations=1)
        
        free_space = (self.collision_map == 0).astype(np.uint8)
        self.dist_to_obs = distance_transform_edt(free_space)
        self.grad_y, self.grad_x = np.gradient(self.dist_to_obs)
        
        # 3. Execution Pipeline with English Logging
        self.skeleton = skeletonize(free_space.astype(bool)).astype(np.uint8)
        self.G = nx.Graph()
        
        print(f">> Initializing planner for scene: {scene_name}")
        print(">> STEP 1: Extracting raw topology from skeleton...")
        self.build_initial_graph_no_filter()
        self.save_debug_plot("01_raw_skeleton.png", "Raw Skeleton Topology")
        
        print(f">> STEP 2: Iterative factor merging (Initial nodes: {len(self.G.nodes)})...")
        self.iterative_clearance_merge()
        self.save_debug_plot("02_iterative_merged.png", "Topological Node Merging")
        
        print(">> STEP 3: Pruning isolated components and dead-ends...")
        self.prune_islands()
        self.prune_tentacles()
        self.shrink_dead_ends(1.0)
        self.save_debug_plot("03_pruned_topology.png", "Cleaned Graph Structure")
        
        print(">> STEP 4: Enhancing connectivity via Visibility Check...")
        self.enhance_connectivity_visibility()
        self.save_debug_plot("04_visibility_enhanced.png", "Visibility Shortcut Connections")
        
        print(">> STEP 5: Planning A* shortest path (Racing Lines)...")
        self.optimize_all_edges_shortest()
        self.save_debug_plot("05_astar_optimized.png", "A* Geometric Optimization")
        
        print(f">> STEP 6: Running Elastic Band fluid optimization ({self.eb_iterations} iterations)...")
        self.apply_elastic_band_optimization()
        self.save_debug_plot("06_elastic_band_optimized.png", "Elastic Band Fluid Smoothing")
        
        print(">> STEP 7: Finalizing results and exporting Roadmap...")
        self.save_results()

    def g2w(self, r, c): return [self.org[0] + c * self.res, self.org[1] + r * self.res]
    def w2g(self, x, y): return int((y - self.org[1]) / self.res), int((x - self.org[0]) / self.res)

    def _get_neighbors(self, r, c, skel_pts):
        pts = []
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in skel_pts: pts.append((nr, nc))
        return pts

    def build_initial_graph_no_filter(self):
        rows, cols = np.where(self.skeleton > 0)
        skel_pts = set(zip(rows, cols))
        nodes = [p for p in skel_pts if len(self._get_neighbors(p[0], p[1], skel_pts)) != 2]
        for p in nodes: self.G.add_node(p, pos=self.g2w(*p))
        processed = set()
        for node in nodes:
            for nbr in self._get_neighbors(node[0], node[1], skel_pts):
                edge_key = tuple(sorted((node, nbr)))
                if edge_key in processed: continue
                path = [node, nbr]; curr, prev = nbr, node
                processed.add(edge_key)
                while curr not in self.G and curr in skel_pts:
                    nxt = [n for n in self._get_neighbors(curr[0], curr[1], skel_pts) if n != prev]
                    if not nxt: break
                    prev, curr = curr, nxt[0]; path.append(curr)
                if curr in self.G and node != curr:
                    self.G.add_edge(node, curr, path=path)

    def iterative_clearance_merge(self):
        limit = self.inflation_m * self.max_dist_factor
        while True:
            changed = False
            for u, v in list(self.G.edges()):
                self.G[u][v]['weight'] = np.linalg.norm(np.array(self.G.nodes[u]['pos']) - np.array(self.G.nodes[v]['pos']))
            edges = sorted(list(self.G.edges(data=True)), key=lambda x: x[2]['weight'])
            for u, v, data in edges:
                if u not in self.G or v not in self.G: continue
                dist_m = data['weight']
                if dist_m > limit: continue
                r_px = (dist_m * self.clearance_factor) / (2 * self.res)
                p1, p2 = np.array(u), np.array(v); center = tuple(((p1 + p2) / 2).astype(int))
                mask = np.zeros((self.h, self.w), dtype=np.uint8); cv2.circle(mask, (center[1], center[0]), int(r_px), 1, -1)
                if not np.any(np.logical_and(mask == 1, self.collision_map == 1)):
                    for node in [u, v]:
                        for nbr in list(self.G.neighbors(node)):
                            if nbr != u and nbr != v:
                                p = self.G[node][nbr]['path']
                                if np.linalg.norm(np.array(p[0]) - np.array(node)) < 1.0: p[0] = center
                                else: p[-1] = center
                    self.G = nx.contracted_nodes(self.G, u, v, self_loops=False)
                    self.G.nodes[u]['pos'] = self.g2w(*center); nx.relabel_nodes(self.G, {u: center}, copy=False)
                    changed = True; break 
            if not changed: break
        print(f">> Merging completed. Final node count: {len(self.G.nodes)}")

    def enhance_connectivity_visibility(self):
        node_list = list(self.G.nodes())
        pos_array = np.array([self.G.nodes[n]['pos'] for n in node_list])
        tree = KDTree(pos_array)
        added_count = 0
        for i, u_node in enumerate(node_list):
            u_pos = np.array(self.G.nodes[u_node]['pos'])
            indices = tree.query_ball_point(u_pos, self.visibility_dist)
            for idx in indices:
                v_node = node_list[idx]
                if u_node == v_node or self.G.has_edge(u_node, v_node): continue
                v_pos = np.array(self.G.nodes[v_node]['pos'])
                dist = np.linalg.norm(u_pos - v_pos)
                num_samples = int(dist / (self.res * 0.5))
                lpts_x, lpts_y = np.linspace(u_pos[0], v_pos[0], num_samples), np.linspace(u_pos[1], v_pos[1], num_samples)
                is_clear = True
                for lx, ly in zip(lpts_x, lpts_y):
                    r, c = self.w2g(lx, ly)
                    if not (0 <= r < self.h and 0 <= c < self.w and self.collision_map[r, c] == 0):
                        is_clear = False; break
                if is_clear:
                    center_px = tuple(((np.array(u_node) + np.array(v_node)) / 2).astype(int))
                    mask = np.zeros((self.h, self.w), dtype=np.uint8); cv2.circle(mask, (center_px[1], center_px[0]), int(dist / (2 * self.res)), 1, -1)
                    if not np.any(np.logical_and(mask == 1, self.collision_map == 1)):
                        r_line = np.linspace(u_node[0], v_node[0], int(dist/self.res)).astype(int)
                        c_line = np.linspace(u_node[1], v_node[1], int(dist/self.res)).astype(int)
                        self.G.add_edge(u_node, v_node, path=list(zip(r_line, c_line))); added_count += 1
        print(f">> Visibility enhancement: Added {added_count} connections.")

    def a_star_shortest(self, start, end):
        def h(a, b): return np.linalg.norm(np.array(a) - np.array(b))
        def find_free(p):
            if self.collision_map[p[0], p[1]] == 0: return p
            for r in range(1, 15):
                for dr in range(-r, r+1):
                    for dc in range(-r, r+1):
                        nr, nc = p[0]+dr, p[1]+dc
                        if 0 <= nr < self.h and 0 <= nc < self.w and self.collision_map[nr, nc] == 0: return (nr, nc)
            return p
        s, e = find_free(start), find_free(end)
        close, came, g, o = set(), {}, {s: 0}, [(h(s, e), s)]
        while o:
            curr = heapq.heappop(o)[1]
            if curr == e:
                path = []
                while curr in came: path.append(curr); curr = came[curr]
                path.append(s); return path[::-1]
            close.add(curr)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                n = (curr[0]+dr, curr[1]+dc)
                if 0 <= n[0] < self.h and 0 <= n[1] < self.w and self.collision_map[n[0], n[1]] == 0:
                    cost = g[curr] + (1.414 if abs(dr)+abs(dc)==2 else 1.0)
                    if n in close and cost >= g.get(n, 0): continue
                    if cost < g.get(n, float('inf')):
                        came[n], g[n] = curr, cost; heapq.heappush(o, (cost+h(n, e), n))
        return [start, end]

    def optimize_all_edges_shortest(self):
        for u, v in self.G.edges(): self.G[u][v]['path'] = self.a_star_shortest(u, v)

    def apply_elastic_band_optimization(self):
        for u, v, data in self.G.edges(data=True):
            raw_pts = np.array(data['path']).astype(float)
            if len(raw_pts) < 5:
                data['smooth_path'] = np.array([self.g2w(r, c) for r, c in raw_pts])
                continue
            dists = np.sqrt(np.sum(np.diff(raw_pts, axis=0)**2, axis=1))
            cum_dist = np.insert(np.cumsum(dists), 0, 0)
            interp = interpolate.interp1d(cum_dist, raw_pts, axis=0)
            band = interp(np.linspace(0, cum_dist[-1], max(int(cum_dist[-1]), 5)))
            for _ in range(self.eb_iterations):
                internal = np.zeros_like(band); internal[1:-1] = band[:-2] + band[2:] - 2 * band[1:-1]
                rows = np.clip(band[1:-1, 0].astype(int), 0, self.h - 1)
                cols = np.clip(band[1:-1, 1].astype(int), 0, self.w - 1)
                dist_vals = self.dist_to_obs[rows, cols] + 0.1
             
                force_mag = np.clip(1.0 / (dist_vals ** 2), 0.0, 2.0) 
                external = np.zeros_like(band)
                external[1:-1, 1] = self.grad_x[rows, cols] * force_mag
                external[1:-1, 0] = self.grad_y[rows, cols] * force_mag
                step_update = self.eb_alpha * internal + self.eb_beta * external
                step_update = np.clip(step_update, -1.0, 1.0) 
                band += step_update
            data['smooth_path'] = np.array([self.g2w(r, c) for r, c in band])

    def prune_islands(self):
        if self.G.number_of_nodes() == 0: return
        largest_cc = max(nx.connected_components(self.G), key=len)
        self.G.remove_nodes_from([n for n in self.G.nodes if n not in largest_cc])

    def prune_tentacles(self):
        while True:
            to_rm = [n for n in self.G.nodes() if self.G.degree(n) == 1 and np.linalg.norm(np.array(self.G.nodes[n]['pos']) - np.array(self.G.nodes[list(self.G.neighbors(n))[0]]['pos'])) < self.prune_dist_m]
            if not to_rm: break
            self.G.remove_nodes_from(to_rm)

    def shrink_dead_ends(self, shrink_dist_m=1.5):

        if self.G.number_of_nodes() == 0: return
        shrink_px = shrink_dist_m / self.res
        mapping = {}
        
        for n in list(self.G.nodes()):
            if self.G.degree(n) == 1:
                nbr = list(self.G.neighbors(n))[0]
                path = self.G[n][nbr]['path']
                
                if np.linalg.norm(np.array(path[0]) - np.array(n)) > np.linalg.norm(np.array(path[-1]) - np.array(n)):
                    path = path[::-1]
                
                dist_accum = 0.0
                new_n = n
                for i in range(len(path) - 1):
                    dist_accum += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                    if dist_accum >= shrink_px:
                        new_n = tuple(path[i+1])
                        self.G[n][nbr]['path'] = path[i+1:] # 截斷舊路徑
                        break
                
                if new_n != n:
                    self.G.nodes[n]['pos'] = self.g2w(*new_n)
                    mapping[n] = new_n
        
        if mapping:
            nx.relabel_nodes(self.G, mapping, copy=False)
            print(f">> Shrunk {len(mapping)} dead-end edges by {shrink_dist_m}m.")

    def check_collision(self, pts):
        for pt in pts:
            r, c = self.w2g(pt[0], pt[1])
            if not (0 <= r < self.h and 0 <= c < self.w and self.collision_map[r, c] == 0): return True
        return False

    def get_filleted_turn(self, edge_a, edge_b, node, radius):
        pos = self.G.nodes[node]['pos']
        la = edge_a if np.linalg.norm(edge_a[0] - pos) < 0.1 else edge_a[::-1]
        lb = edge_b if np.linalg.norm(edge_b[0] - pos) < 0.1 else edge_b[::-1]
        def find_p(line, d):
            cur = 0
            for i in range(len(line)-1):
                cur += np.linalg.norm(line[i+1]-line[i])
                if cur >= d: return line[i+1]
            return line[-1]
        p1, p2, t = find_p(la, radius), find_p(lb, radius), np.linspace(0, 1, 25)
        return ((1-t)**2)[:, None] * p1 + (2*(1-t)*t)[:, None] * pos + (t**2)[:, None] * p2

    def save_debug_plot(self, filename, title):
        plt.figure(figsize=(10, 15))
        ext = [self.org[0], self.org[0]+self.w*self.res, self.org[1], self.org[1]+self.h*self.res]
        plt.imshow(self.flipped_img, cmap='gray', origin='lower', extent=ext, alpha=0.3)
        for u, v, data in self.G.edges(data=True):
            if 'smooth_path' in data: pts = data['smooth_path']
            else: pts = np.array([self.g2w(r, c) for r, c in data['path']])
            plt.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1.2, alpha=0.7)
        pos = nx.get_node_attributes(self.G, 'pos'); coords = np.array(list(pos.values()))
        if len(coords) > 0: plt.scatter(coords[:, 0], coords[:, 1], c='red', s=20, zorder=5)
        plt.title(f"DEBUG: {title} (Nodes: {len(self.G.nodes)})")
        plt.savefig(os.path.join(self.output_dir, filename), dpi=200, bbox_inches='tight'); plt.close()

    def save_results(self):
        plt.figure(figsize=(12, 18))
        ext = [self.org[0], self.org[0]+self.w*self.res, self.org[1], self.org[1]+self.h*self.res]
        plt.imshow(self.flipped_img, cmap='gray', origin='lower', extent=ext, alpha=0.4)
        for u, v, data in self.G.edges(data=True):
            if 'smooth_path' in data: plt.plot(data['smooth_path'][:, 0], data['smooth_path'][:, 1], color='blue', linewidth=1.5, alpha=0.8)
        for n in self.G.nodes():
            if self.G.degree(n) > 1:
                adj = list(self.G.edges(n, data=True))
                for i in range(len(adj)):
                    for j in range(i+1, len(adj)):
                        r = self.fillet_r
                        while r >= 0.15:
                            p = self.get_filleted_turn(adj[i][2]['smooth_path'], adj[j][2]['smooth_path'], n, r)
                            if not self.check_collision(p): plt.plot(p[:,0], p[:,1], color='green', linewidth=2.5, alpha=0.8); break
                            r -= 0.1
        pos = nx.get_node_attributes(self.G, 'pos'); coords = np.array(list(pos.values()))
        if len(coords) > 0: plt.scatter(coords[:, 0], coords[:, 1], c='red', s=40, zorder=10)
        plt.title("Final Elastic Band Optimized Roadmap")
        plt.savefig(os.path.join(self.output_dir, "final_roadmap.png"), dpi=300, bbox_inches='tight')
        
        # Binary Graph Export (The data file for path planning)
        roadmap_data_path = os.path.join(self.output_dir, "roadmap.pkl")
        with open(roadmap_data_path, 'wb') as f:
            pickle.dump(self.G, f)
        print(f">> Final roadmap data saved to: {roadmap_data_path}")

if __name__ == "__main__":
    planner = ElasticBandPlanner(
        png_path='../cropped_maps/hospital.png', 
        yaml_path='../cropped_maps/hospital.yaml',
        inflation_m=0.5,      
        fillet_r=0.9,           
        max_dist_factor=8.0,    
        clearance_factor=1.0,   
        prune_dist_m=2.5,       
        eb_iterations=200,      
        eb_alpha=0.2,           
        eb_beta=0.1,            
        visibility_max_dist_m=20.0
    )