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
from shapely.geometry import Polygon as ShapelyPolygon

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
                 visibility_max_dist_m=20.0,
                 split_long_edges_over_m=6.0,
                 split_target_len_m=5.0,
                 high_overlap_min_dist_m=1.0,
                 high_overlap_max_dist_m=2.5,
                 high_overlap_dist_step_m=0.1,
                 high_overlap_angle_step_deg=5.0,
                 high_overlap_path_inflation_m=None,
                 high_overlap_fov_deg=89.0,
                 max_world_y=None,
                 output_scene_name=None,
                 output_root=None):
        
        # Define output directory: map/planning/trajectory/{scene_name}/planner by default.
        scene_name = output_scene_name or os.path.splitext(os.path.basename(png_path))[0]
        if output_root is None:
            output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory")
        self.output_dir = os.path.join(output_root, scene_name, "planner")
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
        self.split_long_edges_over_m = split_long_edges_over_m
        self.split_target_len_m = split_target_len_m
        self.high_overlap_min_dist_m = high_overlap_min_dist_m
        self.high_overlap_max_dist_m = high_overlap_max_dist_m
        self.high_overlap_dist_step_m = high_overlap_dist_step_m
        self.high_overlap_angle_step_deg = high_overlap_angle_step_deg
        self.high_overlap_path_inflation_m = (
            inflation_m if high_overlap_path_inflation_m is None
            else high_overlap_path_inflation_m
        )
        self.high_overlap_fov_deg = high_overlap_fov_deg
        self.max_world_y = max_world_y

        # 1. Load Map and Metadata
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.res, self.org = cfg['resolution'], cfg['origin']
        
        raw_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        self.flipped_img = cv2.flip(raw_img, 0)
        self.h, self.w = self.flipped_img.shape[:2]
        
        # 2. Collision Map and Potential Field Preparation
        obs_mask = (self.flipped_img < 200).astype(np.uint8)
        if self.max_world_y is not None:
            row_world_y = self.org[1] + np.arange(self.h) * self.res
            clipped_rows = row_world_y > self.max_world_y
            obs_mask[clipped_rows, :] = 1
            print(f">> Clipping planning area above world Y={self.max_world_y:.2f}m.")
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
        self.remove_degenerate_edges()
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

        print(
            f">> STEP 7: Splitting edges longer than {self.split_long_edges_over_m:.1f}m "
            f"into ~{self.split_target_len_m:.1f}m segments..."
        )
        self.split_long_edges()
        self.remove_degenerate_edges()
        self.save_debug_plot("07_long_edges_split.png", "Long Edge Segmentation")
        
        print(">> STEP 8: Finalizing results and exporting Roadmap...")
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

    def _polyline_lengths(self, pts):
        if len(pts) < 2:
            return np.array([0.0])
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.insert(np.cumsum(seg_lens), 0, 0.0)

    def _point_at_distance(self, pts, cum_dist, dist):
        dist = np.clip(dist, 0.0, cum_dist[-1])
        idx = np.searchsorted(cum_dist, dist, side='right') - 1
        idx = min(max(idx, 0), len(pts) - 2)
        seg_len = cum_dist[idx + 1] - cum_dist[idx]
        if seg_len <= 1e-9:
            return pts[idx].copy()
        t = (dist - cum_dist[idx]) / seg_len
        return pts[idx] * (1.0 - t) + pts[idx + 1] * t

    def _slice_polyline(self, pts, cum_dist, start_dist, end_dist):
        start_pt = self._point_at_distance(pts, cum_dist, start_dist)
        end_pt = self._point_at_distance(pts, cum_dist, end_dist)
        interior_mask = (cum_dist > start_dist) & (cum_dist < end_dist)
        interior = pts[interior_mask]
        if len(interior) > 0:
            return np.vstack([start_pt, interior, end_pt])
        return np.vstack([start_pt, end_pt])

    def _unique_split_node(self, world_pt):
        base = self.w2g(world_pt[0], world_pt[1])
        if base not in self.G:
            return base

        for radius in range(1, 20):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    candidate = (base[0] + dr, base[1] + dc)
                    r, c = candidate
                    if (
                        0 <= r < self.h and 0 <= c < self.w
                        and self.collision_map[r, c] == 0
                        and candidate not in self.G
                    ):
                        return candidate
        raise RuntimeError(f"Unable to place split node near {world_pt}.")

    def split_long_edges(self):
        if self.split_long_edges_over_m <= 0 or self.split_target_len_m <= 0:
            print(">> Long edge splitting disabled.")
            return

        split_count = 0
        for u, v, data in list(self.G.edges(data=True)):
            pts = np.array(data.get('smooth_path', []), dtype=float)
            if len(pts) < 2:
                continue

            cum_dist = self._polyline_lengths(pts)
            total_len = float(cum_dist[-1])
            if total_len <= self.split_long_edges_over_m:
                data['weight'] = total_len
                continue

            segment_count = int(np.ceil(total_len / self.split_target_len_m))
            if segment_count < 2:
                data['weight'] = total_len
                continue

            split_distances = np.linspace(0.0, total_len, segment_count + 1)
            split_nodes = [u]
            for dist in split_distances[1:-1]:
                split_world = self._point_at_distance(pts, cum_dist, dist)
                split_node = self._unique_split_node(split_world)
                self.G.add_node(split_node, pos=self.g2w(*split_node))
                split_nodes.append(split_node)
            split_nodes.append(v)

            self.G.remove_edge(u, v)
            for i in range(segment_count):
                start_node = split_nodes[i]
                end_node = split_nodes[i + 1]
                segment_pts = self._slice_polyline(pts, cum_dist, split_distances[i], split_distances[i + 1])
                pixel_path = [self.w2g(pt[0], pt[1]) for pt in segment_pts]
                segment_len = float(self._polyline_lengths(segment_pts)[-1])
                self.G.add_edge(
                    start_node,
                    end_node,
                    path=pixel_path,
                    smooth_path=segment_pts,
                    weight=segment_len,
                )

            split_count += 1

        print(f">> Split {split_count} long edges. Graph now has {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")

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
        if self.G.number_of_nodes() == 0:
            return

        shrink_px = shrink_dist_m / self.res
        mapping = {}
        reserved_targets = set()

        for n in list(self.G.nodes()):
            if self.G.degree(n) != 1:
                continue

            nbr = next(iter(self.G.neighbors(n)))
            path = list(self.G[n][nbr]['path'])
            if (
                np.linalg.norm(np.array(path[0]) - np.array(n)) >
                np.linalg.norm(np.array(path[-1]) - np.array(n))
            ):
                path = path[::-1]

            dist_accum = 0.0
            target_idx = None
            for i in range(len(path) - 1):
                dist_accum += np.linalg.norm(
                    np.array(path[i + 1]) - np.array(path[i])
                )
                if dist_accum >= shrink_px:
                    target_idx = i + 1
                    break

            if target_idx is None:
                continue

            new_n = tuple(path[target_idx])
            # Relabeling onto an existing graph node merges node identities and
            # can turn the edge into a self-loop. Skip such shrink targets.
            if (
                new_n == n or
                new_n in self.G or
                new_n in reserved_targets
            ):
                continue

            self.G[n][nbr]['path'] = path[target_idx:]
            self.G.nodes[n]['pos'] = self.g2w(*new_n)
            mapping[n] = new_n
            reserved_targets.add(new_n)

        if mapping:
            nx.relabel_nodes(self.G, mapping, copy=False)
            print(
                f">> Shrunk {len(mapping)} dead-end edges by "
                f"{shrink_dist_m}m."
            )

    def remove_degenerate_edges(self, min_length_m=None):
        """Remove self-loops and zero-length geometric edges."""
        if min_length_m is None:
            min_length_m = max(self.res * 0.1, 1e-6)

        edges_to_remove = []
        reason_counts = {'self_loop': 0, 'zero_length': 0}
        for u, v, data in list(self.G.edges(data=True)):
            if u == v:
                edges_to_remove.append((u, v))
                reason_counts['self_loop'] += 1
                continue

            if 'smooth_path' in data:
                pts = np.asarray(data['smooth_path'], dtype=float)
            elif data.get('path'):
                pts = np.asarray(
                    [self.g2w(r, c) for r, c in data['path']],
                    dtype=float,
                )
            else:
                pts = np.asarray([
                    self.G.nodes[u]['pos'],
                    self.G.nodes[v]['pos'],
                ], dtype=float)

            length_m = (
                float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                if len(pts) >= 2 else 0.0
            )
            if length_m <= min_length_m:
                edges_to_remove.append((u, v))
                reason_counts['zero_length'] += 1

        self.G.remove_edges_from(edges_to_remove)
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)

        if edges_to_remove or isolated_nodes:
            print(
                ">> Removed degenerate roadmap elements: "
                f"self-loops={reason_counts['self_loop']}, "
                f"zero-length edges={reason_counts['zero_length']}, "
                f"isolated nodes={len(isolated_nodes)}."
            )
        return reason_counts

    def check_collision(self, pts):
        for pt in pts:
            r, c = self.w2g(pt[0], pt[1])
            if not (0 <= r < self.h and 0 <= c < self.w and self.collision_map[r, c] == 0): return True
        return False

    # ================= High-FOV-overlap partner paths =================

    def _dense_polyline(self, pts, step_m=None):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return pts.copy()

        step_m = step_m or max(self.res * 0.5, 0.025)
        dense = [pts[0]]
        for start, end in zip(pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(end - start))
            if seg_len <= 1e-9:
                continue
            count = max(1, int(np.ceil(seg_len / step_m)))
            for t in np.linspace(0.0, 1.0, count + 1)[1:]:
                dense.append(start * (1.0 - t) + end * t)
        return np.asarray(dense, dtype=float)

    def _points_clear_in_mask(self, pts, blocked_mask):
        if len(pts) == 0:
            return False
        rows = np.rint((pts[:, 1] - self.org[1]) / self.res).astype(int)
        cols = np.rint((pts[:, 0] - self.org[0]) / self.res).astype(int)
        inside = (
            (rows >= 0) & (rows < self.h) &
            (cols >= 0) & (cols < self.w)
        )
        return bool(np.all(inside) and not np.any(blocked_mask[rows, cols] != 0))

    def _build_roadmap_exclusion_mask(self):
        path_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        for _, _, data in self.G.edges(data=True):
            pts = np.asarray(data.get('smooth_path', []), dtype=float)
            if len(pts) < 2:
                continue
            rows = np.rint((pts[:, 1] - self.org[1]) / self.res).astype(np.int32)
            cols = np.rint((pts[:, 0] - self.org[0]) / self.res).astype(np.int32)
            pixels = np.column_stack([cols, rows]).reshape(-1, 1, 2)
            cv2.polylines(path_mask, [pixels], isClosed=False, color=1, thickness=1)

        radius_px = max(0, int(np.ceil(self.high_overlap_path_inflation_m / self.res)))
        if radius_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (radius_px * 2 + 1, radius_px * 2 + 1),
            )
            path_mask = cv2.dilate(path_mask, kernel, iterations=1)
        return np.maximum(self.collision_map, path_mask)

    def _translation_has_clear_connections(self, ego_path, partner_path, sample_count=9):
        if len(ego_path) == 0:
            return False
        indices = np.linspace(
            0, len(ego_path) - 1, min(sample_count, len(ego_path))
        ).astype(int)
        for idx in indices:
            connection = np.linspace(ego_path[idx], partner_path[idx], 32)
            if not self._points_clear_in_mask(connection, self.collision_map):
                return False
        return True

    def _nominal_fov_polygon(self, pos, heading_deg):
        angles = np.radians(np.linspace(
            heading_deg - self.high_overlap_fov_deg / 2.0,
            heading_deg + self.high_overlap_fov_deg / 2.0,
            48,
        ))
        arc = np.column_stack([
            pos[0] + self.visibility_dist * np.cos(angles),
            pos[1] + self.visibility_dist * np.sin(angles),
        ])
        return ShapelyPolygon(np.vstack([pos, arc]))

    def _mean_nominal_fov_iou(self, ego_path, partner_path, sample_count=7):
        if len(ego_path) < 2:
            return 0.0
        indices = np.linspace(
            0, len(ego_path) - 1, min(sample_count, len(ego_path))
        ).astype(int)
        values = []
        for idx in indices:
            before = max(0, idx - 1)
            after = min(len(ego_path) - 1, idx + 1)
            delta = ego_path[after] - ego_path[before]
            if np.linalg.norm(delta) <= 1e-9:
                continue
            heading = float(np.degrees(np.arctan2(delta[1], delta[0])))
            ego_fov = self._nominal_fov_polygon(ego_path[idx], heading)
            partner_fov = self._nominal_fov_polygon(partner_path[idx], heading)
            union_area = float(ego_fov.union(partner_fov).area)
            if union_area > 1e-9:
                values.append(float(ego_fov.intersection(partner_fov).area) / union_area)
        return float(np.mean(values)) if values else 0.0

    def _find_high_overlap_translation(self, ego_path, blocked_mask):
        ego_path = np.asarray(ego_path, dtype=float)
        if len(ego_path) < 2 or self._polyline_lengths(ego_path)[-1] <= 1e-6:
            return None, 'degenerate_path'

        dense_ego = self._dense_polyline(ego_path)
        radii = np.arange(
            self.high_overlap_min_dist_m,
            self.high_overlap_max_dist_m + self.high_overlap_dist_step_m * 0.5,
            self.high_overlap_dist_step_m,
        )
        angles = np.radians(
            np.arange(0.0, 360.0, self.high_overlap_angle_step_deg)
        )
        best = None
        first_valid_radius = None

        for radius in radii:
            if (
                first_valid_radius is not None and
                radius > first_valid_radius + 0.3 + 1e-9
            ):
                break

            for angle in angles:
                shift = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                ])
                dense_partner = dense_ego + shift
                if not self._points_clear_in_mask(dense_partner, blocked_mask):
                    continue
                if not self._translation_has_clear_connections(
                    dense_ego, dense_partner
                ):
                    continue

                if first_valid_radius is None:
                    first_valid_radius = float(radius)

                partner_path = ego_path + shift
                fov_iou = self._mean_nominal_fov_iou(ego_path, partner_path)
                angle_deg = float(np.degrees(angle) % 360.0)
                score = (fov_iou, -float(radius), -angle_deg)
                if best is None or score > best['score']:
                    best = {
                        'path': partner_path,
                        'translation': shift,
                        'distance_m': float(radius),
                        'translation_angle_deg': angle_deg,
                        'mean_nominal_fov_iou': fov_iou,
                        'score': score,
                    }

        return (best, None) if best is not None else (None, 'no_valid_translation')

    def save_high_overlap_results(self):
        blocked_mask = self._build_roadmap_exclusion_mask()
        output_graph = self.G.copy()
        matched = 0
        failure_counts = {}

        for _, _, data in output_graph.edges(data=True):
            ego_path = np.asarray(data.get('smooth_path', []), dtype=float)
            result, reason = self._find_high_overlap_translation(
                ego_path, blocked_mask
            )
            data['high_overlap_valid'] = result is not None
            data['high_overlap_failure_reason'] = reason

            if result is None:
                data['high_overlap_path'] = None
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
                continue

            data['high_overlap_path'] = result['path']
            data['high_overlap_translation'] = result['translation']
            data['high_overlap_distance_m'] = result['distance_m']
            data['high_overlap_translation_angle_deg'] = (
                result['translation_angle_deg']
            )
            data['high_overlap_mean_nominal_fov_iou'] = (
                result['mean_nominal_fov_iou']
            )
            matched += 1

        output_graph.graph['high_overlap'] = {
            'min_distance_m': self.high_overlap_min_dist_m,
            'max_distance_m': self.high_overlap_max_dist_m,
            'distance_step_m': self.high_overlap_dist_step_m,
            'angle_step_deg': self.high_overlap_angle_step_deg,
            'roadmap_path_inflation_m': self.high_overlap_path_inflation_m,
            'obstacle_inflation_m': self.inflation_m,
            'fov_deg': self.high_overlap_fov_deg,
            'fov_range_m': self.visibility_dist,
            'matched_edges': matched,
            'total_edges': output_graph.number_of_edges(),
            'failure_counts': failure_counts,
        }

        roadmap_path = os.path.join(
            self.output_dir, 'roadmap_highoverlap.pkl'
        )
        with open(roadmap_path, 'wb') as f:
            pickle.dump(output_graph, f)

        fig, ax = plt.subplots(figsize=(12, 18))
        ext = [
            self.org[0], self.org[0] + self.w * self.res,
            self.org[1], self.org[1] + self.h * self.res,
        ]
        ax.imshow(
            self.flipped_img, cmap='gray', origin='lower',
            extent=ext, alpha=0.35
        )

        first_original = True
        first_partner = True
        first_missing = True
        for _, _, data in output_graph.edges(data=True):
            ego_path = np.asarray(data.get('smooth_path', []), dtype=float)
            if len(ego_path) < 2:
                continue
            if data.get('high_overlap_valid'):
                ax.plot(
                    ego_path[:, 0], ego_path[:, 1],
                    color='#2878b5', linewidth=1.2, alpha=0.65,
                    label='Original trajectory' if first_original else None,
                )
                first_original = False
                partner_path = np.asarray(
                    data['high_overlap_path'], dtype=float
                )
                ax.plot(
                    partner_path[:, 0], partner_path[:, 1],
                    color='#f28e2b', linewidth=1.8, alpha=0.9,
                    label=(
                        'High-overlap trajectory'
                        if first_partner else None
                    ),
                )
                first_partner = False
                mid = len(ego_path) // 2
                ax.plot(
                    [ego_path[mid, 0], partner_path[mid, 0]],
                    [ego_path[mid, 1], partner_path[mid, 1]],
                    color='#59a14f', linewidth=0.6, alpha=0.5,
                )
            else:
                ax.plot(
                    ego_path[:, 0], ego_path[:, 1],
                    color='#d62728', linewidth=1.0,
                    alpha=0.5, linestyle='--',
                    label=(
                        'No valid high-overlap path'
                        if first_missing else None
                    ),
                )
                first_missing = False

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_title(
            'High-FOV-Overlap Roadmap '
            f'({matched}/{output_graph.number_of_edges()} edges matched)'
        )
        ax.legend(loc='best')
        fig.tight_layout()
        image_path = os.path.join(
            self.output_dir, 'final_roadmap_highoverlap.png'
        )
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(
            f">> High-overlap roadmap saved: {roadmap_path} "
            f"({matched}/{output_graph.number_of_edges()} edges matched; "
            f"failures={failure_counts})"
        )
        print(f">> High-overlap roadmap plot saved: {image_path}")
        return output_graph

    @classmethod
    def export_high_overlap_from_existing(
            cls,
            png_path,
            yaml_path,
            roadmap_path,
            inflation_m=0.5,
            output_dir=None,
            min_distance_m=1.0,
            max_distance_m=2.5,
            path_inflation_m=None,
            visibility_max_dist_m=20.0,
            fov_deg=89.0):
        """Export paired high-overlap paths without rebuilding the base roadmap."""
        planner = cls.__new__(cls)
        planner.inflation_m = inflation_m
        planner.high_overlap_min_dist_m = min_distance_m
        planner.high_overlap_max_dist_m = max_distance_m
        planner.high_overlap_dist_step_m = 0.1
        planner.high_overlap_angle_step_deg = 5.0
        planner.high_overlap_path_inflation_m = (
            inflation_m if path_inflation_m is None
            else path_inflation_m
        )
        planner.high_overlap_fov_deg = fov_deg
        planner.visibility_dist = visibility_max_dist_m
        planner.output_dir = (
            output_dir or os.path.dirname(os.path.abspath(roadmap_path))
        )
        os.makedirs(planner.output_dir, exist_ok=True)

        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        planner.res, planner.org = cfg['resolution'], cfg['origin']
        raw_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise FileNotFoundError(f'Unable to read map image: {png_path}')
        planner.flipped_img = cv2.flip(raw_img, 0)
        planner.h, planner.w = planner.flipped_img.shape[:2]
        obs_mask = (planner.flipped_img < 200).astype(np.uint8)
        radius_px = int(inflation_m / planner.res)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius_px * 2 + 1, radius_px * 2 + 1),
        )
        planner.collision_map = cv2.dilate(
            obs_mask, kernel, iterations=1
        )
        with open(roadmap_path, 'rb') as f:
            planner.G = pickle.load(f)
        return planner.save_high_overlap_results()

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
        self.save_high_overlap_results()

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_map_dir = os.path.join(repo_root, "map", "raw_maps")
    scenes = [
        ("hospital", "hospital", None),
        ("warehouse", "warehouse", 10.0),
        ("office", "office", None),
    ]

    for raw_scene_name, output_scene_name, max_world_y in scenes:
        print(f"\n{'=' * 60}")
        print(f">> Building roadmap for raw map: {raw_scene_name} -> {output_scene_name}")
        print(f"{'=' * 60}")
        planner = ElasticBandPlanner(
            png_path=os.path.join(raw_map_dir, f"{raw_scene_name}.png"),
            yaml_path=os.path.join(raw_map_dir, f"{raw_scene_name}.yaml"),
            inflation_m=0.5,
            fillet_r=0.9,
            max_dist_factor=8.0,
            clearance_factor=1.0,
            prune_dist_m=2.5,
            eb_iterations=200,
            eb_alpha=0.2,
            eb_beta=0.1,
            visibility_max_dist_m=20.0,
            split_long_edges_over_m=6.0,
            split_target_len_m=5.0,
            max_world_y=max_world_y,
            output_scene_name=output_scene_name,
        )
