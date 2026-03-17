# -----------------------------------------------------------------------------
# CA McClurg
# This class holds a class for managing graph, features, temporal
# statistics, and loading trained GNN models for simulation rollouts.
# -----------------------------------------------------------------------------

import os
import numpy as np
import networkx as nx
import tensorflow as tf
from pathlib import Path

from src.utils.sho import derive_pids
from src.gnn import create_model, get_info
from src.des.stats import NodeStats
from src.utils.des import cache_fn, compute_empirical_stats, min_trans_dt, summarize_from_hist
from src.utils.gnn import get_base_dict
from src.utils.env import (
    get_static_layout_data, get_outside_nodes, get_connection_matrix, get_weighted_shortest_paths,
    get_nodeType, precompute_ez_mask, get_nodeTypeMap)
from src.utils.robot import compute_D, get_robotNodes
from src.utils.paths import CACHE_DIR, ensure_dir

class SimContext:
    def __init__(self, split, robot_test=True, robot_train=True, verbose=True, robot_stairs=True):
        self.verbose = verbose
        self.robot_stairs = robot_stairs
        self.location = "columbine"
        self.split = split
        self.final_time = 299.5  # cutoff time for simulating episode
        self.max_steps = 200     # max steps used for predicting nodes
        self.is_robot = robot_test
        self.model_info = get_info()
        self._init_graph()
        self._init_static()
        self._init_temporal_stats(robot_test, robot_train)
        self.static_features = np.zeros((len(self.node_order), len(self.features)), dtype=np.float32)
        for i, node in enumerate(self.node_order):
            for j, feat in enumerate(self.features):
                if feat not in ('recency', 'dir_sim', 'has_target'):  # skip dynamic
                    self.static_features[i, j] = self.base_dict[node][feat]

    @property
    def features(self):
        return self.model_info['features']

    def _init_graph(self):
        self.A, self.node_order, self.node_names = get_connection_matrix(self.location)
        self.A_np = np.asarray(self.A, dtype=np.float32)
        self.A_sparse = tf.sparse.from_dense(tf.convert_to_tensor(self.A_np))
        self.G = nx.from_numpy_array(self.A_np)
        self.outside_nodes = get_outside_nodes(self.location)
        self.shortest_paths = get_weighted_shortest_paths(self.G, self.node_order, self.outside_nodes, outside_weight=1)
        self.base_dict = get_base_dict(self.G, self.features, self.node_order)
        self.neighbors_idx = [
            np.array([i for i in np.where(self.A_np[idx])[0] if self.node_order[i] != 200], dtype=np.int32)
            for idx in range(len(self.node_order))
        ]
        self.node_idx_map = {n: i for i, n in enumerate(self.node_order)} # dict[node] = idx
        self.idx_node_map = {i: n for i, n in enumerate(self.node_order)} # dict[idx] = node
        self.Dij = compute_D(self.A_np)

        # Compute graph diameter from shortest paths
        max_dist = 0
        for u, paths_from_u in self.shortest_paths.items():
            for v, path in paths_from_u.items():
                if path:  # non-empty path
                    d = len(path) - 1
                    if d > max_dist:
                        max_dist = d
        self.graph_diameter = max_dist

        if self.verbose:
            print(f"[CTX Graph] nodes={self.G.number_of_nodes()} edges={self.G.number_of_edges()}")

    def _init_static(self):
        self.hw_nodes = get_nodeType(self.location)

        if not self.robot_stairs:
            self.robot1_nodes = get_robotNodes(self.location, robotNo=1)
            self.robot2_nodes = get_robotNodes(self.location, robotNo=2)
        else:
            self.robot1_nodes = get_robotNodes(self.location)
            self.robot2_nodes = get_robotNodes(self.location)

        _, _, _, cents, _, _ = get_static_layout_data(self.location, self.node_order)
        self.cents = cents
        self.ez_mask = precompute_ez_mask(self.A_sparse, self.node_order, self.hw_nodes)
        self.node_type_map = get_nodeTypeMap(self.location)
        self.cents_idx = np.zeros((len(self.node_order), 3), dtype=np.float32)

        for i, n in enumerate(self.node_order):
            self.cents_idx[i] = self.cents[n]
        if self.verbose:
            print(f"[CTX Static] hallways={len(self.hw_nodes)} cents={len(self.cents)} mask={len(self.ez_mask)}")


    def _init_temporal_stats(self, robot_test: bool, robot_train: bool):
        """
        Initialize temporal statistics for baseline or robot-augmented conditions.

        Parameters
        ----------
        robot_test : bool
            If True, evaluate on E4–E5 (robot-present participants).
        robot_train : bool
            If True, include E4–E5 in the training pool.
        """
        # ----------------------------------------------------------
        # cache paths
        stats_dir = ensure_dir(CACHE_DIR / "generated_stats")
        base_name = f"TR{int(robot_train)}_TE{int(robot_test)}"
        split_tag = "SN" if self.split is None else f"S{self.split}"

        def make_path(suffix):
            return stats_dir / f"{base_name}_{split_tag}_{suffix}.pkl"

        path_train = make_path("train")
        path_test  = make_path("test")

        # ----------------------------------------------------------
        # participant splits
        p1, p2, p3 = derive_pids(self.split)
        pids_80p, pids_20p, pids_100p = (p1 + p2, p3, p1 + p2 + p3)
        pids_train, pids_test = (pids_80p, pids_20p) if self.split is not None else (pids_100p, pids_100p)
        self.n_test = len(pids_test) * 2

        # ----------------------------------------------------------
        # dataset selection
        if not robot_test:
            ds_train, ds_test = ['E2', 'E3'], ['E2', 'E3']
        elif robot_train:
            ds_train, ds_test = ['E2', 'E3', 'E4', 'E5'], ['E4', 'E5']
        else:
            ds_train, ds_test = ['E2', 'E3'], ['E4', 'E5']

        if self.verbose:
            print(f"[CTX Temporal] split={self.split} | train={ds_train} × {len(pids_train)} | test={ds_test} × {len(pids_test)}")

        # ----------------------------------------------------------
        # training stats
        stats_dict, _ = cache_fn(
            path_train, compute_empirical_stats,
            pids_train, ds_train, self.node_order, self.shortest_paths, self.A_np, self.Dij
        )
        if self.verbose:
            print(f"[CTX Temporal] Loaded cached empirical data: {path_train}")

        self.min_pt_shooter = min_trans_dt(agent="shooter")
        self.min_pt_robot   = min_trans_dt(agent="robot")
        self.stats = NodeStats.from_empirical_dict(self.node_order, stats_dict, self.min_pt_shooter, self.node_type_map)

        self.node_visits = {k: len(stats_dict['dt'][k]) for k in stats_dict['dt']}
        total_visits = sum(self.node_visits.values())
        self.node_freq = {k: self.node_visits[k]/total_visits for k in self.node_visits}
        self.node_bias = {k: 1.0 for k in self.node_order if k != 200}

        # Online model visitation stats (start empty)
        self.model_visits = {k: 0 for k in self.node_order if k != 200}
        self.model_freq   = {k: 0.0 for k in self.node_order if k != 200}
        self.model_visits_total = 0

        # ----------------------------------------------------------
        # testing stats and summary
        _, self.hist = cache_fn(
            path_test, compute_empirical_stats,
            pids_test, ds_test, self.node_order, self.shortest_paths, self.A_np, self.Dij
        )
        if self.verbose:
            print(f"[CTX Temporal] Loaded cached empirical data: {path_test}")
        a_start, a_node, *stats = summarize_from_hist(self.hist, self.node_order)
        a_node0, a_node1 = zip(*a_node)

        self.emp_nh = [[n[0] for n in pid_hist] for pid_hist in self.hist]
        self.r2_nh = [[n[-1] for n in pid_hist] for pid_hist in self.hist]
        self.r1_nh = [[n[-2] for n in pid_hist] for pid_hist in self.hist]
        self.rte = [[n[-4] for n in pid_hist] for pid_hist in self.hist] # robot test eff
        self.eval_results = (a_node0, a_node1, *stats)
        self.a_start = a_start

    def load_models(self):
        split = self.split or 0
        model_wts_file = Path(self.model_info['weights']) / f"split_{split}.ckpt"
        self.model_gnn, _, _ = create_model(
            hidden_dim=64,
            n_nodes=self.A_np.shape[0],
            n_features=len(self.features),
            path=str(model_wts_file)  # ensure compatibility
        )

        if self.verbose:
            display_path = model_wts_file.relative_to(Path.cwd())
            print(f"[CTX Model] Loaded GNN weights from .\\{display_path}")

    def update_model_visits(self, ep_nodes):
        """Update visitation counts from a single generated trajectory."""
        for node in ep_nodes:
            if node == 200:
                continue
            self.model_visits[node] += 1
            self.model_visits_total += 1

        # Update normalized frequencies
        if self.model_visits_total > 0:
            self.model_freq = {
                k: self.model_visits[k] / self.model_visits_total
                for k in self.model_visits
            }

    def update_bias_weights(self, beta=0.1, eps=1e-6):
        """
        Update per-node bias weights so the model distribution converges
        toward empirical distribution.
        """
        for k in self.node_bias:
            emp = self.node_freq[k]
            mod = self.model_freq[k]
            ratio = (emp + eps) / (mod + eps)
            self.node_bias[k] *= ratio ** beta

        # Normalize the bias so it stays stable
        mean_b = np.mean(list(self.node_bias.values()))
        if mean_b > 0:
            for k in self.node_bias:
                self.node_bias[k] /= mean_b