import os, pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.paths import CACHE_DIR

def load_shooter_paths(ctx, use_actual = False, is_greedy=False):
    is_robot = ctx.is_robot
    if not use_actual:
        ans = []
        dir_paths = CACHE_DIR / "generated_paths" / f"greedy_{str(is_greedy).lower()}" / f"robot_{str(is_robot).lower()}"
        print(dir_paths)
        for file in dir_paths.iterdir():
            print(file)
            if not file.is_file():
                continue
            with open(file, "rb") as f:
                payload = pickle.load(f)
            paths_split = [payload["paths"][k] for k in payload["paths"]]
            ans.extend(paths_split)
    else:
        ans = ctx.emp_nh

    num_total = len(ans)
    num_unique = len({tuple(xi) for xi in ans})
    per_unique = np.round((num_unique/num_total)*100, 1)

    print(f"[RL] Loaded {num_total} shooter paths ({per_unique}% unique)")
    return ans

def plot_robot_graph(robot, idx_to_node):
    """
    Visualize a robot's neighbor connectivity graph.

    Parameters
    ----------
    robot : object
        Robot instance with a `nbrs_idx` dict (node_idx -> [neighbor_idx]).
    idx_to_node : dict
        Mapping from node indices to node labels for readable node names.
    """
    G = nx.Graph()
    for k, nbrs in robot.nbrs_idx.items():
        for n in nbrs:
            G.add_edge(idx_to_node[k], idx_to_node[n])

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        with_labels=True,
        node_size=600,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title(f"Robot Neighbor Graph ({len(G.nodes)} nodes, {len(G.edges)} edges)")
    plt.show()

def numeric_heuristic(
    ro1_policy,
    ro2_policy,
    env,
    ctx,
    ro1_tar_idx,
    ro2_tar_idx,
):
    """
    Two-robot heuristic policy using consistent naming:
        idx_sho     : shooter node index
        idx_ro1     : robot1 node index
        idx_ro2     : robot2 node index
        nbr1_idx    : neighbors of robot1
        nbr2_idx    : neighbors of robot2

    Each robot has:
        - its own policy_id (ro1_policy, ro2_policy)
        - its own target node (ro1_tar_idx, ro2_tar_idx)
    """

    # ------------------------------------------------------
    # Current positions
    # ------------------------------------------------------
    idx_sho = env.sho.curr_idx
    idx_ro1 = env.ro1.curr_idx
    idx_ro2 = env.ro2.curr_idx

    # per-robot neighbors
    nbr1_idx = env.ro1.nbrs_idx[idx_ro1]
    nbr2_idx = env.ro2.nbrs_idx[idx_ro2]

    paths = ctx.shortest_paths  # dict of shortest paths

    # ------------------------------------------------------
    # Helper for a single robot
    # ------------------------------------------------------
    def compute_action(policy_id, idx_ro, nbr_idx, target_idx, stay=0):
        """Compute the relative action for a single robot."""

        # --------------------------------------------------
        # Policy 0 → move toward robot-specific target node
        # --------------------------------------------------
        if policy_id == 0:
            if idx_ro == target_idx:
                return stay

            curr_dist = len(paths[idx_ro][target_idx]) - 1
            best_dist = curr_dist
            best_act = None

            for a_idx, nbr in enumerate(nbr_idx):
                a_rel = a_idx + 1
                d = len(paths[nbr][target_idx]) - 1
                if d < best_dist:
                    best_dist = d
                    best_act = a_rel

            return best_act if best_act is not None else stay

        # --------------------------------------------------
        # Policy 1 → move toward shooter
        # --------------------------------------------------
        if policy_id == 1:
            curr_dist = len(paths[idx_ro][idx_sho]) - 1
            best_dist = curr_dist
            best_act = None

            for a_idx, nbr in enumerate(nbr_idx):
                a_rel = a_idx + 1
                d = len(paths[nbr][idx_sho]) - 1
                if d < best_dist:
                    best_dist = d
                    best_act = a_rel

            return best_act if best_act is not None else stay

        # Default → stay still
        return stay

    # ------------------------------------------------------
    # Compute each robot's action
    # ------------------------------------------------------
    a1_rel = compute_action(ro1_policy, idx_ro1, nbr1_idx, ro1_tar_idx)
    a2_rel = compute_action(ro2_policy, idx_ro2, nbr2_idx, ro2_tar_idx)

    # ------------------------------------------------------
    # Flatten into joint action
    # ------------------------------------------------------
    action = env._unflatten_action(a1_rel, a2_rel)
    return action