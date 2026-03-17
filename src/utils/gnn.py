import copy, random
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from collections import Counter
from datetime import datetime
from src.utils.sho import get_participant_data
from src.utils.env import compute_exit_dists, get_layout, get_static_layout_data, get_label
from src.utils.paths import ENV_DIR, SHOOTER_DIR

ALLOWED_INFERRED = {
                    tuple(sorted((299, 284, 214))), tuple(sorted((290, 282, 1))),
                    tuple(sorted((1, 282, 281))), tuple(sorted((279, 287, 294))),
                    tuple(sorted((218, 284, 282))), tuple(sorted((288, 289, 264))),
                    tuple(sorted((277, 285, 292))), tuple(sorted((284, 282, 290))),
                    tuple(sorted((291, 285, 277))), tuple(sorted((277, 276, 297))),
                    tuple(sorted((279, 286, 295))), tuple(sorted((291, 299, 290))),
                    tuple(sorted((103, 134, 140))), tuple(sorted((103, 134, 139))),
                    tuple(sorted((139, 134, 140))), tuple(sorted((1, 131, 101))),
                    tuple(sorted((201, 272, 275)))}

def resolve_ambiguous_transitions(time_labels, pairs_to_resolve={(1, 275), (214, 290), (227, 277), (0, 204), (2,264), (0, 101), (2,289)}, verbose=False):
    resolved = [time_labels[0]]
    suppressed = 0

    for i in range(1, len(time_labels)):
        prev_time, prev_label = resolved[-1]
        curr_time, curr_label = time_labels[i]

        if (prev_label, curr_label) in pairs_to_resolve or (curr_label, prev_label) in pairs_to_resolve:
            if verbose:
                print(f"🔁 Suppressed {prev_label} → {curr_label} at t={curr_time}, staying in {prev_label}")
            suppressed += 1
            resolved.append((curr_time, prev_label))  # override with previous
        else:
            resolved.append((curr_time, curr_label))
    return resolved

def get_nbr(prev_idx, target_idx, A_np, shortest_paths, nbr_stats, nbr_type):
    none_label = f'{nbr_type}_noneIdx'
    same_label = f'{nbr_type}_sameIdx'
    path_label = f'{nbr_type}_nonePath'
    len_label = f'{nbr_type}_lenPath'
    fb_str = f'{nbr_type}_fallback'
    to_str = f'{nbr_type}_total'
    nbrs = np.where(A_np[prev_idx])[0]

    if target_idx is None:
        nbr_stats[to_str] += 1
        nbr_stats[fb_str] += 1
        nbr_stats[none_label] += 1
        return random.choice(nbrs)

    if prev_idx == target_idx:
        nbr_stats[to_str] += 1
        nbr_stats[fb_str] += 1
        nbr_stats[same_label] += 1
        return random.choice(nbrs)

    path = shortest_paths.get(prev_idx, {}).get(target_idx)
    if path is None:
        nbr_stats[to_str] += 1
        nbr_stats[fb_str] += 1
        nbr_stats[path_label] += 1
        if nbr_type == 'ct':
            print(f'no path! {prev_idx} to {target_idx}')
        return random.choice(nbrs)

    elif len(path) < 2:
        nbr_stats[to_str] += 1
        nbr_stats[fb_str] += 1
        nbr_stats[len_label] += 1
        return random.choice(nbrs)
    else:
        nbr_stats[to_str] += 1
        return path[1]

def init_nbr_stats():
    stats = Counter()
    for kind in ['ct', 'ce', 'fe', 'cv', 'la']:
        for suffix in ['noneIdx', 'sameIdx', 'nonePath', 'lenPath', 'total', 'fallback']:
            stats[f'{kind}_{suffix}'] = 0
    return stats

def get_static_layout_data(location, node_order):
    df = pd.read_excel(ENV_DIR / f"{location}.xlsx", index_col=0)
    df = df[df['access'] == 1]
    nodes = df['node'].tolist()
    inside = df['inside'].tolist()
    areas = dict(zip(nodes, inside))
    la_node = nodes[np.argmax(inside)]
    _, cents = get_layout(location, nodes)
    exit_nodes = df[df['is_entrance'] == 1]['node'].tolist()
    exit_dists = compute_exit_dists(node_order, cents, exit_nodes)
    return df, areas, la_node, cents, exit_nodes, exit_dists

def get_base_dict(G, selected_feats, node_order, exit_dists=None, location="columbine"):
    ans = {}
    df = pd.read_excel(ENV_DIR / f"{location}.xlsx", index_col=0)
    df = df[df['access'] == 1]
    centrality_map = {
        'degree': nx.degree_centrality(G) if 'degree' in selected_feats else {},
        'betweenness': nx.betweenness_centrality(G) if 'betweenness' in selected_feats else {},
        'closeness': nx.closeness_centrality(G) if 'closeness' in selected_feats else {},
    }

    if 'room_area' in selected_feats:
        inside_norm = (df['inside'] - df['inside'].min()) / (df['inside'].max() - df['inside'].min())
        df['room_area'] = inside_norm

    if not selected_feats:
        print("NO FEATURES! CREATING DUMMY.")
        selected_feats = ['dummy']

    row_feats = {'room_area', 'is_outside', 'is_stair', 'is_hallway',
                 'is_common', 'is_classroom', 'is_entrance'}

    for _, row in df.iterrows():
        node = row['node']
        ans[node] = {}
        for feat in selected_feats:
            if feat in row_feats:
                ans[node][feat] = row.get(feat, 0)
            elif feat in centrality_map:
                ans[node][feat] = centrality_map[feat].get(node, 0)
            elif feat == 'exit_dist':
                ans[node][feat] = exit_dists.get(node, 0.0) if exit_dists else 0.0
            elif feat == 'dummy':
                ans[node][feat] = 1
            else:
                ans[node][feat] = 0  # Default fallback
    return ans

def get_ct(prev_label, vic_nodes, cents, node_idx_map, x0, y0, z0, nbr_stats=None):
    """
    Returns the index of the closest target (ct_idx) to the previous node.
    """
    ct_node, ct_dist = None, np.inf

    valid_vics = [xi for xi in vic_nodes if xi != prev_label]
    valid_vics = [xi for xi in vic_nodes if xi in node_idx_map]
    if not valid_vics and nbr_stats is not None:
        nbr_stats['no_target'] += 1

    for label in valid_vics:
        x1, y1, zi1 = cents[label]
        z1 = (zi1 - 1) * 14.0
        d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        if d < ct_dist:
            ct_dist = d
            ct_node = label

    if ct_node not in node_idx_map.keys():
        ct_node = None

    return ct_node, node_idx_map[ct_node] if ct_node is not None else None

def get_ce(prev_label, exit_nodes, cents, node_idx_map, x0, y0, z0):
    """
    Returns the index of the closest exit (ce_idx) to the previous node.
    """
    ce_node, ce_dist = None, np.inf

    for label in exit_nodes:
        if label != prev_label:
            x1, y1, zi1 = cents[label]
            z1 = (zi1 - 1) * 14.0
            d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
            if d < ce_dist:
                ce_dist = d
                ce_node = label

    return node_idx_map[ce_node] if ce_node is not None else None

def get_fe(nbr_nodes, exit_nodes, cents, node_idx_map):
    """
    Returns the index of the node in nbr_nodes that is furthest from the closest exit.
    """
    fe_node, max_dist = None, 0.0

    for label in nbr_nodes:
        x0, y0, _ = cents[label]
        # Compute distances from this label to all exits (2D only)
        dists = [np.linalg.norm(np.array([x0 - x1, y0 - y1])) for x1, y1, _ in [cents[e] for e in exit_nodes]]
        min_exit_dist = min(dists)
        if min_exit_dist > max_dist:
            max_dist = min_exit_dist
            fe_node = label

    return node_idx_map[fe_node] if fe_node else None

def get_cv(prev_nodes, nbr_nodes, cents, node_idx_map):
    """
    Returns the index of the neighbor node most aligned with previous movement direction.
    """
    if len(prev_nodes) < 2:
        return None  # Not enough history to compute direction

    n0, n1 = prev_nodes[-2], prev_nodes[-1]
    x0, y0, _ = cents[n0]
    x1, y1, _ = cents[n1]
    v_in = np.array([x1 - x0, y1 - y0])

    best_sim, best_node = -np.inf, None
    for label in nbr_nodes:
        x2, y2, _ = cents[label]
        v_out = np.array([x2 - x1, y2 - y1])
        denom = np.linalg.norm(v_in) * np.linalg.norm(v_out)
        if denom == 0:
            continue  # Skip degenerate cases
        cos_sim = np.dot(v_in, v_out) / denom
        if cos_sim > best_sim:
            best_sim = cos_sim
            best_node = label

    return node_idx_map[best_node] if best_node else None

def update_recency_array(base_recency, x_np, idx_recency, curr_idx, step):
    """Update recency memory (index-based) and normalize to [0, 1]."""
    base_recency[curr_idx] = step + 1
    delta = step + 1 - base_recency
    delta = np.where(base_recency > 0, delta, np.inf)
    x_np[:, idx_recency] = np.where(np.isfinite(delta), 1.0 / (1.0 + delta), 0.0)

def update_binary_feature_array(x_np, idx_has_target, target_idx):
    """Set has_target=1 for target indices, 0 otherwise."""
    x_np[:, idx_has_target] = 0.0
    if len(target_idx):
        x_np[np.array(target_idx, dtype=int), idx_has_target] = 1.0

def update_dir_sim_array(x_np, idx_dirsim, episode_nodes_idx, nbrs_idx, cents_idx):
    """Compute direction similarity feature for neighbors (index-based)."""
    x_np[:, idx_dirsim] = 0.0
    if len(episode_nodes_idx) < 2:
        return

    # Incoming direction vector (use indices directly)
    x0, y0, _ = cents_idx[episode_nodes_idx[-2]]
    x1, y1, _ = cents_idx[episode_nodes_idx[-1]]
    v_in = np.array([x1 - x0, y1 - y0])
    norm_v_in = np.linalg.norm(v_in)
    if norm_v_in == 0:
        return

    if len(nbrs_idx) == 0:
        return

    coords = cents_idx[nbrs_idx, :2]
    v_outs = coords - np.array([x1, y1])
    dot_products = np.dot(v_outs, v_in)
    norms_out = np.linalg.norm(v_outs, axis=1)
    cos_sims = np.divide(dot_products, norms_out * norm_v_in,
                         out=np.zeros_like(dot_products), where=norms_out > 0)
    dir_sims = 0.5 * (cos_sims + 1.0)  # Map [-1, 1] → [0, 1]
    x_np[nbrs_idx, idx_dirsim] = dir_sims

def update_last_node(feat_dict, prev_label):
    feat_dict[prev_label]['last_node'] = 1

def update_binary_feature(feat_dict, locs, feat_name):
    for loc in locs:
        if loc in feat_dict.keys():
            feat_dict[loc][feat_name] = 1

def update_dir_sim(feat_dict, prev_nodes, nbr_nodes, cents):
    for node in feat_dict:
        feat_dict[node]['dir_sim'] = 0
    if len(prev_nodes) > 1:
        x0, y0, _ = cents[prev_nodes[-2]]
        x1, y1, _ = cents[prev_nodes[-1]]
        v_in = np.array([x1 - x0, y1 - y0])
        for label in nbr_nodes:
            x2, y2, _ = cents[label]
            v_out = np.array([x2 - x1, y2 - y1])
            cos_sim = np.dot(v_in, v_out) / (np.linalg.norm(v_in) * np.linalg.norm(v_out))
            sim = cos_sim / 2 + 0.5
            feat_dict[label]['dir_sim'] = sim

def update_ct(feat_dict, ct_node, nbr_nodes, cents):
    for node in feat_dict:
        feat_dict[node]['ct'] = 0
    if ct_node is not None:
        d_nbrs = []
        max_dist = 0.0
        x0, y0, zi0 = cents[ct_node]
        z0 = zi0 * 14.0
        for label in nbr_nodes:
            x1, y1, zi1 = cents[label]
            z1 = zi1 * 14.0
            d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
            d_nbrs.append(d)
            if d > max_dist:
                max_dist = d
        d_nbrs = [1 - xi / max_dist if max_dist > 0 else 1.0 for xi in d_nbrs]
        for ix, label in enumerate(nbr_nodes):
            feat_dict[label]['ct'] = d_nbrs[ix]

def update_time_spent(base_dict, prev_label, prev_time, curr_time):
    base_dict[prev_label]['time_spent'] += (curr_time - prev_time)

def normalize_time_spent(feat_dict, curr_time):
    for node in feat_dict:
        feat_dict[node]['time_spent'] /= max(curr_time, 1e-3)

def update_recency(base_dict, prev_label, visit_count):
    base_dict[prev_label]['recency'] = visit_count

def normalize_recency(feat_dict, visit_count):
    for node in feat_dict:
        last_visit = feat_dict[node].get('recency', -np.inf)
        if last_visit == -np.inf or last_visit == 0:
            feat_dict[node]['recency'] = 0.0
        else:
            delta = visit_count - last_visit
            feat_dict[node]['recency'] = 1 / (1 + delta)

def convert_event_dicts_to_array(event_list, node_order, selected_feats):
    n_events = len(event_list)
    n_nodes = len(node_order)
    n_features = len(event_list[0][node_order[0]])
    data = np.zeros((n_events, n_nodes, n_features), dtype=np.float32)
    node_idx_map = {node: i for i, node in enumerate(node_order)}
    feat_order = selected_feats
    feat_idx_map = {feat: i for i, feat in enumerate(feat_order)}
    for t, snapshot in enumerate(event_list):
        for node_id, feat in snapshot.items():
            i = node_idx_map[node_id]
            for feat_name, value in feat.items():
                j = feat_idx_map[feat_name]
                data[t, i, j] = value
    return data, feat_order

def get_data(A_np, G, shortest_paths, pids, node_order, hw_nodes, selected_feats, data_type="", ignore_dataset = None):

    def append_target_idx(source_idx, target_idx, nbrs_idx, tag, ylist):
        idx = get_nbr(source_idx, target_idx, A_np, shortest_paths, nbr_stats, tag)
        if idx not in nbrs_idx:
            raise ValueError(f"[{tag}] idx {idx} not in nbrs_idx from {source_idx}. Graph mismatch or path error.")
        ylist.append(np.where(nbrs_idx == idx)[0][0])

    # count fallbacks
    nbr_stats = init_nbr_stats()

    # static data upfront
    df, areas, la_node, cents, exit_nodes, exit_dists = get_static_layout_data('columbine', node_order)

    # initialize output lists
    event_list, yTest, edge_candidates, edge_types = [], [], [], []
    yCT, yCE, yFE, yCV, yLA = [], [], [], [], []
    good_count = bad_count = 0
    unrecoverable_transitions = []
    node_idx_map = {node: i for i, node in enumerate(node_order)}
    all_nodes = set(node_order)
    dat_count = 0
    datasets = ['E2', 'E3']
    if ignore_dataset is not None:
        datasets = [xi for xi in datasets if xi != ignore_dataset]

    for dataset in datasets:
        for pid in pids:
            pt, player, [nos, nvs, nds], layout = get_participant_data('columbine', dataset, pid)
            x, y, z = player['px'], player['py'], player['pz']
            time_labels = [(pt[i], get_label(layout, x[i], y[i], z[i], fallback_label=None, accessible_nodes=all_nodes)) for i in range(len(pt))]
            base_dict   = get_base_dict(G, selected_feats, node_order, exit_dists)
            smoothed    = resolve_ambiguous_transitions(time_labels)
            tar_nodes, vic_nodes, prev_nodes = [], [], []
            visit_count = 1
            for i in range(1, len(smoothed)): #loop through timesteps
                for d in nds[i]:  # loop through victims per timestep
                    label = get_label(layout, d, fallback_label=None, accessible_nodes=all_nodes)
                    if label not in vic_nodes:
                        vic_nodes.append(label)
                for v in nvs[i]: # loop through visible alive per timestep
                    label = get_label(layout, v, fallback_label=None, accessible_nodes=all_nodes)
                    if label not in tar_nodes:
                        tar_nodes.append(label)

                # continue if not a new event
                prev_time, prev_label = smoothed[i-1]
                curr_time, curr_label = smoothed[i]
                if curr_label == prev_label:
                    continue

                # NEW EVENT
                visit_count += 1
                prev_nodes.append(prev_label)
                prev_idx = node_idx_map[prev_label]
                curr_idx = node_idx_map[curr_label]
                nbrs_idx = np.where(A_np[prev_idx])[0]
                nbr_nodes = [node_order[idx] for idx in nbrs_idx]
                nbr_nodes = [xi for xi in nbr_nodes if xi != 200]
                x0, y0, zi0 = cents[prev_label]
                z0 = (zi0 - 1) * 14.0
                ct_node, ct_idx = get_ct(prev_label, tar_nodes, cents, node_idx_map, x0, y0, z0, nbr_stats)
                if data_type == "test":
                    ce_idx = get_ce(prev_label, exit_nodes, cents, node_idx_map, x0, y0, z0)
                    fe_idx = get_fe(nbr_nodes, exit_nodes, cents, node_idx_map)
                    cv_idx = get_cv(prev_nodes, nbr_nodes, cents, node_idx_map)
                    la_idx = node_idx_map[la_node] if la_node else None

                # dynamic feature updates -------------------------------------
                if 'time_spent' in selected_feats:
                    update_time_spent(base_dict, prev_label, prev_time, curr_time)
                if 'recency' in selected_feats:
                    update_recency(base_dict, prev_label, visit_count)
                feat_dict = copy.deepcopy(base_dict)
                if 'last_node' in selected_feats:
                    update_last_node(feat_dict, prev_label)
                if 'time_spent' in selected_feats:
                    normalize_time_spent(feat_dict, curr_time)
                if 'recency' in selected_feats:
                    normalize_recency(feat_dict, visit_count)
                if 'has_dead' in selected_feats:
                    update_binary_feature(feat_dict, vic_nodes, 'has_dead')
                if 'has_target' in selected_feats:
                    update_binary_feature(feat_dict, tar_nodes, 'has_target')
                if 'dir_sim' in selected_feats:
                    update_dir_sim(feat_dict, prev_nodes, nbr_nodes, cents)
                if 'ct' in selected_feats:
                    update_ct(feat_dict, ct_node, nbr_nodes, cents)

                # handle error transitions ------------------------------------
                if curr_idx not in nbrs_idx:
                    try:
                        path = shortest_paths.get(prev_idx, {}).get(curr_idx, None)

                        if path is None:
                            unrecoverable_transitions.append((prev_label, curr_label))
                            bad_count += 1
                            continue
                        if len(path) == 3: # there must be an intermediate node!
                            trans_nodes = [node_order[path[j]] for j in range(3)]
                            sorted_trans = tuple(sorted(trans_nodes))

                            if sorted_trans in ALLOWED_INFERRED:
                                a, b, c = trans_nodes
                                a_idx, b_idx, c_idx = path
                                nbrs_idx_ab = np.where(A_np[a_idx])[0]
                                edge_candidates.append(nbrs_idx_ab)
                                edge_types.append([hw_nodes[node_order[idx]] for idx in nbrs_idx_ab])
                                yTest.append(np.where(nbrs_idx_ab == b_idx)[0][0])
                                event_list.append(copy.deepcopy(feat_dict))
                                good_count += 1
                                prev_nodes.append(b)

                                if data_type == "test":
                                    append_target_idx(a_idx, ct_idx, nbrs_idx_ab, 'ct', yCT)
                                    append_target_idx(a_idx, ce_idx, nbrs_idx_ab, 'ce', yCE)
                                    append_target_idx(a_idx, fe_idx, nbrs_idx_ab, 'fe', yFE)
                                    append_target_idx(a_idx, cv_idx, nbrs_idx_ab, 'cv', yCV)
                                    append_target_idx(a_idx, la_idx, nbrs_idx_ab, 'la', yLA)

                                # dynamic feature updates for node b ----------
                                visit_count += 1
                                if 'last_node' in selected_feats:
                                    update_last_node(base_dict, b)
                                if 'recency' in selected_feats:
                                    update_recency(base_dict, b, visit_count)
                                feat_b = copy.deepcopy(base_dict)
                                if 'time_spent' in selected_feats:
                                    normalize_time_spent(feat_b, curr_time)
                                if 'recency' in selected_feats:
                                    normalize_recency(feat_b, visit_count)
                                if 'has_dead' in selected_feats:
                                    update_binary_feature(feat_b, vic_nodes, 'has_dead')
                                if 'has_target' in selected_feats:
                                    update_binary_feature(feat_b, tar_nodes, 'has_target')
                                nbrs_idx_bc = np.where(A_np[node_idx_map[b]])[0]
                                nbr_nodes_bc = [node_order[idx] for idx in nbrs_idx_bc if node_order[idx] != 200]
                                if 'dir_sim' in selected_feats:
                                    update_dir_sim(feat_b, prev_nodes, nbr_nodes_bc, cents)
                                if 'ct' in selected_feats:
                                    update_ct(feat_b, ct_node, nbr_nodes_bc, cents)

                                edge_candidates.append(nbrs_idx_bc)
                                edge_types.append([hw_nodes[node_order[idx]] for idx in nbrs_idx_bc])
                                yTest.append(np.where(nbrs_idx_bc == node_idx_map[c])[0][0])
                                event_list.append(feat_b)
                                good_count += 1

                                if data_type == "test":
                                    append_target_idx(b_idx, ct_idx, nbrs_idx_bc, 'ct', yCT)
                                    append_target_idx(b_idx, ce_idx, nbrs_idx_bc, 'ce', yCE)
                                    append_target_idx(b_idx, fe_idx, nbrs_idx_bc, 'fe', yFE)
                                    append_target_idx(b_idx, cv_idx, nbrs_idx_bc, 'cv', yCV)
                                    append_target_idx(b_idx, la_idx, nbrs_idx_bc, 'la', yLA)

                                prev_label = c
                                prev_time = curr_time
                                continue
                            else:
                                bad_count += 1
                                print(prev_label, curr_label)
                                continue
                        else:
                            unrecoverable_transitions.append((prev_label, curr_label))
                            bad_count += 1
                    except Exception:
                        unrecoverable_transitions.append((prev_label, curr_label))
                        bad_count += 1
                    continue
                else:
                    if data_type == "test":
                        append_target_idx(prev_idx, ct_idx, nbrs_idx, 'ct', yCT)
                        append_target_idx(prev_idx, ce_idx, nbrs_idx, 'ce', yCE)
                        append_target_idx(prev_idx, fe_idx, nbrs_idx, 'fe', yFE)
                        append_target_idx(prev_idx, cv_idx, nbrs_idx, 'cv', yCV)
                        append_target_idx(prev_idx, la_idx, nbrs_idx, 'la', yLA)

                    edge_candidates.append(nbrs_idx)
                    edge_types.append([hw_nodes[node_order[idx]] for idx in nbrs_idx])
                    yTest.append(np.where(nbrs_idx == curr_idx)[0][0])
                    event_list.append(feat_dict)
                    good_count += 1
                    prev_label = curr_label
                    prev_time = curr_time
            dat_count += 1
    X_t, feat_order = convert_event_dicts_to_array(event_list, node_order, selected_feats)
    candidates = tf.keras.preprocessing.sequence.pad_sequences(edge_candidates, padding='post', value=-1)
    types = tf.keras.preprocessing.sequence.pad_sequences(edge_types, padding='post', value=-1)
    return {
        "X": X_t,
        "y": np.array(yTest),
        "candidates": candidates,
        "types": types,
        "feat_order": feat_order,
        "ct": np.array(yCT),
        "ce": np.array(yCE),
        "fe": np.array(yFE),
        "cv": np.array(yCV),
        "la": np.array(yLA),
        "stats": nbr_stats
    }

def get_real(A_np, G, shortest_paths, node_order, hw_nodes, selected_feats, location='columbine', offender="klebold"):
    def append_target_idx(source_idx, target_idx, nbrs_idx, tag, ylist):
        idx = get_nbr(source_idx, target_idx, A_np, shortest_paths, nbr_stats, tag)
        if idx not in nbrs_idx:
            raise ValueError(f"[{tag}] idx {idx} not in nbrs_idx from {source_idx}. Graph mismatch or path error.")
        ylist.append(np.where(nbrs_idx == idx)[0][0])

    nbr_stats = init_nbr_stats()
    df, areas, la_node, cents, exit_nodes, exit_dists = get_static_layout_data(location, node_order)
    event_list, yTest, edge_candidates, edge_types = [], [], [], []
    yCT, yCE, yFE, yCV, yLA = [], [], [], [], []
    node_idx_map = {node: i for i, node in enumerate(node_order)}

    df = pd.read_excel(SHOOTER_DIR / f"{offender}.xlsx")
    t = [datetime.combine(datetime.today(), xi) for xi in df['Time']]
    t = [int((ti - t[0]).total_seconds()) for ti in t]
    c = list(df['Region'])
    a = [[int(x.strip()) for x in str(row).split(',') if x != 'nan'] for row in df['Target']]
    v = [[int(x.strip()) for x in str(row).split(',') if x != 'nan'] for row in df['Victim']]
    base_dict = get_base_dict(G, selected_feats, node_order, exit_dists, location)
    prev_nodes = []
    visit_count = 1
    for i in range(1, len(t)):
        prev_time, prev_label = t[i-1], c[i-1]
        curr_time, curr_label = t[i], c[i]
        visit_count += 1
        prev_nodes.append(prev_label)
        prev_idx = node_idx_map[prev_label]
        curr_idx = node_idx_map[curr_label]
        nbrs_idx = np.where(A_np[prev_idx])[0]
        nbr_nodes = [node_order[idx] for idx in nbrs_idx if str(node_order[idx]) != '200']
        x0, y0, zi0 = cents[prev_label]
        z0 = (zi0 - 1) * 14.0

        ct_node, ct_idx = get_ct(prev_label, a[i], cents, node_idx_map, x0, y0, z0, nbr_stats)
        ce_idx = get_ce(prev_label, exit_nodes, cents, node_idx_map, x0, y0, z0)
        fe_idx = get_fe(nbr_nodes, exit_nodes, cents, node_idx_map)
        cv_idx = get_cv(prev_nodes, nbr_nodes, cents, node_idx_map)
        la_idx = node_idx_map[la_node] if la_node else None

        # dynamic feature updates -------------------------------------
        if 'time_spent' in selected_feats:
            update_time_spent(base_dict, prev_label, prev_time, curr_time)
        if 'recency' in selected_feats:
            update_recency(base_dict, prev_label, visit_count)
        feat_dict = copy.deepcopy(base_dict)
        if 'last_node' in selected_feats:
            update_last_node(feat_dict, prev_label)
        if 'time_spent' in selected_feats:
            normalize_time_spent(feat_dict, curr_time)
        if 'recency' in selected_feats:
            normalize_recency(feat_dict, visit_count)
        if 'has_dead' in selected_feats:
            update_binary_feature(feat_dict, v[i], 'has_dead')
        if 'has_target' in selected_feats:
            update_binary_feature(feat_dict, a[i], 'has_target')
        if 'dir_sim' in selected_feats:
            update_dir_sim(feat_dict, prev_nodes, nbr_nodes, cents)

        append_target_idx(prev_idx, ct_idx, nbrs_idx, 'ct', yCT)
        append_target_idx(prev_idx, ce_idx, nbrs_idx, 'ce', yCE)
        append_target_idx(prev_idx, fe_idx, nbrs_idx, 'fe', yFE)
        append_target_idx(prev_idx, cv_idx, nbrs_idx, 'cv', yCV)
        append_target_idx(prev_idx, la_idx, nbrs_idx, 'la', yLA)

        event_list.append(feat_dict)
        edge_candidates.append(nbrs_idx)
        edge_types.append([hw_nodes[node_order[idx]] for idx in nbrs_idx])
        try:
            yTest.append(np.where(nbrs_idx == curr_idx)[0][0])
        except:
            print(f"Error at node {node_order[curr_idx]}")
    X_t, feat_order = convert_event_dicts_to_array(event_list, node_order, selected_feats)
    candidates = tf.keras.preprocessing.sequence.pad_sequences(edge_candidates, padding='post', value=-1)
    types = tf.keras.preprocessing.sequence.pad_sequences(edge_types, padding='post', value=-1)
    return {
        "X": X_t,
        "y": np.array(yTest),
        "candidates": candidates,
        "types": types,
        "feat_order": feat_order,
        "ct": np.array(yCT),
        "ce": np.array(yCE),
        "fe": np.array(yFE),
        "cv": np.array(yCV),
        "la": np.array(yLA),
        "stats": nbr_stats
    }