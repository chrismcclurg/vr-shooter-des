import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict
from src.utils.paths import ENV_DIR, VISUAL_DIR, ensure_dir


def compute_exit_dists(node_order, cents, exit_nodes, floor_height=14.0):
    exit_dists = {}
    max_exit_dist = 0.0

    for node in node_order:
        x0, y0, zi0 = cents[node]
        z0 = zi0 * floor_height

        min_dist = float('inf')
        for e in exit_nodes:
            x1, y1, zi1 = cents[e]
            z1 = zi1 * floor_height
            d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
            min_dist = min(min_dist, d)

        exit_dists[node] = min_dist
        max_exit_dist = max(max_exit_dist, min_dist)

    # Normalize and invert: closer = 1.0, farthest = 0.0
    for node in exit_dists:
        if max_exit_dist > 0:
            exit_dists[node] = 1 - exit_dists[node] / max_exit_dist
        else:
            exit_dists[node] = 1.0

    return exit_dists

def get_layout(location, nodes_for_centroids = None):
    layout_files = {
        'columbine':    ['map1.xlsx', 'map2.xlsx'],
        'parkland':     ['map1.xlsx', 'map2.xlsx', 'map3.xlsx'],
        'newtown':      ['map1.xlsx'],
        'uvalde':       ['map1.xlsx'],
    }

    path_layout = VISUAL_DIR / location
    ans = {}

    for i, filename in enumerate(layout_files.get(location, []), 1):
        full_path = path_layout / filename
        layout = pd.read_excel(full_path, header=None).values.astype(int)
        ans[f'layout{i}'] = layout

    # calculate centroids
    centroids = {}
    if nodes_for_centroids:
        for ix, (key, layout) in enumerate(ans.items()):
            labels = np.unique(layout)
            for lab in labels:
                if lab in nodes_for_centroids:
                    rows, cols = np.nonzero(layout == lab)
                    centroids[lab] = (rows.mean(), cols.mean(), ix + 1)

        if location == 'columbine':
            centroids[200] = (70, 100, 2)

    return ans, centroids

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

def get_outside_nodes(location):
    df = pd.read_excel(ENV_DIR / f"{location}.xlsx", index_col=0)
    df = df[df['is_outside'] == 1]
    ans = list(df['node'])
    return ans

def get_nodeType(location):
    df = pd.read_excel(ENV_DIR / f"{location}.xlsx", index_col=1)
    df = df[df['access'] == 1]
    ans = df['is_hallway'].to_dict()
    return ans

def get_nodeTypeMap(location):
    categories = ["outside", "entrance", "cafe", "stairs", "classroom", "library", "hallway"]
    df = pd.read_excel(ENV_DIR / f"{location}.xlsx")
    df = df[df['access'] == 1]
    node_ids = list(df['node'])
    node_names = list(df['location'])
    ans = dict()
    for ix, name_ in enumerate(node_names):
        matched = False
        temp = np.zeros((8))
        for i, cat in enumerate(categories):
            if cat in name_.lower():
                temp[i] = 1
                matched = True
                break
        if not matched:
            temp[-1] = 1  # assign to 'other'
        ans[node_ids[ix]] = temp
    return ans

def get_weighted_shortest_paths(G, node_order, outside_labels, outside_weight=10):
    label_to_idx = {label: i for i, label in enumerate(node_order)}
    outside_indices = set(label_to_idx[lbl] for lbl in outside_labels if lbl in label_to_idx)

    G_weighted = G.copy()
    for u, v in G_weighted.edges():
        if u in outside_indices or v in outside_indices:
            G_weighted[u][v]['weight'] = outside_weight
        else:
            G_weighted[u][v]['weight'] = 1

    return dict(nx.all_pairs_dijkstra_path(G_weighted, weight='weight'))

def idx(x, y, z, DX  = 130, DY = 70, DXY = 3):
    if np.isnan(x):
        xbar = np.nan
        ybar = np.nan
        zbar = np.nan
    else:
        xbar = int(np.floor(x / DXY + 0.5))
        ybar = int(np.floor(y / DXY + 0.5))
        if z < 12:
            zbar = 1
        elif z > 26:
            zbar = 3
        else:
            zbar = 2

    xbar = np.clip(xbar, 0, DX-1)
    ybar = np.clip(ybar, 0, DY-1)
    return xbar, ybar, zbar

def get_label(layout, x, y=None, z=None, fallback_label=None, margin=2, accessible_nodes=None, last_label=None):
    if isinstance(x, tuple):
        x, y, z = x  # unpack

    DX = layout['layout1'].shape[1]
    DY = layout['layout1'].shape[0]
    xi, yi, zi = idx(x, y, z, DX, DY)
    lo = layout[f'layout{zi}']
    try:
        ans = lo[yi, xi]
    except IndexError:
        ans = -1  # outside bounds

    # direct match check
    if accessible_nodes is None or ans in accessible_nodes:
        return ans

    # fuzzy search with Euclidean distance
    closest = None
    ct_dist = float('inf')
    for dx in range(-margin, margin + 1):
        for dy in range(-margin, margin + 1):
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < lo.shape[1] and 0 <= ny < lo.shape[0]:
                neighbor = lo[ny, nx]
                if accessible_nodes is None or neighbor in accessible_nodes:
                    # Prioritize last_label match
                    if neighbor == last_label:
                        return neighbor
                    dist = dx**2 + dy**2  # squared Euclidean distance
                    if dist < ct_dist:
                        closest = neighbor
                        ct_dist = dist
    if closest is not None:
        return closest
    if fallback_label is not None:
        return fallback_label
    return None

def get_connection_matrix(location = "columbine"):
    df = pd.read_excel(ENV_DIR / f"{location}.xlsx", index_col=0)
    df = df[df['access'] == 1]
    df['connectivity'] = df['connectivity'].apply(
        lambda x: [int(i.strip()) for i in str(x).split(',')] if pd.notna(x) else [])

    nodes = list(df['node'])
    df['connectivity'] = df['connectivity'].apply(
        lambda x: [xi for xi in x if xi in nodes])

    canonical_order = sorted(df['node'].unique())
    canonical_names = [xi[:-4] for xi in df['location']]
    ans_names = {}
    for i in range(len(canonical_order)):
        ans_names[canonical_order[i]] = canonical_names[i]

    edges = defaultdict(set)
    G = nx.Graph()
    for _, row in df.iterrows():
        node = row['node']
        neighbors = row['connectivity']
        for neighbor in neighbors:
            edges[node].add(neighbor)
            G.add_edge(node, neighbor)

    asymmetries = []
    for node, neighbors in edges.items():
        for neighbor in neighbors:
            if node not in edges[neighbor]:
                asymmetries.append((node, neighbor))

    if asymmetries:
        print("\n=> asymmetries found (one-way connections):")
        for a, b in asymmetries:
            print(f"{a} → {b} but not {b} → {a}")

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=8)
    plt.title("Connectivity Graph")
    out_dir = ensure_dir(VISUAL_DIR / location)
    plt.savefig(out_dir / "connectivity.png")
    plt.close()
    A = nx.adjacency_matrix(G, nodelist=canonical_order).todense()
    return A, canonical_order, ans_names

def ez_hard(types_array):
    ans = []
    easy_count = 0
    hard_count = 0
    for row in types_array:
        valid = row[row != -1]
        if len(valid) == 1 or np.all(valid == 1):
            easy_count += 1
            ans.append(0)
        else:
            hard_count += 1
            ans.append(1)

    return ans

def precompute_ez_mask(A_sparse, node_order, hw_nodes):
    # A_dense is the dense adjacency matrix (from sparse format)
    A_dense = tf.sparse.to_dense(A_sparse)

    ez_mask = {}  # Dictionary to store easy/hard transition for each node

    for node_idx in range(A_dense.shape[0]):
        # Get the neighbors of the current node
        neighbors_idx = tf.where(A_dense[node_idx] == 1).numpy().flatten()  # Find neighbors (where adjacency is 1)

        # Get the types of neighbors (hallway or not)
        arr = np.array([hw_nodes[node_order[idx]] for idx in neighbors_idx]).reshape(1, -1)

        # Use ez_hard to determine whether the node represents an easy (0) or hard (1) decision
        is_hard = ez_hard(arr)[0]

        # Store the mask for this node (easy/hard transitions for its neighbors)
        ez_mask[node_order[node_idx]] = is_hard

    return ez_mask