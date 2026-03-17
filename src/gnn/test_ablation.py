"""
Ablation study for GNN-based shooter trajectory models across participants.

Outputs:
    CSV file with mean accuracies per metric and difficulty type.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from pathlib import Path
from datetime import datetime
from src.utils.sho import derive_pids
from src.gnn import test, create_model, get_info
from src.utils.gnn import get_data
from src.utils.env import (
    get_connection_matrix, get_outside_nodes, get_weighted_shortest_paths, get_nodeType, ez_hard)


seed = 42
model_info = get_info()
model_name = model_info['name']
model_feat = model_info['features']
model_wts_path = Path(model_info['weights'])
model_out_path = Path(model_info['output'])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

location = 'columbine'
A, node_order, _ = get_connection_matrix(location)
outside_nodes = get_outside_nodes(location)
hw_nodes = get_nodeType(location)
A_tensor = tf.convert_to_tensor(A, dtype=tf.float32)
A_np = np.array(A)
A_sparse = tf.sparse.from_dense(A_tensor)
G = nx.from_numpy_array(A_np)
shortest_paths = get_weighted_shortest_paths(G, node_order, outside_nodes, 1)

metrics = ["GNN"]

for remove_feature in model_feat:
    accs = {m: {"all": [], "easy": [], "hard": []} for m in metrics}
    output_file = model_out_path / f"ablation_{timestamp}_remove_{remove_feature}.csv"

    for ignore_ds in ["E3", "E2"]:
        for split in range(5):
            # Build and load model ONCE per split
            _, _, pid_test = derive_pids(split)
            test_dict = get_data(A_np, G, shortest_paths, pid_test[:1],
                                 node_order, hw_nodes, model_feat,
                                 "test", ignore_dataset=ignore_ds)
            x_dummy = test_dict['X']
            model, _, _ = create_model(64, A_np.shape[0], x_dummy.shape[-1])
            _ = model([tf.convert_to_tensor(x_dummy[:1], dtype=tf.float32), A_sparse])
            model_wts_file = model_wts_path / f"split_{split}.ckpt"
            model.load_weights(model_wts_file)

            # Precompute the column index to zero-out
            idx = model_feat.index(remove_feature)

            for pid in pid_test:
                test_dict = get_data(A_np, G, shortest_paths, [pid],
                                     node_order, hw_nodes, model_feat,
                                     "test", ignore_dataset=ignore_ds)
                xTest, yTest, cTest, tTest = test_dict['X'], test_dict['y'], test_dict['candidates'], test_dict['types']
                eh_pid = ez_hard(tTest)

                # Reset seeds for reproducibility
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)

                # Zero out the selected feature column (ablation)
                xTest_masked = xTest.copy()
                xTest_masked[..., idx] = 0.0

                acc_te, acc_te_e, acc_te_h, _, _ = test(model, xTest_masked, A_sparse, yTest, cTest, eh_pid)

                for m, (a, e, h) in {"GNN": (acc_te, acc_te_e, acc_te_h)}.items():
                    accs[m]["all"].append(a)
                    accs[m]["easy"].append(e)
                    accs[m]["hard"].append(h)

    # Save results for this feature ablation
    ans = {f"{m}_{t}": accs[m][t] for m in metrics for t in ["all", "easy", "hard"]}
    df = pd.DataFrame(ans)
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved ablation results to {output_file}")
