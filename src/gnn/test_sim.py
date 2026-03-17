"""
Evaluate GNN-based shooter trajectory models across participants.

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
from src.gnn import test, create_model, get_info, get_acc, get_acc_random
from src.utils.gnn import get_data
from src.utils.env import (
    get_connection_matrix, get_outside_nodes, get_weighted_shortest_paths, get_nodeType, ez_hard
    )

seed            = 42
model_info      = get_info()
model_name      = model_info['name']
model_feat      = model_info['features']
model_wts_path  = Path(model_info['weights'])
model_out_path  = Path(model_info['output'])
timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file     = model_out_path / f"sim_{timestamp}.csv"

location        = 'columbine'
A, node_order, _ = get_connection_matrix(location)
outside_nodes   = get_outside_nodes(location)
hw_nodes        = get_nodeType(location)
A_tensor        = tf.convert_to_tensor(A, dtype=tf.float32)#[None, ...]
A_np            = np.array(A)
A_sparse        = tf.sparse.from_dense(A_tensor)
G               = nx.from_numpy_array(A_np)
shortest_paths  = get_weighted_shortest_paths(G, node_order, outside_nodes, 1)

metrics = ["RA", "CT", "CE", "FE", "CV", "LA", "GNN"]
accs    = {m: {"all": [], "easy": [], "hard": []} for m in metrics}

yPredList, yTrueList = [], []

for ignore_ds in ["E3", "E2"]:
    for split in range(5):
        _, _, pid_test = derive_pids(split)
        for pid in pid_test:
            test_dict = get_data(A_np, G, shortest_paths, [pid], node_order, hw_nodes, model_feat, "test", ignore_dataset=ignore_ds)
            xTest, yTest, cTest, tTest  = test_dict['X'], test_dict['y'], test_dict['candidates'] , test_dict['types']
            yCT = test_dict["ct"]
            yCE = test_dict["ce"]
            yFE = test_dict["fe"]
            yCV = test_dict["cv"]
            yLA = test_dict["la"]
            stat = test_dict["stats"]
            eh_pid = ez_hard(tTest)

            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

            model_wts_file  = model_wts_path / f'split_{split}.ckpt'
            model, _, _ = create_model(64, A_np.shape[0], xTest[0].shape[-1])
            _ = model([tf.convert_to_tensor(xTest[:1], dtype=tf.float32), A_sparse])  # build model
            model.load_weights(model_wts_file)
            acc_te, acc_te_e, acc_te_h, yPred, yTrue = test(model, xTest, A_sparse, yTest, cTest, eh_pid)

            results_pid = {
                "RA":   get_acc_random(cTest, eh_pid),
                "CT":   get_acc(yTest, yCT, eh_pid),
                "CE":   get_acc(yTest, yCE, eh_pid),
                "FE":   get_acc(yTest, yFE, eh_pid),
                "CV":   get_acc(yTest, yCV, eh_pid),
                "LA":   get_acc(yTest, yLA, eh_pid),
                "GNN":  (acc_te, acc_te_e, acc_te_h),
            }

            yPredList.extend(list(yPred))
            yTrueList.extend(list(yTrue))

            for m, (a, e, h) in results_pid.items():
                if m in metrics:
                    accs[m]["all"].append(a)
                    accs[m]["easy"].append(e)
                    accs[m]["hard"].append(h)


ans = dict()
for _type in ["all", "easy", "hard"]:
    for m in metrics:
        k = f"{m}_{_type}"
        ans[k] = accs[m][_type]

df = pd.DataFrame(ans)
df.to_csv(output_file, index=False)

