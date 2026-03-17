"""
Evaluate GNN-based shooter trajectory models across real locations/perpetrators.

Outputs:
    CSV file with mean accuracies per metric and difficulty type.
"""

import os
import random
import numpy as np
import tensorflow as tf
import networkx as nx
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.gnn import get_acc, get_acc_random, create_model, test, get_info
from src.utils.gnn import get_real
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
output_file     = model_out_path / f"real_{timestamp}.csv"

print(f'=> Time:           {timestamp}')
print(f'=> Features:       {model_feat}')

metrics = ["RA", "CT", "CE", "FE", "CV", "LA", "GNN"]
final   = []

for perp, loc in [('harris', 'columbine'),
                  ('klebold', 'columbine'),
                  ('cruz', 'parkland'),
                  ('lanza', 'newtown'),
                  ('ramos', 'uvalde')]:

    print(f"\n=> Evaluating {perp} @ {loc}")

    accs = {m: {"all": [], "easy": [], "hard": []} for m in metrics}
    A, node_order, _ = get_connection_matrix(loc)
    outside_nodes   = get_outside_nodes(loc)
    hw_nodes        = get_nodeType(loc)
    A_tensor        = tf.convert_to_tensor(A, dtype=tf.float32)#[None, ...]
    A_np            = np.array(A)
    A_sparse        = tf.sparse.from_dense(A_tensor)
    G               = nx.from_numpy_array(A_np)
    shortest_paths  = get_weighted_shortest_paths(G, node_order, outside_nodes, 1)


    for split in range(5):

        test_dict = get_real(A_np, G, shortest_paths, node_order, hw_nodes, model_feat, loc, perp)
        xTest, yTest, cTest, tTest  = test_dict['X'], test_dict['y'], test_dict['candidates'] , test_dict['types']

        yCT = test_dict["ct"]
        yCE = test_dict["ce"]
        yFE = test_dict["fe"]
        yCV = test_dict["cv"]
        yLA = test_dict["la"]

        eh_test = ez_hard(tTest)

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        model_wts_file = model_wts_path / f"split_{split}.ckpt"
        model, _, _ = create_model(64, A_np.shape[0], xTest[0].shape[-1])
        _ = model([tf.convert_to_tensor(xTest[:1], dtype=tf.float32), A_sparse])  # build model
        model.load_weights(model_wts_file)

        acc_te, acc_te_e, acc_te_h, _, _ = test(model, xTest, A_sparse, yTest, cTest, eh_test)

        if split == 0:
            results_split = {
                "RA":   get_acc_random(cTest, eh_test),
                "CT":   get_acc(yTest, yCT, eh_test),
                "CE":   get_acc(yTest, yCE, eh_test),
                "FE":   get_acc(yTest, yFE, eh_test),
                "CV":   get_acc(yTest, yCV, eh_test),
                "LA":   get_acc(yTest, yLA, eh_test)}
        results_split["GNN"] = (acc_te, acc_te_e, acc_te_h)

        for m, (a, e, h) in results_split.items():
            if m in metrics:
                accs[m]["all"].append(a)
                accs[m]["easy"].append(e)
                accs[m]["hard"].append(h)

    ans = dict()
    ans['shooter'] = perp
    ans['location'] = loc
    for _type in ["all", "easy", "hard"]:
        for m in metrics:
            k = f"{m}_{_type}"
            ans[k] = np.round(np.mean(np.array(accs[m][_type])),4)
    final.append(ans)

df = pd.DataFrame(final)
df.to_csv(output_file, index=False)
