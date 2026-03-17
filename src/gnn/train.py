"""
Train and evaluate a GNN model for shooter trajectory prediction across 5 splits.

This script:
    • Loads graph structure, node features, and train/val/test partitions.
    • Trains the model with early stopping and learning-rate scheduling.
    • Saves model weights and per-epoch loss curves for each split.
    • Computes accuracy metrics (overall, easy, hard) for GNN and baselines (Rand, CT, CE, FE, CV, LA).
    • Logs detailed training progress and final averaged results to a timestamped log file.
"""

import os
import random
import numpy as np
import tensorflow as tf
import networkx as nx
from pathlib import Path
from datetime import datetime
from src.utils.sho import derive_pids
from src.utils.gnn import get_data
from src.utils.env import get_connection_matrix, get_outside_nodes, get_nodeType, ez_hard, \
    get_weighted_shortest_paths
from src.gnn import get_acc, get_acc_random, log, CallbackManager, build_opt, get_info, train, \
    validate, test, create_model
from src.utils.paths import ensure_dir


seed            = 42
model_info      = get_info()
model_name      = model_info['name']
model_feat      = model_info['features']
model_wts_path  = ensure_dir(Path(model_info['weights']))
model_log_path  = ensure_dir(Path(model_info['logs']))
model_crv_path  = ensure_dir(Path(model_info['curves']))
model_out_path  = ensure_dir(Path(model_info['output']))
timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
model_log_file  = model_log_path / f"{timestamp}.txt"

location        = 'columbine'
A, node_order, _ = get_connection_matrix(location)
outside_nodes   = get_outside_nodes(location)
hw_nodes        = get_nodeType(location)
A_tensor        = tf.convert_to_tensor(A, dtype=tf.float32)#[None, ...]
A_np            = np.array(A)
A_sparse        = tf.sparse.from_dense(A_tensor)
G               = nx.from_numpy_array(A_np)
shortest_paths  = get_weighted_shortest_paths(G, node_order, outside_nodes, 1)

f = open(model_log_file, "w")
log(f'=> Time:           {timestamp}', f)
log(f'=> Features:       {model_feat}', f)

metrics = ["GNN", "Rand", "CT", "CE", "FE", "CV", "LA"]
accs = {m: {"all": [], "easy": [], "hard": []} for m in metrics}

for split in range(5):

    log("", f)
    log(f'=> PID split:      {split}', f)
    log("", f)

    model_wts_file = model_wts_path / f'split_{split}.ckpt'
    model_crv_file = model_crv_path / f"{timestamp}_{split}.npz"

    pid_train, pid_val, pid_test = derive_pids(split)
    train_dict  = get_data(A_np, G, shortest_paths, pid_train, node_order, hw_nodes, model_feat, "train")
    val_dict    = get_data(A_np, G, shortest_paths, pid_val, node_order, hw_nodes, model_feat, "val")
    test_dict   = get_data(A_np, G, shortest_paths, pid_test, node_order, hw_nodes, model_feat, "test")

    xTrain, yTrain, cTrain, tTrain = train_dict['X'], train_dict['y'], train_dict['candidates'], train_dict['types']
    xValid, yValid, cValid, tValid  = val_dict['X'], val_dict['y'], val_dict['candidates'], val_dict['types']
    xTest, yTest, cTest, tTest  = test_dict['X'], test_dict['y'], test_dict['candidates'] , test_dict['types']

    yCT = test_dict["ct"]
    yCE = test_dict["ce"]
    yFE = test_dict["fe"]
    yCV = test_dict["cv"]
    yLA = test_dict["la"]

    eh_train = ez_hard(tTrain)
    eh_val  = ez_hard(tValid)
    eh_test = ez_hard(tTest)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model, opt, lfn = create_model(64, A_np.shape[0], xTest[0].shape[-1], eh_train)
    eh_dummy = np.array([0])
    build_opt(model, opt, lfn, tf.convert_to_tensor(xTrain[:1], dtype=tf.float32), A_sparse, yTrain[:1], cTrain[:1], eh_dummy)

    train_loss_list = []
    valid_loss_list = []
    num_epochs      = 300
    cb = CallbackManager(patience=15, min_delta=0.0005,
                         lr_patience=10, factor=0.5, min_lr=1e-6)

    for epoch in range(num_epochs):
        train_loss, train_loss_easy, train_loss_hard = train(model, xTrain, A_sparse, yTrain, cTrain, opt, eh_train, lfn)
        val_loss, val_loss_easy, val_loss_hard = validate(model, xValid, A_sparse, yValid, cValid, eh_val, lfn)
        train_loss_list.append(train_loss)
        valid_loss_list.append(val_loss)
        current_lr = float(tf.keras.backend.get_value(opt.lr))
        log(f"=> Epoch {epoch:03d} | "
            f"Train: {train_loss.numpy():.4f} (Easy: {train_loss_easy.numpy():.4f}, Hard: {train_loss_hard.numpy():.4f}) | "
            f"Val: {val_loss.numpy():.4f} (Easy: {val_loss_easy.numpy():.4f}, Hard: {val_loss_hard.numpy():.4f}) | "
            f"LR: {current_lr:.6f}", f)
        stop = cb.on_epoch_end(model, opt, val_loss, f)
        if stop:
            log(f"=> Early stopping triggered at epoch {epoch}", f)
            break

    model.save_weights(str(model_wts_file))
    acc_te, acc_te_e, acc_te_h, _, _ = test(model, xTest, A_sparse, yTest, cTest, eh_test)

    np.savez(str(model_crv_file),
             train_loss=np.array(train_loss_list),
             valid_loss=np.array(valid_loss_list))

    results_split = {
        "GNN":  (acc_te, acc_te_e, acc_te_h),
        "Rand": get_acc_random(cTest, eh_test),
        "CT":   get_acc(yTest, yCT, eh_test),
        "CE":   get_acc(yTest, yCE, eh_test),
        "FE":   get_acc(yTest, yFE, eh_test),
        "CV":   get_acc(yTest, yCV, eh_test),
        "LA":   get_acc(yTest, yLA, eh_test),
    }

    for m, (a, e, h) in results_split.items():
        accs[m]["all"].append(a)
        accs[m]["easy"].append(e)
        accs[m]["hard"].append(h)
        log(f"=> Split {split} {m:5s}: {a:.4f} | Easy: {e:.4f} | Hard: {h:.4f}", f)


for m in metrics:
    acc_all = np.mean(accs[m]["all"])
    acc_e   = np.mean(accs[m]["easy"])
    acc_h   = np.mean(accs[m]["hard"])
    log(f"=> Final {m:5s}: {acc_all:.4f} | Easy: {acc_e:.4f} | Hard: {acc_h:.4f}", f)

f.close()
