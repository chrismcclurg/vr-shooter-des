"""
Greedy feature selection for GNN shooter trajectory prediction.
Runs a forward selection procedure, training and validating a model on each candidate feature set,
logging results to disk, and resuming from previous log files if available.
"""

import os
import random
import numpy as np
import tensorflow as tf
import networkx as nx
import ast, re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from src.utils.sho import derive_pids
from src.gnn import log, CallbackManager, build_opt, train, validate, test, create_model, get_info
from src.utils.gnn import get_data
from src.utils.env import (
    get_connection_matrix, get_outside_nodes, get_weighted_shortest_paths, get_nodeType, ez_hard
    )

def run_training(split, feat_set, A_np, G, shortest_paths, node_order, hw_nodes, A_sparse, seed, f_log):
    log(f"\n=> Split: {split} | Features: {feat_set}", f_log)

    pid_train, pid_val, pid_test = derive_pids(split)
    train_dict = get_data(A_np, G, shortest_paths, pid_train, node_order, hw_nodes, feat_set, "train")
    val_dict   = get_data(A_np, G, shortest_paths, pid_val, node_order, hw_nodes, feat_set, "val")
    test_dict  = get_data(A_np, G, shortest_paths, pid_test, node_order, hw_nodes, feat_set, "test")

    xTrain, yTrain, cTrain, tTrain = train_dict['X'], train_dict['y'], train_dict['candidates'], train_dict['types']
    xValid, yValid, cValid, tValid = val_dict['X'], val_dict['y'], val_dict['candidates'], val_dict['types']
    xTest,  yTest,  cTest,  tTest  = test_dict['X'], test_dict['y'], test_dict['candidates'], test_dict['types']

    eh_train, eh_val, eh_test = ez_hard(tTrain), ez_hard(tValid), ez_hard(tTest)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model, opt, lfn = create_model(64, A_np.shape[0], xTest[0].shape[-1], eh_train)
    eh_dummy = np.array([0])
    build_opt(model, opt, lfn, tf.convert_to_tensor(xTrain[:1], dtype=tf.float32), A_sparse, yTrain[:1], cTrain[:1], eh_dummy)

    cb = CallbackManager(patience=15, min_delta=0.0005,
                         lr_patience=10, factor=0.5, min_lr=1e-6)

    for epoch in range(300):
        train_loss, train_loss_easy, train_loss_hard = train(model, xTrain, A_sparse, yTrain, cTrain, opt, eh_train, lfn)
        val_loss, val_loss_easy, val_loss_hard = validate(model, xValid, A_sparse, yValid, cValid, eh_val, lfn)
        current_lr = float(tf.keras.backend.get_value(opt.lr))
        log(f"=> Epoch {epoch:03d} | "
            f"Train: {train_loss.numpy():.4f} (Easy: {train_loss_easy.numpy():.4f}, Hard: {train_loss_hard.numpy():.4f}) | "
            f"Val: {val_loss.numpy():.4f} (Easy: {val_loss_easy.numpy():.4f}, Hard: {val_loss_hard.numpy():.4f}) | "
            f"LR: {current_lr:.6f}", f_log)

        if cb.on_epoch_end(model, opt, val_loss, f_log):
            break  # early stop, no weight restoration needed

    acc_all, acc_easy, acc_hard, _, _ = test(model, xTest, A_sparse, yTest, cTest, eh_test)
    log(f"=> Final Test Accuracy: {acc_all:.4f} | Easy: {acc_easy:.4f} | Hard: {acc_hard:.4f}", f_log)
    return acc_all

def resume_state_from_logs(log_dir, all_features):
    trial_by_len = defaultdict(list)
    pat_trial = re.compile(r"Trial\s+(\[.*?\])")
    pat_acc = re.compile(r"Avg Accuracy:\s+([0-9.]+)")

    for fname in log_dir.iterdir():
        if not (fname.name.startswith("greedy_") and fname.name.endswith(".txt")):
            continue

        with open(fname, "r") as f:
            for line in reversed(f.readlines()):
                if "=> Trial" in line and "Avg Accuracy" in line:
                    try:
                        feats = ast.literal_eval(pat_trial.search(line).group(1))
                        acc = float(pat_acc.search(line).group(1))
                        trial_by_len[len(feats)].append((feats, acc))
                    except Exception:
                        pass
                    break

    if not trial_by_len:
        return [], all_features.copy(), []

    n_total = len(all_features)
    results = []
    selected = []

    for stage_len in range(1, n_total + 1):
        expected = n_total - (stage_len - 1)
        trials = trial_by_len.get(stage_len, [])

        if len(trials) < expected:
            current_stage = trials
            break

        best_feats, best_acc = max(trials, key=lambda x: x[1])
        results.append((best_feats, best_acc))
        selected = list(best_feats)

    else:
        return selected, [], results

    already_tried = {
        (set(f) - set(selected)).pop()
        for f, _ in current_stage if set(selected).issubset(f)
    }

    stage_pool = [f for f in all_features if f not in selected and f not in already_tried]
    return selected, stage_pool, results

def greedy_feature_selection():
    location = 'columbine'
    A, node_order, _ = get_connection_matrix(location)
    outside_nodes = get_outside_nodes(location)
    hw_nodes = get_nodeType(location)
    A_tensor = tf.convert_to_tensor(A, dtype=tf.float32)
    A_np = np.array(A)
    A_sparse = tf.sparse.from_dense(A_tensor)
    G = nx.from_numpy_array(A_np)
    shortest_paths = get_weighted_shortest_paths(G, node_order, outside_nodes, 1)
    seed = 42

    all_features = [
        'recency', 'last_node', 'closeness', 'degree', 'betweenness', 'has_target',
        'has_dead', 'is_outside', 'is_hallway', 'is_classroom',
        'is_common', 'is_stair', 'is_entrance', 'dir_sim', 'room_area', 'ct', 'exit_dist'
    ]

    model_info = get_info()
    log_path = Path(model_info['greedy'])
    selected_features, stage_pool, results = resume_state_from_logs(log_path, all_features)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_log = open(log_path / f"greedy_resume_{timestamp}.txt", "w")
    log(f"=> Resuming Greedy Feature Selection: {timestamp}", f_log)
    log(f"=> Starting with: {selected_features}", f_log)

    while True:
        stage_pool = [f for f in all_features if f not in selected_features]

        if not stage_pool:
            break

        for feat in stage_pool:
            trial_set = selected_features + [feat]
            feat_name = "_".join(trial_set)
            trial_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            trial_log_name = log_path / f"greedy_{feat_name}_{trial_stamp}.txt"

            with open(trial_log_name, "w") as trial_log:
                log(f"=> Trial Start: {trial_set}", trial_log)
                acc_list = [
                    run_training(split, trial_set, A_np, G, shortest_paths, node_order, hw_nodes, A_sparse, seed, trial_log)
                    for split in range(5)
                ]
                avg_acc = np.mean(acc_list)
                log(f"=> Trial {trial_set} => Avg Accuracy: {avg_acc:.4f}", trial_log)

        # Collect and evaluate results for current stage
        n_features = len(selected_features)
        trial_files = [f for f in log_path.iterdir() if f.name.startswith("greedy_") and f.name.endswith(".txt")]
        current_trials = []
        for f in trial_files:
            with open(f, "r") as trial_file:
                for line in trial_file:
                    if "=> Trial" in line and "Avg Accuracy" in line:
                        feats = ast.literal_eval(re.search(r"Trial\s+(\[.*?\])", line).group(1))
                        acc = float(re.search(r"Avg Accuracy:\s+([0-9.]+)", line).group(1))
                        if len(feats) == n_features + 1 and set(selected_features).issubset(feats):
                            current_trials.append((feats, acc))
                        break
        stage_results = current_trials

        if stage_results:
            prev_stage_acc = results[-1][1] if results else 0.0
            best_feats, best_acc = max(stage_results, key=lambda x: x[1])

            if best_acc > prev_stage_acc:
                selected_features = list(best_feats)
                results.append((selected_features, best_acc))
                log(f"\n=> Selected: {selected_features} (Acc: {best_acc:.4f})\n", f_log)
                continue  # restart main loop with updated features
            else:
                log("\n=> No further improvement over previous stage. Stopping.\n", f_log)
                break
        else:
            log("\n=> No trials found for current stage. Stopping.\n", f_log)
            break

    log("\n==== Final Feature Set ====", f_log)
    for feat_set, acc in results:
        log(f"{feat_set} => Accuracy: {acc:.4f}", f_log)

    f_log.close()

if __name__ == "__main__":
    greedy_feature_selection()
