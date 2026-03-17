# -----------------------------------------------------------------------------
# CA McClurg
# Main script for running the discrete event simulator
# -----------------------------------------------------------------------------

import os, random, pickle
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from src.des import SimContext, SimEvents
from src.gnn import neighbor_probs as gnn_probs
from src.utils.print import print_results
from src.utils.gnn import update_recency_array, update_dir_sim_array, update_binary_feature_array
from src.utils.des import cache_shooter_paths
from src.utils.paths import CACHE_DIR, ensure_dir

# ============================================================
# Configuration
# ============================================================

TRAIN_ROBOT  = True     # whether to train robot effects or not
TEST_ROBOT   = True     # whether to test on robot datasets
MODEL_TYPE   = 'samp'   # event model type, options: samp / mean
USE_ROUNDING = True     # whether to round after each epsiode
USE_GNN      = True     # whether to use model for shooter paths
GREEDY_GNN   = False    # whether to use softmax or argmax in model
N_PER_PID    = 10       # number of episodes per participant

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# ============================================================
#  Shooter rollout (GNN)
# ============================================================
def _safe_masked_normalize(prob, mask_idx, fallback_idxs):
    """Mask one index, renormalize, or fall back to uniform over fallback_idxs."""
    prob[mask_idx] = 0.0
    s = prob.sum()
    if s == 0:
        prob[:] = 0.0
        if len(fallback_idxs) > 0:
            for i in fallback_idxs:
                prob[i] = 1.0 / len(fallback_idxs)
    else:
        prob /= s
    return prob

def scheduled_beta(n_eps, tot_eps, beta_min=0.05, beta_max=0.40, power=2.0):
    """
    Smoothly increases beta from beta_min → beta_max as total_visits increases.
    """
    # Clamp between 0 and 1
    ratio = min(1.0, n_eps / float(tot_eps))

    # Smooth schedule using polynomial curve
    beta = beta_min + (beta_max - beta_min) * (ratio ** power)

    return beta

def single_shooter_path(ctx, args):
    model_gnn     = ctx.model_gnn
    greedy        = args["is_greedy"]

    A_sparse, cents_idx, stats = ctx.A_sparse, ctx.cents_idx, ctx.stats
    node_order   = ctx.node_order
    node_idx_map = ctx.node_idx_map
    n_nodes      = len(node_order)
    max_steps    = ctx.max_steps

    # dynamic feature indices
    feat_set     = ctx.features
    i_dirsim     = feat_set.index("dir_sim")
    i_rec        = feat_set.index("recency")
    i_targ       = feat_set.index("has_target")

    # base features
    x_np = np.copy(ctx.static_features)
    x_np[:, [i_rec, i_dirsim, i_targ]] = 0
    base_rec = np.zeros(n_nodes, dtype=np.float32)

    # start node
    start_node = args["ep_nodes"][0] or np.random.choice([n for n in node_order if n != 200])
    curr_idx   = node_idx_map[start_node]

    ep_nodes_idx = [curr_idx]
    ep_targ_idx  = []
    node200_idx = node_order.index(200) if 200 in node_order else None

    # ======================================================
    # Rollout
    # ======================================================
    for step in range(max_steps):

        nbrs = ctx.neighbors_idx[curr_idx]

        # --- dynamic feature updates ---
        update_recency_array(base_rec, x_np, i_rec, curr_idx, step)
        update_dir_sim_array(x_np, i_dirsim, ep_nodes_idx, nbrs, cents_idx)
        update_binary_feature_array(x_np, i_targ, ep_targ_idx)

        # --- 1) GNN ---
        x_t = tf.convert_to_tensor(x_np, dtype=tf.float32)
        p_gnn = gnn_probs(model_gnn, x_t, A_sparse, node_order)
        prob = p_gnn.copy()

        # apply global bias
        for nid, bias in ctx.node_bias.items():
            prob[node_idx_map[nid]] *= bias

        # normalize probabilities
        s = prob.sum()
        if s == 0:
            prob[:] = 0.0
            for ni in nbrs:
                prob[ni] = 1.0 / len(nbrs)
        else:
            prob /= s

        # mask + renormalize cleanly
        if node200_idx is not None:
            prob = _safe_masked_normalize(prob, node200_idx, nbrs)

        # --- 4) Sample next ---
        next_idx = np.argmax(prob) if greedy else np.random.choice(n_nodes, p=prob)
        assert node200_idx is None or node_order[next_idx] != 200, "ERROR: Sampled node 200!"

        ep_nodes_idx.append(next_idx)
        curr_idx = next_idx

        # update visible targets
        for n in stats.sample_visible_nodes(node_order[curr_idx]):
            j = node_idx_map[n]
            if j not in ep_targ_idx:
                ep_targ_idx.append(j)

    # final output
    ep_nodes = [node_order[i] for i in ep_nodes_idx]
    if node200_idx is not None and 200 in ep_nodes:
        raise RuntimeError(f"[CRITICAL] shooter rollout reached node 200! Path={ep_nodes}")

    return ep_nodes

# ============================================================
# Generate shooter paths
# ============================================================
def generate_shooter_paths(ctx, n_test_pids, n_per_pid, seed_base):
    cached = {}
    total_eps = n_test_pids * n_per_pid
    for ep in tqdm(range(total_eps), desc="[DES] Generating shooter paths"):
        ep_seed = seed_base + ep
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        tf.random.set_seed(ep_seed)
        start_node = ctx.emp_nh[ep_seed % len(ctx.emp_nh)][0]
        cached[ep_seed] = single_shooter_path(
            ctx, dict(ep_nodes=[start_node], is_greedy=GREEDY_GNN)
        )

        path = cached[ep_seed]
        ctx.update_model_visits(path)
        beta = scheduled_beta(ep, total_eps, beta_min=0.05, beta_max=0.40, power=2.0)
        ctx.update_bias_weights(beta)
    return cached

def get_or_load_shooter_paths(ctx, split, seed_base=42, test_robot=TEST_ROBOT, n_per_pid=N_PER_PID, use_greedy=GREEDY_GNN):
    dir = ensure_dir(
        CACHE_DIR
        / "generated_paths"
        / f"greedy_{str(use_greedy).lower()}"
        / f"robot_{str(test_robot).lower()}"
    )
    cache_path = dir / f"n{n_per_pid}_S{split}.pkl"

    # load if available
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if "paths" in payload:
                print(f"[DES] Loaded cached shooter paths for split={split} "
                      f"({len(payload['paths'])} episodes)")
                return payload["paths"]
        except Exception as e:
            print(f"[DES] Cache load failed ({e}) — regenerating.")

    # regenerate
    print(f"[DES] Generating shooter paths for split={split}")
    ctx.load_models()
    paths = cache_shooter_paths(
        cache_path,
        generate_shooter_paths,
        ctx,
        n_test_pids=ctx.n_test,
        n_per_pid=n_per_pid,
        seed_base=seed_base
    )

    # free GPU
    del ctx.model_gnn
    tf.keras.backend.clear_session()

    unique_paths = {tuple(v) for v in paths.values()}
    print(f"[DES] Received {len(unique_paths)} unique paths of {len(paths)} total")
    return paths

# ============================================================
# Run a single simulation episode
# ============================================================
def run_episode(ctx, args):

    ep_seed, model_type, use_gnn = args
    random.seed(ep_seed)
    np.random.seed(ep_seed)
    tf.random.set_seed(ep_seed)

    idx = ep_seed % len(ctx.emp_nh)

    # shooter path
    path = ctx.cached_paths[ep_seed] if use_gnn else ctx.emp_nh[idx]

    # robot path
    ep_reff  = ctx.rte[idx]
    r1_nodes = ctx.r1_nh[idx]
    r2_nodes = ctx.r2_nh[idx]

    # event simulation
    result = SimEvents(ctx, path, ep_reff, r1_nodes, r2_nodes, USE_ROUNDING).simulate(model_type)
    result.update({"model_type": model_type})
    result.update({"use_gnn": use_gnn})
    return result

# ============================================================
# Multiprocessing
# ============================================================
def init_worker(split, test_robot, train_robot, cached_paths):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from src.des import SimContext
    global _ctx
    _ctx = SimContext(split, test_robot, train_robot)
    _ctx.cached_paths = cached_paths

def run_episode_process(args):
    global _ctx
    return run_episode(_ctx, args)

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)
    num_workers = min(multiprocessing.cpu_count(), 8)
    seed_base   = 42
    sim_results = []

    for split in range(5):
        ctx = SimContext(split, TEST_ROBOT, TRAIN_ROBOT)
        n_test_pids = ctx.n_test
        n_ep_per_model = n_test_pids * N_PER_PID
        print(f"[DES] split={split} | test_pids={n_test_pids} | eps={n_ep_per_model}")

        # transition model
        if USE_GNN:
            ctx.cached_paths = get_or_load_shooter_paths(ctx, split)
        else:
            ctx.cached_paths = None

        # event model
        args = []
        for ep in range(n_ep_per_model):
            args.append((seed_base + ep, MODEL_TYPE, USE_GNN))

        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(split, TEST_ROBOT, TRAIN_ROBOT, ctx.cached_paths)
        ) as pool:
            current_result = list(tqdm(
                pool.imap_unordered(run_episode_process, args),
                total=len(args),
                desc=f"[DES] simulating split {split}",
                dynamic_ncols=True
            ))

        sim_results.extend(current_result)

    ctx = SimContext(None, TEST_ROBOT, TRAIN_ROBOT)
    print_results(sim_results, ctx)