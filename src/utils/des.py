import os, pickle
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from scipy.stats import truncnorm
from tqdm import tqdm
from src.utils.sho import get_participant_data
from src.utils.env import get_label
from src.utils.gnn import resolve_ambiguous_transitions
from src.utils.robot import get_robot_data, get_robot_snapshot
from src.utils.paths import ENV_DIR


def moment_matched_normal(mean, lo, hi, size, std=None, min_std=1e-6):
    """Draw truncated-normal samples with approximate mean preservation."""
    if not all(map(np.isfinite, [mean, lo, hi])):
        raise ValueError(f"Invalid input: mean={mean}, lo={lo}, hi={hi}")
    if hi <= lo or size <= 0:
        return np.full(size, mean)
    if np.isclose(hi, lo, atol=1e-8):
        return np.full(size, lo)

    std = std if (std and std >= min_std) else max((hi - lo) / 2.0, min_std)
    a, b = (lo - mean) / std, (hi - mean) / std
    if a >= b:
        return np.full(size, mean)

    try:
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    except Exception as e:
        print(f"[TruncNorm Error] mean={mean:.3f} std={std:.3f} lo={lo:.3f} hi={hi:.3f} a={a:.3f} b={b:.3f} err={e}")
        return np.full(size, mean)

def moment_matched_lognormal(mu, sigma):
    """Return a single lognormal sample with the same mean and std as (mu, sigma)."""
    if mu <= 0 or sigma <= 0:
        return mu  # fallback for degenerate cases
    sigma_ln = np.sqrt(np.log(1 + (sigma / mu) ** 2))
    mu_ln = np.log(mu) - 0.5 * sigma_ln ** 2
    return np.random.lognormal(mu_ln, sigma_ln)

def cache_fn(cache_path, func, *args, **kwargs):
    """
    Load result from cache if present; otherwise call `func(*args, **kwargs)`,
    save to disk, and return a consistent (stats_dict, full_history) tuple.

    Supports both legacy tuple returns and new dict+history payloads.
    """

    # ------------------------------------------------------------------
    # Try loading cached result
    # ------------------------------------------------------------------
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            # Normalize to (stats_dict, full_history)
            if isinstance(result, dict) and "stats" in result:
                return result["stats"], result["history"]
            elif isinstance(result, tuple) and len(result) == 2:
                return result
            elif isinstance(result, tuple) and len(result) >= 7:
                # Legacy empirical tuple
                return result
            else:
                # Unknown type; just return raw for safety
                return result

        except Exception as e:
            print(f"[CacheFn] Failed to load {cache_path}: {e} — recomputing.")

    # ------------------------------------------------------------------
    # Compute new result
    # ------------------------------------------------------------------
    print(f"[DES Utils] Computing empirical data (no cache found): {cache_path}")
    result = func(*args, **kwargs)

    # ------------------------------------------------------------------
    # Save result back to cache
    # ------------------------------------------------------------------
    try:
        if isinstance(result, tuple) and len(result) >= 7:
            # Legacy tuple format
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            print(f"[DES Utils] Cached legacy empirical tuple saved to: {cache_path}")
            return result

        elif isinstance(result, tuple) and len(result) == 2:
            # New-style (stats_dict, full_history)
            stats_dict, full_history = result
            payload = dict(stats=stats_dict, history=full_history)
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)
            print(f"[DES Utils] Cached new empirical data saved to: {cache_path}")
            return stats_dict, full_history

        else:
            # Unknown structure — save generically but still return tuple
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            print(f"[DES Utils] Cached result (generic) saved to: {cache_path}")
            # Fallback: return (result, None) to keep tuple form
            return result, None

    except Exception as e:
        print(f"[Cache Warning] Failed to write cache to {cache_path}: {e}")
        # Still return tuple form for consistency
        return result, None

def cache_shooter_paths(cache_path, func, *args, **kwargs):
    """
    Load cached shooter paths if present; otherwise compute them via func(*args, **kwargs),
    save to disk, and return the shooter path dictionary.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if "paths" in payload:
                print(f"[DES Utils] Loaded cached shooter paths: {cache_path}")
                return payload["paths"]
        except Exception as e:
            print(f"[CacheShooterPaths] Failed to load {cache_path}: {e} — recomputing.")

    print(f"[DES Utils] Generating shooter paths (no cache found): {cache_path}")
    paths = func(*args, **kwargs)

    try:
        tmp_path = str(cache_path) + ".tmp"
        payload = dict(split=kwargs.get("split", None), paths=paths)
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, cache_path)
        print(f"[DES Utils] Shooter paths cached to: {cache_path}")
    except Exception as e:
        print(f"[CacheShooterPaths Warning] Failed to write cache to {cache_path}: {e}")

    return paths

def compute_empirical_stats(pids, datasets, node_order, shortest_paths, A_np, Dij, alpha=1.0, min_dt=0.5):
    """Compute empirical per-node temporal statistics and visibility probabilities."""

    def label_set(xyz_list, layout, all_nodes):
        """Return set of labels corresponding to visible positions."""
        return {
            get_label(layout, xyz, fallback_label=None, accessible_nodes=all_nodes)
            for xyz in xyz_list
        }

    def get_shooter_data(dataset, pid):
        """Return smoothed shooter path and basic state variables."""
        pt, player, [nos, nvs, nds], layout = get_participant_data('columbine', dataset, pid)
        x, y, z, ns = player['px'], player['py'], player['pz'], player['ns']
        labels = [
            (pt[i], get_label(layout, x[i], y[i], z[i],
                              fallback_label=None, accessible_nodes=all_nodes))
            for i in range(len(pt))
        ]
        smoothed = resolve_ambiguous_transitions(labels)
        return pt, ns, nos, nvs, nds, layout, smoothed

    def get_robot_labels(dataset, pid, layout, L):
        """Return robot label lists (or empty placeholders)."""
        if dataset not in ['E4', 'E5']:
            return ([None] * L, [None] * L, [None] * L, [None] * L)

        rt, r1, r2 = get_robot_data('columbine', dataset, pid)
        rx1, ry1, rx2, ry2 = r1['px'], r1['py'], r2['px'], r2['py']
        r1_labels = [get_label(layout, rx1[i], ry1[i], 0, fallback_label=None, accessible_nodes=all_nodes)
                     for i in range(len(rt))]
        r2_labels = [get_label(layout, rx2[i], ry2[i], 15, fallback_label=None, accessible_nodes=all_nodes)
                     for i in range(len(rt))]
        r1_idxs = [node_idx_map.get(lbl, None) for lbl in r1_labels]
        r2_idxs = [node_idx_map.get(lbl, None) for lbl in r2_labels]
        return r1_labels, r2_labels, r1_idxs, r2_idxs

    def record_transition(label, dt, ds, dv, R_eff, Rt, r1_lbl, r2_lbl):
        """Compact helper to record a new event."""
        dt_node[label].append(dt)
        dv_node[label].append(dv)
        ds_node[label].append(ds)
        R_node[label].append(R_eff)
        n_tot[label].append(len(nvs[i]) + len(nos[i]) + len(nds[i]))
        dt_minself[label] = min(dt_minself[label], dt)
        total_dv_node[label][pid] += dv
        pid_history.append((label, dt, ds, dv, R_eff, Rt, r1_lbl, r2_lbl))

    # Initialization
    node_idx_map = {node: i for i, node in enumerate(node_order)}
    all_nodes = set(node_order)
    N = len(node_order)

    dt_node, dv_node, ds_node, R_node = ({node: [] for node in node_order} for _ in range(4))
    n_tot = {node: [] for node in node_order}
    dt_minself = {node: np.inf for node in node_order}

    vis = np.zeros((N, N), dtype=np.int32)
    tot = np.zeros((N, N), dtype=np.int32)
    total_dv_node = {node: defaultdict(float) for node in node_order}
    full_history = []

    # Main Data Loop
    for dataset in datasets:
        for pid in tqdm(pids, desc=f"=> {dataset} empirical data"):

            pt, ns, nos, nvs, nds, layout, smoothed = get_shooter_data(dataset, pid)
            r1_labels, r2_labels, r1_idxs, r2_idxs = get_robot_labels(dataset, pid, layout, len(pt))

            start_t, start_label = smoothed[0]
            start_v = start_s = 0
            start_Rt = np.zeros(N)

            pid_history = []
            node_vis, node_tot = set(), set()
            node_vis.update(label_set(nvs[0], layout, all_nodes))
            node_tot.update(node_vis | label_set(nos[0], layout, all_nodes))
            r1_last120, r2_last120 = deque(maxlen=120), deque(maxlen=120)

            for i in range(1, len(smoothed)):
                node_vis.update(label_set(nvs[i], layout, all_nodes))
                node_tot.update(label_set(nvs[i], layout, all_nodes))
                node_tot.update(label_set(nos[i], layout, all_nodes))
                r1_last120.append(r1_idxs[i])
                r2_last120.append(r2_idxs[i])

                _, curr_label = smoothed[i]
                if curr_label == start_label:
                    continue

                curr_t, _ = smoothed[i]
                curr_v, curr_s = len(nds[i]), ns[i]
                curr_dt, curr_dv, curr_ds = curr_t - start_t, curr_v - start_v, curr_s - start_s
                curr_Rt = get_robot_snapshot(r1_last120, r2_last120, Dij)

                start_idx, curr_idx = node_idx_map[start_label], node_idx_map[curr_label]
                R_eff = 0.5 * (start_Rt[start_idx] + curr_Rt[start_idx])

                # update visibility matrix
                vis[start_idx, [node_idx_map[n] for n in node_vis]] += 1
                tot[start_idx, [node_idx_map[n] for n in node_tot]] += 1
                node_vis.clear(); node_tot.clear()

                if A_np[start_idx, curr_idx] > 0:
                    record_transition(start_label, curr_dt, curr_ds, curr_dv, R_eff, curr_Rt,
                                      r1_labels[i], r2_labels[i])
                    start_label, start_t, start_v, start_s, start_Rt = curr_label, curr_t, curr_v, curr_s, curr_Rt
                else:
                    path = shortest_paths.get(start_idx, {}).get(curr_idx)
                    if path and len(path) == 3:
                        a, b, c = [node_order[j] for j in path]
                        record_transition(a, max(min_dt, curr_dt - min_dt), curr_ds, curr_dv, R_eff,
                                          curr_Rt, r1_labels[i], r2_labels[i])

                        b_idx = node_idx_map[b]
                        R_eff = 0.5 * (start_Rt[b_idx] + curr_Rt[b_idx])
                        record_transition(b, min_dt, 0, 0, R_eff, curr_Rt,
                                          r1_labels[i], r2_labels[i])
                        start_label, start_t, start_v, start_s, start_Rt = c, curr_t, curr_v, curr_s, curr_Rt

            # Final event
            final_time, _ = smoothed[-1]
            if final_time > start_t:
                curr_dt = final_time - start_t
                curr_dv = len(nds[-1]) - start_v
                curr_ds = ns[-1] - start_s
                curr_Rt = get_robot_snapshot(r1_last120, r2_last120, Dij)
                start_idx = node_idx_map[start_label]
                R_eff = 0.5 * (start_Rt[start_idx] + curr_Rt[start_idx])
                record_transition(start_label, curr_dt, curr_ds, curr_dv, R_eff, curr_Rt,
                                  r1_labels[-1], r2_labels[-1])

            full_history.append(pid_history)

    # Post-Processing
    finite_vals = [v for v in dt_minself.values() if np.isfinite(v)]
    mean_self_dt = np.mean(finite_vals) if finite_vals else min_dt
    for k in dt_minself:
        if not np.isfinite(dt_minself[k]):
            dt_minself[k] = mean_self_dt

    prob = np.zeros_like(vis, dtype=np.float32)
    np.divide(vis + alpha, tot + 2 * alpha, out=prob, where=tot > 0)

    node_max_dv = {node: max(total_dv_node[node].values(), default=0)
                   for node in node_order}

    stats = dict(
        dt=dt_node,
        dv=dv_node,
        ds=ds_node,
        p_seen=prob,
        min_st=dt_minself,
        n_tot=n_tot,
        R_eff=R_node,
        max_dv=node_max_dv,
    )
    return stats, full_history

def summarize_from_hist(hist, node_order):
    ans_nodes = []
    ans_victs = []
    ans_time = []
    ans_shots = []
    ans_start = []

    # aggregate totals
    vpn = defaultdict(float)   # victims per node
    tpn = defaultdict(float)   # time per node
    spn = defaultdict(float)   # shots per node
    npn = defaultdict(int)     # visits per node

    tot_vict = 0.0
    tot_time = 0.0
    tot_shot = 0.0
    tot_visit = 0

    for pid_history in hist:
        if not pid_history:
            ans_nodes.append((0, 0))
            ans_victs.append(0)
            ans_time.append(0.0)
            ans_shots.append(0)
            ans_start.append(None)
            continue

        nodes   = [entry[0] for entry in pid_history]
        times   = [entry[1] for entry in pid_history]
        shots   = [entry[2] for entry in pid_history]
        victims = [entry[3] for entry in pid_history]

        # summary metrics
        total_time  = sum(times)
        total_shots = sum(shots)
        total_victs = sum(victims)

        ans_nodes.append((len(nodes), len(set(nodes))))
        ans_victs.append(total_victs)
        ans_time.append(total_time)
        ans_shots.append(total_shots)
        ans_start.append(nodes[0])

        # per-node aggregation
        for (node, dt, ds, dv, *_) in pid_history:
            vpn[node] += dv
            tpn[node] += dt
            spn[node] += ds
            npn[node] += 1

            tot_vict += dv
            tot_time += dt
            tot_shot += ds
            tot_visit += 1

    # normalize percentages
    ans_vpn = {}
    ans_tpn = {}
    ans_spn = {}
    ans_npn = {}

    for node in node_order:
        ans_vpn[node] = vpn.get(node, 0.0) / tot_vict if tot_vict > 0 else 0.0
        ans_tpn[node] = tpn.get(node, 0.0) / tot_time if tot_time > 0 else 0.0
        ans_spn[node] = spn.get(node, 0.0) / tot_shot if tot_shot > 0 else 0.0
        ans_npn[node] = npn.get(node, 0)   / tot_visit if tot_visit > 0 else 0.0

    return (
        ans_start,              # 0
        np.array(ans_nodes),    # 1 (total, unique)
        np.array(ans_victs),    # 2
        np.array(ans_time),     # 3
        np.array(ans_shots),    # 4
        ans_vpn,                # 5
        ans_tpn,                # 6
        ans_spn,                # 7
        ans_npn,                # 8
    )

def min_trans_dt(agent = "shooter"):
    path    = ENV_DIR / "columbine_dist-from-node-to-node.xlsx"
    df      = pd.read_excel(path, index_col = 0)
    dist    = df.iloc[:, 0].to_dict()   # PU (cells)
    conv    = 699.23 / 78               # GU / PU, ~ 8.96

    if agent == "shooter":
        max_speed = 32.2                # GU / s (from 95p of E2-E5)

    else:
        max_speed = 29.1               # GU / s (from 95p of E4-E5)

    spd = max_speed# * eff_speed * eff_paths   # GU / s
    ans = {}

    for k,val in dist.items():
        if isinstance(k, str) and k.startswith("("):
            k = eval(k)
        val *= conv                 # GU
        ans[k] = val / spd          # s
    return ans