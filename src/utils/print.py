import warnings
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, ttest_ind, levene, spearmanr

warnings.filterwarnings("ignore", message="Precision loss occurred")

def model_key(record):
    """
    Produce canonical model key like 'samp_G' or 'samp_B'.
    """
    suffix = "G" if record.get("use_gnn") else "B"
    return f"{record['model_type']}_{suffix}"

def mean_std(arr):
    """Return formatted mean ± sd."""
    arr = np.asarray(arr)
    return f"{np.mean(arr):5.1f} ± {np.std(arr):4.1f}"

def mape(emp, pred, mode="paired"):
    """
    Compute error.

    mode = "means"   → percent error of means
    mode = "paired"  → absolute deviation in percentage points
    """
    emp = np.asarray(emp, float)
    pred = np.asarray(pred, float)

    if mode == "means":
        mu_e = np.mean(emp)
        mu_p = np.mean(pred)
        return np.nan if mu_e == 0 else 100 * abs(mu_p - mu_e) / abs(mu_e)

    if mode == "paired":
        if emp.shape != pred.shape:
            raise ValueError("Paired MAPE expects same shape.")
        return np.mean(np.abs(pred - emp))

    raise ValueError("Invalid mode: choose 'means' or 'paired'.")

def welch_p(emp, pred):
    """Welch t-test p-value."""
    _, p = ttest_ind(emp, pred, equal_var=False)
    return p

def levene_p(emp, pred):
    """Return Levene test p-value for equality of variances."""
    _, p = levene(emp, pred, center='median')
    return p

def safe_r2(a, b):
    """Return Pearson r² or nan."""
    a = np.asarray(a)
    b = np.asarray(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r, _ = pearsonr(a, b)
    return r ** 2

def jsd(P, Q, eps=1e-12):
    """
    Jensen-Shannon distance (base-2).
    Inputs P, Q must be iterable probabilities that sum to the same value.
    Output is in [0,1].
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)

    # avoid zero problems
    P = P + eps
    Q = Q + eps
    P /= P.sum()
    Q /= Q.sum()

    M = 0.5 * (P + Q)

    KL_PM = np.sum(P * np.log2(P / M))
    KL_QM = np.sum(Q * np.log2(Q / M))

    return float(np.sqrt(0.5 * (KL_PM + KL_QM)))

def sprmn(dx1, dx2):
    rho, _ = spearmanr(dx1, dx2)
    return rho

def extract_dt_ds_dv(res):
    all_dt = []
    all_dv = []
    all_ds = []
    for r in res:
        for node, dt, ds, dv, *_ in r["full_history"]:
            all_dt.append(dt)
            all_dv.append(dv)
            all_ds.append(ds)
    return np.array(all_dt), np.array(all_ds), np.array(all_dv)

def extract_dt_ds_dv_from_hist(ctx_hist):
    all_dt, all_ds, all_dv = [], [], []
    for traj in ctx_hist:             # each participant
        for (_, dt, ds, dv, *_) in traj:
            all_dt.append(dt)
            all_ds.append(ds)
            all_dv.append(dv)
    return np.array(all_dt), np.array(all_ds), np.array(all_dv)

def aggregate_node_percentages(results, model_types):
    """
    Returns dict:
        model_name -> (pct_time, pct_vict, pct_shot, pct_visit)

    Each mapping is dict: node -> fraction.
    """
    stats = {}

    for m in model_types:
        time_sum  = defaultdict(float)
        vict_sum  = defaultdict(float)
        shot_sum  = defaultdict(float)
        visit_cnt = defaultdict(int)

        tot_t = tot_v = tot_s = tot_n = 0

        for r in [r for r in results if model_key(r) == m]:
            for node, dt, ds, dv, *_ in r["full_history"]:
                time_sum[node]  += dt
                shot_sum[node]  += ds
                vict_sum[node]  += dv
                visit_cnt[node] += 1

                tot_t += dt
                tot_s += ds
                tot_v += dv
                tot_n += 1

        pct_time  = {k: v / tot_t for k, v in time_sum.items()}  if tot_t else {}
        pct_vict  = {k: v / tot_v for k, v in vict_sum.items()}  if tot_v else {}
        pct_shot  = {k: v / tot_s for k, v in shot_sum.items()}  if tot_s else {}
        pct_visit = {k: v / tot_n for k, v in visit_cnt.items()} if tot_n else {}

        stats[m] = (pct_time, pct_vict, pct_shot, pct_visit)

    return stats

def print_r2_row(metrics, model_types, width=6):
    """
    metrics[model] = (time[], vict[], shot[], visit[])
    """

    print(f"{'Pearson r²':<20} | ", end="")

    blocks = []
    for i in range(4):  # time, vict, shot, node
        block = ["--"]
        emp = metrics["EM"][i]
        for m in model_types:
            r2 = safe_r2(emp, metrics[m][i])
            block.append(f"{r2:.3f}" if np.isfinite(r2) else "--")
        blocks.append("".join(f"{v:>{width}}" for v in block))

    print(" | ".join(blocks))

def print_mape_row(metrics, model_types, width=6):
    print(f"{'MAPE':<20} | ", end="")

    blocks = []
    for i in range(4):
        block = ["--"]
        emp = metrics["EM"][i]
        for m in model_types:
            val = mape(emp, metrics[m][i], "paired")
            block.append(f"{val:6.1f}" if np.isfinite(val) else "--")
        blocks.append("".join(f"{v:>{width}}" for v in block))

    print(" | ".join(blocks))

def print_jsd_row(metrics, model_types, width=6):
    """
    JSD between EM and each model for:
        nodes, time, shots, victims
    """
    print(f"{'JSD':<20} | ", end="")

    blocks = []
    for i in range(4):  # nodes, time, shots, victims
        block = ["--"]

        emp = np.asarray(metrics["EM"][i], float)

        for m in model_types:
            pred = np.asarray(metrics[m][i], float)
            val = jsd(emp, pred)
            block.append(f"{val:.3f}")

        blocks.append("".join(f"{v:>{width}}" for v in block))

    print(" | ".join(blocks))

def print_summary_stats(results, empirical, ctx_hist):
    """
    empirical corresponds to ctx.eval_results.
    Returns mean values of empirical (time, shots, victims).
    """
    emp_nodes, emp_unique, emp_vict, emp_time, emp_shot, *_ = empirical

    model_types = sorted({model_key(r) for r in results})
    grouped = {m: [r for r in results if model_key(r) == m] for m in model_types}

    def unpack(res):
        return (
            [r["victims"] for r in res],
            [len(r["nodes"]) for r in res],
            [len(set(r["nodes"])) for r in res],
            [r["time"] for r in res],
            [r["shots"] for r in res],
        )

    unpacked = {m: unpack(grouped[m]) for m in model_types}

    # === TEMPORAL FIT METRICS ===
    temporal_fit_vict = {}
    temporal_fit_shot = {}

    # empirical DT/DS/DV arrays
    emp_dt, emp_ds, emp_dv = extract_dt_ds_dv_from_hist(ctx_hist)

    # Victim dependence on time
    rho_emp_v, _ = spearmanr(emp_dt, emp_dv)
    temporal_fit_vict["EM"] = rho_emp_v

    # Shot dependence on time
    rho_emp_s, _ = spearmanr(emp_dt, emp_ds)
    temporal_fit_shot["EM"] = rho_emp_s

    # For each model
    for m in model_types:
        dt_m, ds_m, dv_m = extract_dt_ds_dv(grouped[m])

        rho_m_v, _ = spearmanr(dt_m, dv_m)
        rho_m_s, _ = spearmanr(dt_m, ds_m)

        temporal_fit_vict[m] = rho_m_v
        temporal_fit_shot[m] = rho_m_s


    empirical_dict = {
        "victims": emp_vict,
        "nodes": emp_nodes,
        "unique": emp_unique,
        "time": emp_time,
        "shots": emp_shot,
    }

    fields = [
        ("Nodes", "nodes"),
        ("Time", "time"),
        ("Shots", "shots"),
        ("Victims", "victims"),
    ]

    print("\n" + f"{'SUMMARY STATISTICS':^90}")
    header = " | ".join(f"{m:^12}" for m in ["Empirical"] + model_types)
    print(f"{'Metric':<12} | {header}")
    print("-" * (15 + (len(model_types) + 1) * 15))

    means_out = []

    for label, key in fields:
        idx = ["victims", "nodes", "unique", "time", "shots"].index(key)
        emp_vals = empirical_dict[key]

        print(f"{label:<12} | {mean_std(emp_vals):>12} | " +
              " | ".join(f"{mean_std(unpacked[m][idx]):>12}" for m in model_types))

        print(f"{'Welch p':<12} | {'--':>12} | " +
              " | ".join(f"{welch_p(emp_vals, unpacked[m][idx]):12.3f}" for m in model_types))

        print(f"{'Levene p':<12} | {'--':>12} | " +
              " | ".join(f"{levene_p(emp_vals, unpacked[m][idx]):12.3f}" for m in model_types))

        print(f"{'MAPE (%)':<12} | {'--':>12} | " +
              " | ".join(f"{mape(emp_vals, unpacked[m][idx], 'means'):12.1f}" for m in model_types))

        print()

        if key in ("time", "shots", "victims"):
            means_out.append(np.mean(emp_vals))

    print(f"{'Temporal ρ (vict)':<20} | {temporal_fit_vict['EM']:>12.3f} | " +
          " | ".join(f"{temporal_fit_vict[m]:12.3f}" for m in model_types))

    print(f"{'Temporal ρ (shots)':<20} | {temporal_fit_shot['EM']:>12.3f} | " +
          " | ".join(f"{temporal_fit_shot[m]:12.3f}" for m in model_types))


    return means_out

def print_node_level(results, ctx, empirical_means):
    """
    Print NODES (%), TIME (%), SHOTS (%), VICT (%) for each node.
    Includes Pearson r² and MAPE.
    """

    emp = ctx.eval_results
    node_order = ctx.node_order
    node_names = ctx.node_names

    LEFT_WIDTH = 20

    # empirical distributions
    _, _, _, _, _, pct_emp_vict, pct_emp_time, pct_emp_shot, pct_emp_visit = emp

    model_types = sorted({model_key(r) for r in results})
    pct_by_model = aggregate_node_percentages(results, model_types)

    # ---------- formatting helpers ----------
    def build_col_header(title, n_models, width=6):
        total_width = (n_models + 1) * width
        return f"{title:^{total_width}}"

    def fmt6_header(label):
        if len(label) > 6:
            label = label[:6]
        return f"{label:>6}"

    def fmt6(v):
        s = f"{v:.1f}"
        if len(s) > 6:
            s = s[:6]
        return f"{s:>6}"

    # ---------- dynamic headers ----------
    n_models = len(model_types)

    line1 = (
        "\n" +
        f"{'Node  Name':<{LEFT_WIDTH}}| "
        f"{build_col_header('NODES (%)', n_models)} | "
        f"{build_col_header('TIME (%)',  n_models)} | "
        f"{build_col_header('SHOTS (%)', n_models)} | "
        f"{build_col_header('VICT (%)',  n_models)}"
    )

    hdr = "".join(fmt6_header(m[:2].upper()) for m in ["EM"] + model_types)

    line2 = f"{'':<{LEFT_WIDTH}}| {hdr} | {hdr} | {hdr} | {hdr}"

    print(line1)
    print(line2)
    print("-" * len(line2))

    # ---------- collection for r² + MAPE ----------
    metrics = {m: ([], [], [], []) for m in ["EM"] + model_types}
    #                  (time, vict, shot, nodes)
    #   BUT printed order will be nodes → time → shots → vict

    for node in node_order:
        # empirical values
        et = pct_emp_time.get(node, 0)  * 100
        ev = pct_emp_vict.get(node, 0)  * 100
        es = pct_emp_shot.get(node, 0)  * 100
        en = pct_emp_visit.get(node, 0) * 100

        metrics["EM"][0].append(en)
        metrics["EM"][1].append(et)
        metrics["EM"][2].append(es)
        metrics["EM"][3].append(ev)

        # rows in *display order*
        row_n = [en]
        row_t = [et]
        row_s = [es]
        row_v = [ev]

        # fill model predictions
        for m in model_types:
            pct_t, pct_v, pct_s, pct_n = pct_by_model[m]

            mt = pct_t.get(node, 0) * 100
            mv = pct_v.get(node, 0) * 100
            ms = pct_s.get(node, 0) * 100
            mn = pct_n.get(node, 0) * 100

            row_n.append(mn)
            row_t.append(mt)
            row_s.append(ms)
            row_v.append(mv)

            metrics[m][0].append(mn)
            metrics[m][1].append(mt)
            metrics[m][2].append(ms)
            metrics[m][3].append(mv)

        NODE_FIELD = 6
        NAME_FIELD = LEFT_WIDTH - NODE_FIELD + 1

        left = (
            f"{node:<{NODE_FIELD}}"
            f"{node_names.get(node,'unknown'):<{NAME_FIELD}}"
        )
        print(
            f"{left:<{LEFT_WIDTH}}| "
            f"{''.join(fmt6(x) for x in row_n)} | "
            f"{''.join(fmt6(x) for x in row_t)} | "
            f"{''.join(fmt6(x) for x in row_s)} | "
            f"{''.join(fmt6(x) for x in row_v)}"
        )

    print_r2_row(metrics, model_types)
    print_mape_row(metrics, model_types)
    print_jsd_row(metrics, model_types)

def print_type_level(node_names, pct_by_model, emp, empirical_means):
    """
    Aggregates TIME/VICT/SHOTS/NODES by semantic type.
    """
    _, _, _, _, _, pct_emp_vict, pct_emp_time, pct_emp_shot, pct_emp_visit = emp

    model_types = list(pct_by_model.keys())

    # semantic categories
    categories = ["outside", "entrance", "cafe", "stairs",
                  "classroom", "library", "hallway"]

    def group(pct_map):
        out = defaultdict(float)
        for node, val in pct_map.items():
            label = next((c for c in categories if c in node_names[node].lower()), "other")
            out[label] += val
        return out

    def build_col_header(title, n_models, width=6):
        total_width = (n_models + 1) * width  # EM + model_cols
        return f"{title:^{total_width}}"

    def fmt6_header(label):
        if len(label) > 6:
            label = label[:6]
        return f"{label:>6}"

    agg = {
        "EM": (
            group(pct_emp_time),
            group(pct_emp_vict),
            group(pct_emp_shot),
            group(pct_emp_visit),
        )
    }

    for m in model_types:
        pct_t, pct_v, pct_s, pct_n = pct_by_model[m]
        agg[m] = (group(pct_t), group(pct_v), group(pct_s), group(pct_n))

    all_labels = sorted({lbl for x in agg.values() for lbl in x[0]})

    n_models = len(model_types)  # e.g., 5 → EM + 5 columns = 6

    line1 = (
        "\n" + f"{'Type':<20} | "
        f"{build_col_header('NODES (%)', n_models)} | "
        f"{build_col_header('TIME (%)',  n_models)} | "
        f"{build_col_header('SHOTS (%)', n_models)} | "
        f"{build_col_header('VICT (%)',  n_models)}"
    )

    hdr = "".join(fmt6_header(m[:2].upper()) for m in ["EM"] + model_types)
    line2 = f"{'':<20} | {hdr} | {hdr} | {hdr} | {hdr}"

    print(line1)
    print(line2)
    print("-" * len(line2))

    def fmt6(v):
        s = f"{v:.1f}"
        if len(s) > 6:
            s = s[:6]
        return f"{s:>{6}}"

    metrics = {m: ([], [], [], []) for m in ["EM"] + model_types}

    for label in all_labels:
        row_t = []
        row_v = []
        row_s = []
        row_n = []

        for m in ["EM"] + model_types:
            pt, pv, ps, pn = agg[m]

            row_t.append(pt.get(label, 0) * 100)
            row_v.append(pv.get(label, 0) * 100)
            row_s.append(ps.get(label, 0) * 100)
            row_n.append(pn.get(label, 0) * 100)

        for i, arr in enumerate([row_t, row_v, row_s, row_n]):
            for j, m in enumerate(["EM"] + model_types):
                metrics[m][i].append(arr[j])

        print(
            f"{label:<20} | "
            f"{''.join(fmt6(x) for x in row_n)} | "
            f"{''.join(fmt6(x) for x in row_t)} | "
            f"{''.join(fmt6(x) for x in row_s)} | "
            f"{''.join(fmt6(x) for x in row_v)}"
        )

    print_r2_row(metrics, model_types)
    print_mape_row(metrics, model_types)

def print_results(results, ctx):
    """
    Full pipeline:
     1) Summary stats
     2) Node-level TIME/VICT/SHOTS/NODES
     3) Type-level TIME/VICT/SHOTS/NODES
    """
    empirical = ctx.eval_results

    empirical_means = print_summary_stats(results, empirical, ctx.hist)

    model_types = sorted({model_key(r) for r in results})
    pct_by_model = aggregate_node_percentages(results, model_types)

    print_node_level(results, ctx, empirical_means)
    print_type_level(ctx.node_names, pct_by_model, empirical, empirical_means)
