# -----------------------------------------------------------------------------
# CA McClurg
# This class computes per-node, per-group, and global statistics (dt, ds, dv)
# and offers sampling utilities for simulation rollouts.
# -----------------------------------------------------------------------------

import numpy as np
from src.utils.des import moment_matched_normal, moment_matched_lognormal

MIN_TIME = 0.5
MIN_TRIM = 3


class NodeStats:
    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_empirical_dict(cls, node_order, stats_dict, min_pt, node_type_map):
        return cls(
            node_order=node_order,
            stats_dict=stats_dict,
            node_type_map=node_type_map,
            min_pt=min_pt,
            min_st=stats_dict.get("min_st", None),
        )

    def __init__(self, node_order, stats_dict, node_type_map, *, min_st=None, min_pt=None):
        """
        Parameters
        ----------
        node_order : list
            Ordered node IDs.
        stats_dict : dict
            Must include {'dt', 'dv', 'ds', 'p_seen', 'n_tot', 'R_eff', 'max_dv'}.
        node_type_map : dict
            One-hot or categorical mapping of node type/group.
        """
        # core inputs
        self.node_order = tuple(node_order)
        self.N = len(node_order)
        self.node_type_map = node_type_map
        self.path_min_time = min_pt or {}
        self.self_min_time = min_st or {}

        # extract stats
        self.dt = stats_dict.get("dt", {})
        self.ds = stats_dict.get("ds", {})
        self.dv = stats_dict.get("dv", {})
        self.r_eff = stats_dict.get("R_eff", {})
        self.r_eff_copy = {k: np.copy(v) for k, v in self.r_eff.items()}
        self.p_seen = stats_dict.get("p_seen", np.zeros((self.N, self.N)))
        self.n_tot = stats_dict.get("n_tot", {})

        # # use this one for max dv per event (max total, degrades in sim)
        # self.nodal_alive_init = stats_dict.get("max_dv", {})

        # use this one for a global integer
        self.global_alive_init = int(np.nanmax([np.nanmax(v) for v in self.n_tot.values() if len(v)]))

        # use this one for a max dv per event at a node (no degradation in sim)
        self.nodal_alive_init = {k: np.max(self.dv[k]) if len(self.dv[k]) > 0 else 0.0 for k in self.dv}

        # compute modulation + baseline stats
        self._compute_k_mod()
        self._filter_no_robot_samples()
        self._compute_stats(self._flatten_data())

    def _compute_k_mod(self, lambda_reg=10.0):
        """Compute weighted regression-style modulation slopes k_x using all data."""
        eps = 1e-6
        kt_eps = 1e-3
        self.k_mod = {}
        self.rate_mod = {}

        def safe_k(R, x):
            R = np.asarray(R, float)
            x = np.asarray(x, float)
            n = len(R)
            if n < 2 or np.std(R) < eps or np.std(x) < eps:
                return 0.0

            # mean-center
            Rc = R - np.mean(R)
            xc = x - np.mean(x)

            # weighted ridge-style shrinkage
            cov = np.sum(Rc * xc)
            varR = np.sum(Rc ** 2)
            k_raw = cov / varR
            w = n / (n + lambda_reg)
            return w * k_raw  # shrink toward 0 when n is small

        vr_raw = []
        for n in self.node_order:
            R = np.asarray(self.r_eff.get(n, []), float)

            kt_curr = safe_k(R, np.asarray(self.dt.get(n, []), float))
            ks_curr = safe_k(R, np.asarray(self.ds.get(n, []), float))
            kv_curr = safe_k(R, np.asarray(self.dv.get(n, []), float))

            self.k_mod[n] = {"dt": kt_curr, "ds": ks_curr, "dv": kv_curr}

            den = max(abs(kt_curr), kt_eps)
            vr_curr = kv_curr / (np.sign(kt_curr) * den) if abs(kt_curr) > 0 else 0.0
            vr_raw.append(vr_curr)

        vr_raw = np.array(vr_raw)
        sigma = np.std(vr_raw)
        if sigma < eps:
            vr = np.zeros_like(vr_raw)
        else:
            vr = np.tanh((vr_raw - np.median(vr_raw)) / (2 * sigma))

        self.rate_mod = {n: vr[i] for i, n in enumerate(self.node_order)}


    def _filter_no_robot_samples(self):
        """Remove samples where R_eff > 0 for baseline statistics."""
        for n in self.node_order:
            R = np.asarray(self.r_eff.get(n, []), float)
            mask = R <= 0.0
            self.dt[n] = np.asarray(self.dt.get(n, []), float)[mask].tolist()
            self.ds[n] = np.asarray(self.ds.get(n, []), float)[mask].tolist()
            self.dv[n] = np.asarray(self.dv.get(n, []), float)[mask].tolist()
            self.r_eff[n] = R[mask].tolist()

    def _flatten_data(self):
        """Flatten per-node arrays for global/group statistics."""
        flat = {k: [] for k in ["dt", "ds", "dv", "re", "nodes", "groups"]}
        for n in self.node_order:
            g = self.get_group_id(n)
            flat["dt"].extend(self.dt.get(n, []))
            flat["ds"].extend(self.ds.get(n, []))
            flat["dv"].extend(self.dv.get(n, []))
            flat["re"].extend(self.r_eff.get(n, []))
            flat["nodes"].extend([n] * len(self.dt.get(n, [])))
            flat["groups"].extend([g] * len(self.dt.get(n, [])))
        return {f"flat_{k}": np.array(v) for k, v in flat.items()}

    def _compute_stats(self, flat):
        """Compute node-, group-, and global-level means and stds."""
        keys = ["dt", "ds", "dv", "re"]

        # global
        self.global_mean = {k: np.nanmean(flat[f"flat_{k}"]) for k in keys}
        self.global_std = {k: np.nanstd(flat[f"flat_{k}"]) for k in keys}

        # per-node
        self.node_stats = {}
        for n in self.node_order:
            self.node_stats[n] = {}
            for k, arr in zip(keys, [self.dt, self.ds, self.dv, self.r_eff]):
                vals = np.asarray(arr.get(n, []), float)
                self.node_stats[n][k] = {
                    "mean": np.nanmean(vals) if len(vals) else 0.0,
                    "std": np.nanstd(vals) if len(vals) else 0.0,
                    "n": len(vals),
                }

        # per-group
        self.group_stats = {}
        groups = np.unique(flat["flat_groups"])
        for g in groups:
            mask = flat["flat_groups"] == g
            self.group_stats[g] = {
                k: {
                    "mean": np.nanmean(flat[f"flat_{k}"][mask]),
                    "std": np.nanstd(flat[f"flat_{k}"][mask]),
                    "n": np.sum(mask),
                }
                for k in keys
            }

    # ------------------------------------------------------------------
    # Utilities / sampling
    # ------------------------------------------------------------------
    def get_group_id(self, node):
        """Return integer group index for a given node."""
        return np.nonzero(self.node_type_map[node])[0][0]

    def _sample_with_fallback(self, node, key, *, lo=-np.inf, hi=np.inf, k=1, noise_scale=1.0):
        """Sample values using node-, group-, or global-level fallback."""
        def try_sample(mean, std):
            if np.isfinite(std) and std > 0:
                return moment_matched_normal(np.clip(mean, lo, hi), lo, hi, k, std * noise_scale)

        ns = self.node_stats[node][key]
        if ns["n"] >= MIN_TRIM:
            val = try_sample(ns["mean"], ns["std"])
            if val is not None:
                return val

        gs = self.group_stats.get(self.get_group_id(node), {}).get(key, {})
        if gs.get("n", 0) >= MIN_TRIM:
            val = try_sample(gs["mean"], gs["std"])
            if val is not None:
                return val

        std = self.global_std.get(key, 1e-3)
        mean = np.clip(self.global_mean.get(key, 0.0), lo, hi)
        return moment_matched_normal(mean, lo, hi, k, std * noise_scale)

    def _sample_lognormal_with_fallback(self, node, key, *, lo=0, hi=np.inf):
        """
        Sample from a moment-matched lognormal distribution using node→group→global fallback.
        The resulting distribution preserves the empirical mean and variance (μ, σ²)
        while ensuring strictly positive and right-skewed samples.
        """
        # ---- node-level stats ----
        ns = self.node_stats[node][key]
        mu, sigma, n = ns["mean"], ns["std"], ns["n"]
        if n >= MIN_TRIM and sigma > 0:
            val = moment_matched_lognormal(mu, sigma)
            return np.clip(val, lo, hi)

        # ---- group-level fallback ----
        gid = self.get_group_id(node)
        gs = self.group_stats.get(gid, {}).get(key, {})
        if gs.get("n", 0) >= MIN_TRIM and gs.get("std", 0) > 0:
            mu_g, sigma_g = gs["mean"], gs["std"]
            val = moment_matched_lognormal(mu_g, sigma_g)
            return np.clip(val, lo, hi)

        # ---- global fallback ----
        mu_g = self.global_mean.get(key, 1.0)
        sigma_g = self.global_std.get(key, 1e-3)
        val = moment_matched_lognormal(mu_g, sigma_g)
        return np.clip(val, lo, hi)

    def safe_mean_with_fallback(self, node, key):
        """Stable mean lookup with node→group→global fallback."""
        ns = self.node_stats[node][key]
        if ns["n"] >= MIN_TRIM and np.isfinite(ns["mean"]) and ns["mean"] > 0:
            return ns["mean"]

        gs = self.group_stats.get(self.get_group_id(node), {}).get(key, {})
        if gs.get("n", 0) >= MIN_TRIM and np.isfinite(gs["mean"]) and gs["mean"] > 0:
            return gs["mean"]

        gm = self.global_mean.get(key, 1.0)
        return gm if np.isfinite(gm) and gm > 0 else 1.0

    def sample_dt(self, node, **kw): return self._sample_with_fallback(node, "dt", **kw)
    def sample_ds(self, node, **kw): return self._sample_with_fallback(node, "ds", **kw)
    def sample_dv(self, node, **kw): return self._sample_with_fallback(node, "dv", **kw)

    def sample_dv_lognorm(self, node, **kw):
        return self._sample_lognormal_with_fallback(node, "dv", **kw)

    def sample_ds_lognorm(self, node, **kw):
        return self._sample_lognormal_with_fallback(node, "ds", **kw)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def min_time_between(self, start, via, end): return self.path_min_time.get((start, via, end))
    def min_time_inout(self, node): return self.self_min_time.get(node, MIN_TIME)
    def mean_victims(self, node): return self.node_stats[node]["dv"]["mean"]
    def mean_time(self, node): return self.node_stats[node]["dt"]["mean"]
    def mean_shots(self, node): return self.node_stats[node]["ds"]["mean"]

    def std_victims(self, node): return self.node_stats[node]["dv"]["std"]
    def std_time(self, node): return self.node_stats[node]["dt"]["std"]
    def std_shots(self, node): return self.node_stats[node]["ds"]["std"]

    def get_grp_means(self, node):
        gid = self.get_group_id(node)
        grp = self.group_stats.get(gid, {})
        return {k: grp[k]["mean"] for k in ["dt", "ds", "dv", "re"] if k in grp}

    def get_global_means(self):
        return {k: self.global_mean.get(k, 1.0) for k in ["dt", "ds", "dv"]}

    def get_grp_vects(self, node): return self.node_type_map[node]
    def sample_visible_nodes(self, node, *, rng=np.random):
        """Return visible nodes sampled using p_seen probabilities."""
        if node not in self.node_order:
            raise ValueError(f"Node '{node}' not found in node_order.")
        idx = self.node_order.index(node)
        mask = rng.random(self.N) < self.p_seen[idx]
        return {n for n, m in zip(self.node_order, mask) if m}
