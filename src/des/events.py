# -----------------------------------------------------------------------------
# CA McClurg
# This class is a container for running node-level simulations using different
# strategies.
# -----------------------------------------------------------------------------

import numpy as np

USE_LEVEL = "nodal"
USE_COUPLING = False

class SimEvents:
    """Container for running node-level simulations using different strategies."""

    def __init__(self, ctx, ep_nodes, ep_reff, r1_nodes, r2_nodes, force_int=False):
        """
        Parameters
        ----------
        ctx : object
            Simulation context (contains stats, models, etc.)
        ep_nodes : list
            Ordered node sequence for this episode.
        force_int : bool, default=True
            If True, force dv and ds to be integers via rounding/flooring.
            If False, leave them as floats.
        """
        self.ctx = ctx
        self.nodes = ep_nodes
        self.reff = ep_reff
        self.stats = ctx.stats
        self.force_int = force_int
        self.r1_nodes = r1_nodes
        self.r2_nodes = r2_nodes
        self.use_gnn = (len(ep_nodes) == ctx.max_steps+1)
        self.global_alive = self.stats.global_alive_init
        self.nodal_alive = self.stats.nodal_alive_init.copy()
    # ------------------------------------------------------------------
    # External entry point
    # ------------------------------------------------------------------
    def simulate(self, model_type, use_coupling = USE_COUPLING, use_level = USE_LEVEL):
        """Run the simulation using the specified model_type."""
        method_map = {
            "mean": self.mean,
            "samp": self.samp}

        if model_type not in method_map:
            raise ValueError(f"[ERROR] model_type={model_type} not recognized!")
        return method_map[model_type](use_coupling, use_level)

    # ------------------------------------------------------------------
    # Strategy dispatchers
    # ------------------------------------------------------------------
    def mean(self, use_coupling, use_level):
        return self._run(use_coupling, use_level, strategy="X_means")

    def samp(self, use_coupling, use_level):
        return self._run(use_coupling, use_level, strategy="X_sampling")

    # ------------------------------------------------------------------
    # Internal shared runner
    # ------------------------------------------------------------------
    def _run(self, use_coupling, use_level, strategy):
        total_time, total_vict, total_shot = 0.0, 0, 0
        history = []
        stats = self.stats
        final_time = self.ctx.final_time
        force_int = self.force_int

        for i, curr in enumerate(self.nodes):
            prev = self.nodes[i - 1] if i > 0 else None
            next_ = self.nodes[i + 1] if i < len(self.nodes) - 1 else None

            # physical bounds
            if any([prev is None, next_ is None, prev == next_]):
                physical_min_time = stats.min_time_inout(curr)
            else:
                physical_min_time = stats.min_time_between(prev, curr, next_)
            physical_max_time = final_time

            physical_min_vict = 0.0
            physical_max_vict = self.nodal_alive.get(curr, 0.0)

            physical_min_shot = 0.0
            physical_max_shot = np.inf

            # leveling effect
            if use_level == "global":
                g = stats.get_global_means()
            elif use_level == "group":
                g = stats.get_grp_means(curr)
            else:
                g = dict()
                g['dt'] = float(stats.mean_time(curr))
                g['ds'] = float(stats.mean_shots(curr))
                g['dv'] = float(stats.mean_victims(curr))

            # pick strategy
            if strategy == "X_means" and not use_coupling:
                dt = np.clip(float(g['dt']), physical_min_time, physical_max_time)
                dv = np.clip(float(g['dv']), physical_min_vict, physical_max_vict)
                ds = np.clip(float(g['ds']), physical_min_shot, physical_max_shot)

            elif strategy == "X_means" and use_coupling:
                s, v, t = g['ds'], g['dv'], g['dt']
                mu_vr = v/t if t> 0 else 0.0
                mu_sr = s/t if t> 0 else 0.0
                dt = np.clip(float(g['dt']), physical_min_time, physical_max_time)
                dv = float(np.clip(mu_vr*dt, physical_min_vict, physical_max_vict))
                ds = float(np.clip(mu_sr*dt, physical_min_shot, physical_max_shot))

            elif strategy == "X_sampling" and not use_coupling:
                # sample time (truncated normal, symmetric about mean)
                mu_dt = np.clip(float(g['dt']), physical_min_time, physical_max_time)
                del_dt = min(mu_dt - physical_min_time, physical_max_time - mu_dt)
                min_dt, max_dt = mu_dt - del_dt, mu_dt + del_dt
                dt = float(stats.sample_dt(curr, lo=min_dt, hi=max_dt)[0])

                # sample victims (truncated normal, symmetric about mean)
                mu_dv = np.clip(float(g['dv']), physical_min_vict, physical_max_vict)
                del_dv = min(mu_dv - physical_min_vict, physical_max_vict - mu_dv)
                min_dv, max_dv = mu_dv - del_dv, mu_dv + del_dv
                dv = float(stats.sample_dv(curr, lo=min_dv, hi=max_dv)[0])

                # sample shots (truncated normal, symmetric about mean)
                mu_ds = np.clip(float(g['ds']), physical_min_shot, physical_max_shot)
                del_ds = min(mu_ds - physical_min_shot, physical_max_shot - mu_ds)
                min_ds, max_ds = mu_ds - del_ds, mu_ds + del_ds
                ds = float(stats.sample_ds(curr, lo=min_ds, hi=max_ds)[0])

            elif strategy == "X_sampling" and use_coupling:

                # sample time (truncated normal, symmetric about mean)
                mu_dt = np.clip(stats.mean_time(curr), physical_min_time, physical_max_time)
                del_dt = min(mu_dt - physical_min_time, physical_max_time - mu_dt)
                min_dt, max_dt = mu_dt - del_dt, mu_dt + del_dt
                dt = float(stats.sample_dt(curr, lo=min_dt, hi=max_dt)[0])

                # dependent shots and victims
                s, v, t = g['ds'], g['dv'], g['dt']
                mu_vr = v/t if t> 0 else 0.0
                mu_sr = s/t if t> 0 else 0.0
                dt = float(stats.sample_dt(curr, lo=min_dt, hi=max_dt)[0])
                dv = float(np.clip(mu_vr*dt, physical_min_vict, physical_max_vict))
                ds = float(np.clip(mu_sr*dt, physical_min_shot, physical_max_shot))

            # apply robot effects
            if i < len(self.r1_nodes):
                r1_label, r2_label = self.r1_nodes[i], self.r2_nodes[i]
                kt, ks, kv = (stats.k_mod.get(curr, {}).get(d, 0.0) for d in ['dt', 'ds', 'dv'])
                R = self.reff[i]
                dt = dt + kt * R
                ds = ds + ks * R
                dv = dv + kv * R
            else:
                r1_label, r2_label, R = None, None, None

            # optionally round to int
            if force_int:
                if strategy == "node_sample":
                    # stochastic rounding for sampling strategies
                    dv = int(np.floor(np.clip(dv + np.random.rand(), 0, physical_max_vict)))
                    ds = int(np.floor(np.clip(ds + np.random.rand(), 0, np.inf)))
                else:
                    # deterministic rounding for mean-based strategies
                    dv = int(round(np.clip(dv, 0, physical_max_vict)))
                    ds = int(round(np.clip(ds, 0, np.inf)))

            # apply constraints and bookkeeping
            dv = min(dv, physical_max_vict)
            self.global_alive = max(0, self.global_alive - dv)
            # self.nodal_alive[curr] = max(0, self.nodal_alive[curr] - dv)

            if self.use_gnn and total_time + dt > final_time:
                remaining = final_time - total_time
                if remaining <= 0:
                    break

                # scale last-node dwell and events proportionally
                scale = remaining / dt
                dt = remaining
                ds = ds * scale
                dv = dv * scale

                total_time += dt
                total_shot += ds
                total_vict += dv
                history.append((curr, dt, ds, dv, self.global_alive, r1_label, r2_label, R))
                break
            else:
                # normal accumulation
                total_time += dt
                total_vict += dv
                total_shot += ds
                history.append((curr, dt, ds, dv, self.global_alive, r1_label, r2_label, R))

        return {
            "nodes": self.nodes[:len(history)],
            "time": total_time,
            "victims": total_vict,
            "shots": total_shot,
            "full_history": history,
        }
