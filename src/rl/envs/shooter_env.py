import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from src.utils.robot import get_robot_snapshot

MIN_DT = 0.5
RO1_START = 1
RO2_START = 281
BETA = 0.01046

class Shooter:
    def __init__(self, ctx, path, env):
        self.ctx = ctx
        self.env = env
        self.stats = ctx.stats
        self.node_to_idx = ctx.node_idx_map
        self.idx_to_node = ctx.idx_node_map
        self.trans_dict = ctx.min_pt_shooter or {}

        self.path = [self.node_to_idx[ix] for ix in path]
        self.prev_idx = None
        self.curr_idx = self.path[0]
        self.goal_idx = self.path[1]
        self.progress = 0.0
        self.trans_time = MIN_DT
        self.num_event = 0
        self.max_time = ctx.final_time

        self.last120 = deque(maxlen=120)
        self.last120.extend([None]*120)

    def get_event_stats(self, episode):

        # get physical constraints
        stats = self.stats
        prev = self.idx_to_node.get(self.prev_idx)
        curr = self.idx_to_node.get(self.curr_idx)
        next_ = self.idx_to_node.get(self.goal_idx)
        if any([prev is None, next_ is None, prev == next_]):
            physical_min_time = self.stats.min_time_inout(curr)
        else:
            physical_min_time = self.stats.min_time_between(prev, curr, next_)
        physical_max_time = self.max_time
        physical_min_vict = 0.0
        physical_max_vict = episode.nodal_alive_init.get(curr, 0.0)
        physical_min_shot = 0.0
        physical_max_shot = np.inf

        # get robot effects
        self.event_kt = self.stats.k_mod.get(curr, {}).get("dt", 0.0)
        self.event_kv = self.stats.k_mod.get(curr, {}).get("dv", 0.0)
        self.event_ks = self.stats.k_mod.get(curr, {}).get("ds", 0.0)

        if episode.strategy == "nodal_sampling":
            # sample time (truncated normal, symmetric about mean)
            mu_dt = np.clip(self.stats.mean_time(curr), physical_min_time, physical_max_time)
            del_dt = min(mu_dt - physical_min_time, physical_max_time - mu_dt)
            min_dt, max_dt = mu_dt - del_dt, mu_dt + del_dt
            self.event_dt = float(self.stats.sample_dt(curr, lo=min_dt, hi=max_dt)[0])

            # sample victims (truncated normal, symmetric about mean)
            mu_dv = np.clip(stats.mean_victims(curr), physical_min_vict, physical_max_vict)
            del_dv = min(mu_dv - physical_min_vict, physical_max_vict - mu_dv)
            min_dv, max_dv = mu_dv - del_dv, mu_dv + del_dv
            self.event_dv = float(stats.sample_dv(curr, lo=min_dv, hi=max_dv)[0])

            # sample shots (truncated normal, symmetric about mean)
            mu_ds = np.clip(stats.mean_shots(curr), physical_min_shot, physical_max_shot)
            del_ds = min(mu_ds - physical_min_shot, physical_max_shot - mu_ds)
            min_ds, max_ds = mu_ds - del_ds, mu_ds + del_ds
            self.event_ds = float(stats.sample_ds(curr, lo=min_ds, hi=max_ds)[0])

        elif episode.strategy == "nodal_means":
            self.event_dt = np.clip(float(stats.mean_time(curr)), physical_min_time, physical_max_time)
            self.event_dv = np.clip(float(stats.mean_victims(curr)), physical_min_vict, physical_max_vict)
            self.event_ds = np.clip(float(stats.mean_shots(curr)), physical_min_shot, physical_max_shot)

        elif episode.strategy == "nodal_coupling":
            g = dict()
            g['dt'] = float(stats.mean_time(curr))
            g['ds'] = float(stats.mean_shots(curr))
            g['dv'] = float(stats.mean_victims(curr))
            s, v, t = g['ds'], g['dv'], g['dt']
            mu_vr = v/t if t> 0 else 0.0
            mu_sr = s/t if t> 0 else 0.0
            dt_now = np.clip(float(g['dt']), physical_min_time, physical_max_time)
            self.event_dt = dt_now
            self.event_dv = float(np.clip(mu_vr*dt_now, physical_min_vict, physical_max_vict))
            self.event_ds = float(np.clip(mu_sr*dt_now, physical_min_shot, physical_max_shot))
        else:
            raise ValueError(f"Episode temporal strategy '{episode.strategy}' not recognized!")

    def move_forward(self):
        self.num_event += 1
        if self.num_event + 1 < len(self.path):
            self.prev_idx = self.path[self.num_event - 1]
            self.curr_idx = self.path[self.num_event]
            self.goal_idx = self.path[self.num_event + 1]
            key = tuple(self.idx_to_node[x] for x in [self.prev_idx, self.curr_idx, self.goal_idx])
            self.trans_time = self.trans_dict.get(key, MIN_DT)
            self.progress = 0.0

    def advance(self, Rt, episode):

        # current smoke
        curr = self.idx_to_node[self.curr_idx]
        R = Rt[self.curr_idx]

        # time logic
        dt_inc = episode.dt                                 # constant timestep increment
        dt_unmod = self.event_dt                            # fixed per event
        dt_mod = max(dt_unmod + self.event_kt * R, dt_inc)  # changes per timestep

        # des progression
        progress_rate = dt_inc / dt_mod
        self.progress += progress_rate

        # accumulate values
        physical_max_vict = episode.nodal_alive_init.get(curr, 0.0)
        ds_inc = progress_rate * np.clip(self.event_ds + self.event_ks * R, 0, np.inf)
        dv_inc = progress_rate * np.clip(self.event_dv + self.event_kv * R, 0, physical_max_vict)
        self.env.tpn[curr] += dt_inc
        self.env.vpn[curr] += dv_inc
        self.env.spn[curr] += ds_inc

        # event finished
        if self.progress >= 1.0:
            self.progress = 0.0
            self.move_forward()
            if not episode.done:
                self.get_event_stats(episode)

class Robot:
    def __init__(self, ctx, start_idx, valid_idx):
        self.trans_dict = ctx.min_pt_robot or {}
        self.prev_idx = None
        self.curr_idx = start_idx
        self.goal_idx = start_idx
        self.valid_idx = valid_idx
        self.n_nodes = len(valid_idx)
        self.progress = 0.0
        self.done = True
        self.trans_time = MIN_DT
        self.last120 = deque(maxlen=120)
        self.last120.extend([None] * 120)
        self.nbrs_idx = {
            i: sorted([nbr for nbr in ctx.neighbors_idx[i] if nbr in self.valid_idx])
            for i in self.valid_idx
        }

    def start_move(self, a_rel, idx_to_node):
        """Start a new move if ready."""
        if not self.done:
            return
        self.done = False
        nbr_idx = self.nbrs_idx[self.curr_idx]
        a_idx = self.curr_idx if len(nbr_idx) == 0 or a_rel == 0 else nbr_idx[(a_rel - 1) % len(nbr_idx)]
        self.goal_idx = a_idx
        prev = self.prev_idx if self.prev_idx is not None else self.curr_idx
        key = tuple(idx_to_node[x] for x in [prev, self.curr_idx, self.goal_idx])
        self.trans_time = self.trans_dict.get(key, MIN_DT)

    def advance(self, dt, idx_to_node):
        """Progress robot along its transition."""
        if self.done or self.goal_idx is None:
            return
        self.progress += dt / self.trans_time
        if self.progress >= 1.0:
            self.done = True
            self.prev_idx, self.curr_idx = self.curr_idx, self.goal_idx
            self.goal_idx = None
            self.progress = 0.0

class Environment:
    def __init__(self, ctx, robot_type):
        self.ctx = ctx
        self.robot_type = robot_type
        self.Aij, self.Dij = ctx.A_np, ctx.Dij
        self.node_order = ctx.node_order
        self.n_nodes = len(self.node_order)
        self.Rt = np.zeros(self.n_nodes, dtype=np.float32)
        self.tpn = {k: 0 for k in self.node_order}
        self.vpn = {k: 0 for k in self.node_order}
        self.spn = {k: 0 for k in self.node_order}


    def advance(self, sho, ro1, ro2):
        """Advance the environment one step:
        - Updates each agent's 120-step position history
        - Recomputes robot influence field Rt
        """
        # append current positions to histories
        ro1.last120.append(ro1.curr_idx)
        ro2.last120.append(ro2.curr_idx)
        sho.last120.append(sho.curr_idx)

        if isinstance(self.robot_type, (int, float)):
            self.Rt = np.zeros(self.n_nodes, dtype=np.float32)
            self.Rt[self.robot_type] = 0.5

        elif self.robot_type == "ignore":
            self.Rt = np.zeros(self.n_nodes, dtype=np.float32)

        elif self.robot_type == "max":
            self.Rt = np.ones(self.n_nodes, dtype=np.float32)

        else:
            # compute instantaneous robot influence field
            nonzero_r1 = [i for i in ro1.last120 if i is not None]
            nonzero_r2 = [i for i in ro2.last120 if i is not None]

            # backfill if histories empty
            if not nonzero_r1:
                nonzero_r1 = [ro1.curr_idx]
            if not nonzero_r2:
                nonzero_r2 = [ro2.curr_idx]

            self.Rt = get_robot_snapshot(nonzero_r1, nonzero_r2, self.Dij)
            self.Rt = np.clip(self.Rt, 0.0, 1.0)

    def enforce_positive(self):
        """Ensures values are non-negative."""
        for d in (self.vpn, self.spn):
            for k, v in d.items():
                d[k] = max(0.0, v)

class Episode:
    def __init__(self, ctx, env, ro1, ro2, strategy):
        self.dt = MIN_DT
        self.env = env
        self.final_time = ctx.final_time
        self.truncated = False
        self.terminated = False
        self.total_time = 0.0
        self.num_step = 0
        self.total_alive_init = ctx.stats.global_alive_init
        self.nodal_alive_init = ctx.stats.nodal_alive_init.copy()
        self.strategy = strategy
        self._get_num_actions(ro1, ro2)
        obs_dim = self.n_actions_r1 + self.n_actions_r2
        self.O_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.step_reward = 0.0
        self.step_term1 = 0.0
        self.step_term2 = 0.0
        self.beta = BETA

    def _get_num_actions(self, ro1, ro2):
        assert all(len(v) > 0 for v in ro1.nbrs_idx.values()), "Robot1 has isolated nodes!"
        assert all(len(v) > 0 for v in ro2.nbrs_idx.values()), "Robot2 has isolated nodes!"
        max_nbrs_r1 = max(len(v) for v in ro1.nbrs_idx.values())
        max_nbrs_r2 = max(len(v) for v in ro2.nbrs_idx.values())
        self.n_actions_r1 = max_nbrs_r1 + 1
        self.n_actions_r2 = max_nbrs_r2 + 1

    def _vr(self, node_name):
        stats = self.env.ctx.stats
        v = stats.mean_victims(node_name)
        t = stats.mean_time(node_name)
        return (v / t) if t > 0 else 0.0

    def update_strategy(self, strategy):
        self.strategy = strategy

    @property
    def total_victims(self):
        """Return total victims across all nodes."""
        return sum(self.env.vpn.values())

    @property
    def total_shots(self):
        """Return total shots across all nodes."""
        return sum(self.env.spn.values())

    @property
    def total_alive(self):
        """Return total alive npcs across all nodes."""
        return max(0, self.total_alive_init - sum(self.env.vpn.values()))

    @property
    def done(self):
        return (self.truncated or self.terminated)

class ShooterEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, ctx, shooter_path, strategy="nodal_sampling", robot_type = "normal"):
        super().__init__()
        self.ctx = ctx
        self.node_to_idx = ctx.node_idx_map
        self.idx_to_node = ctx.idx_node_map
        self.ro1_valid = [self.node_to_idx[n] for n, ok in ctx.robot1_nodes.items() if ok == 1]
        self.ro2_valid = [self.node_to_idx[n] for n, ok in ctx.robot2_nodes.items() if ok == 1]
        self.ro1_start = self.node_to_idx.get(RO1_START)
        self.ro2_start = self.node_to_idx.get(RO2_START)
        self.shooter_path = shooter_path
        self.robot_type = robot_type
        self.strategy = strategy
        self._reset_vars()

    def _reset_vars(self):
        self.env = Environment(self.ctx, self.robot_type)
        self.ro1 = Robot(self.ctx, self.ro1_start, self.ro1_valid)
        self.ro2 = Robot(self.ctx, self.ro2_start, self.ro2_valid)
        self.sho = Shooter(self.ctx, self.shooter_path, self.env)
        self.eps = Episode(self.ctx, self.env, self.ro1, self.ro2, self.strategy)
        self.sho.get_event_stats(self.eps)
        self.action_space = spaces.MultiDiscrete([
            self.eps.n_actions_r1,
            self.eps.n_actions_r2
        ])
        self.observation_space = self.eps.O_space
        return self._get_obs(), {}

    def set_strategy(self, strategy):
        """Change curriculum strategy mid-training."""
        self.strategy = strategy
        self.eps.strategy = strategy

    def _last_unique(self, seq, n=3):
        """
        Return the last n unique non-None elements from a deque or list.
        If fewer than n valid entries exist, pad with the most recent valid one.
        """
        seen = set()
        uniq = []

        # Iterate backward, skipping None placeholders
        for x in reversed(seq):
            if x is None:
                continue
            if x not in seen:
                seen.add(x)
                uniq.append(x)
            if len(uniq) == n:
                break

        # If we ran out of unique nodes, pad with last valid
        if len(uniq) == 0:
            # still empty — no valid data at all
            return [0] * n  # or some sentinel index
        while len(uniq) < n:
            uniq.append(uniq[-1])

        return list(reversed(uniq))  # return oldest → newest

    def _next_dist_after_action(self, robot, a_rel):
        """
        Returns normalized distance to shooter after taking relative action a_rel.
        """
        paths = self.ctx.shortest_paths
        Dmax = float(self.ctx.graph_diameter)

        curr = robot.curr_idx
        nbrs = robot.nbrs_idx[curr]

        # stay
        if a_rel == 0 or len(nbrs) == 0:
            next_idx = curr
        else:
            # a_rel ∈ [1, len(nbrs)]
            next_idx = nbrs[(a_rel - 1) % len(nbrs)]

        d = len(paths[next_idx][self.sho.curr_idx]) - 1
        d = max(d, 0)

        return d / Dmax


    def _get_obs(self):
        A1 = self.eps.n_actions_r1
        A2 = self.eps.n_actions_r2
        act_ro1 = np.ones(A1, dtype=np.float32)
        act_ro2 = np.ones(A2, dtype=np.float32)

        for a in range(A1):
            act_ro1[a] = self._next_dist_after_action(self.ro1, a)

        for a in range(A2):
            act_ro2[a] = self._next_dist_after_action(self.ro2, a)

        obs = np.concatenate([act_ro1,act_ro2])
        return obs



    def _flatten_action(self, action):
        """Convert a single integer action into (a1_rel, a2_rel)."""
        n2 = self.eps.n_actions_r2
        a1_rel, a2_rel = divmod(int(action), n2)
        return a1_rel, a2_rel

    def _unflatten_action(self, a1_rel, a2_rel):
        """Convert (a1, a2) pair back into flattened integer index."""
        n2 = self.eps.n_actions_r2
        return a1_rel * n2 + a2_rel

    def _get_valid_actions(self):
            """
            Return all valid flattened joint actions for the current state.
            Each robot may:
                - stay in place (relative index = 0)
                - move to one of its valid neighbors
            """
            valid = []

            # Robot 1 legal relative actions
            nbr1 = self.ro1.nbrs_idx[self.ro1.curr_idx]
            legal1 = [0] + list(range(1, len(nbr1) + 1))

            # Robot 2 legal relative actions
            nbr2 = self.ro2.nbrs_idx[self.ro2.curr_idx]
            legal2 = [0] + list(range(1, len(nbr2) + 1))

            # Cross-product → valid joint actions
            for a1_rel in legal1:
                for a2_rel in legal2:
                    flat = self._unflatten_action(a1_rel, a2_rel)
                    valid.append(flat)

            return valid

    def _advance_episode(self):
        self.eps.total_time += self.eps.dt
        self.eps.num_step += 1
        if self.eps.total_time >= self.eps.final_time:
            self.eps.truncated = True

    def _get_reward(self):
        paths = self.ctx.shortest_paths

        # shooter and robot *indices*
        idx_sho = self.sho.curr_idx
        idx_r1  = self.ro1.curr_idx
        idx_r2  = self.ro2.curr_idx

        # hop distances (length - 1)
        d1 = len(paths[idx_r1][idx_sho]) - 1
        d2 = len(paths[idx_r2][idx_sho]) - 1
        d1 = max(d1, 0)
        d2 = max(d2, 0)

        # graph diameter
        Dmax = float(self.ctx.graph_diameter)

        # cost in [0, 1]
        dist_cost = ((d1 + d2) / (2.0 * Dmax))
        dist_cost = float(np.clip(dist_cost, 0.0, 1.0))
        return d1, d2, -dist_cost

    def step(self, action):

        # interpret action
        a1_rel, a2_rel = self._flatten_action(action)
        nbr1_idx = self.ro1.nbrs_idx[self.ro1.curr_idx]
        nbr2_idx = self.ro2.nbrs_idx[self.ro2.curr_idx]

        # initialize counts
        step_reward = 0.0
        step_d1 = 0.0
        step_d2 = 0.0

        # take action
        self.ro1.start_move(a1_rel, self.idx_to_node)
        self.ro2.start_move(a2_rel, self.idx_to_node)
        while not (self.eps.done or (self.ro1.done and self.ro2.done)):
            self.ro1.advance(self.eps.dt, self.idx_to_node)
            self.ro2.advance(self.eps.dt, self.idx_to_node)
            self.env.advance(self.sho, self.ro1, self.ro2)
            self.sho.advance(self.env.Rt, self.eps)
            d1, d2, r = self._get_reward()
            step_d1 += d1 * self.eps.dt
            step_d2 += d2 * self.eps.dt
            step_reward += r * self.eps.dt
            self._advance_episode()


        if self.eps.done:
            self.env.enforce_positive()

        self.eps.step_term1 = step_d1
        self.eps.step_term2 = step_d2
        self.eps.step_reward = step_reward

        obs = self._get_obs()
        reward = self.eps.step_reward
        terminated = self.eps.terminated
        truncated = self.eps.truncated
        info = {
            "time": self.eps.total_time,
            "nodes": self.sho.num_event + 1,
            "shots": self.eps.total_shots,
            "victims": self.eps.total_victims,
            "term1": self.eps.step_term1,
            "term2": self.eps.step_term2,
            "nbrs_r1": nbr1_idx,
            "nbrs_r2": nbr2_idx,
            "idx_sho": self.sho.curr_idx,
            "idx_ro1": self.ro1.curr_idx,
            "idx_ro2": self.ro2.curr_idx,
            "curr_Rt": self.env.Rt

        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._reset_vars()

    def render(self, mode="human"):
        sho_node = self.idx_to_node[self.sho.curr_idx]
        ro1_node = self.idx_to_node[self.ro1.curr_idx]
        ro2_node = self.idx_to_node[self.ro2.curr_idx]
        print(f"Step {self.eps.num_step:03d} | Shooter={sho_node} | R1={ro1_node} | R2={ro2_node}")


