import os, pickle
import tensorflow as tf
import numpy as np
from gymnasium import spaces
from pathlib import Path
from .network_mlp import MLPQ
from .prioritized_replay import PrioritizedReplay
from src.utils.paths import ensure_dir


Q_TIE_EPS = 0.05


class DQNAgent:
    def __init__(self, env, ctx, hidden_dim = 64, lr=1e-4, gamma=0.995, tau=0.0015,
                 epsilon_start=1.0, beta_start=0.4, batch_size=64):

        self.env    = env
        self.ctx    = ctx
        self.gamma  = gamma
        self.tau    = tau
        self.beta   = beta_start
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.tie_break_count = 0
        self.hidden_dim = hidden_dim


        # number of joint actions (NOT number of nodes)
        if isinstance(env.action_space, spaces.MultiDiscrete):
            self.n_actions_r1 = env.action_space.nvec[0]
            self.n_actions_r2 = env.action_space.nvec[1]
            self.n_actions = int(self.n_actions_r1 * self.n_actions_r2)
        else:
            raise ValueError("This agent requires MultiDiscrete action space.")

        # ============================================================
        # Networks
        # ============================================================
        self.q_net      = MLPQ(n_actions=self.n_actions, n_hidden = self.hidden_dim)
        self.target_net = MLPQ(n_actions=self.n_actions, n_hidden = self.hidden_dim)

        # Build networks by calling once
        obs_dim = env.observation_space.shape[0]
        dummy = tf.zeros((1, obs_dim))
        self.q_net(dummy)
        self.target_net(dummy)
        self.target_net.set_weights(self.q_net.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.replay    = PrioritizedReplay(alpha=0.5, capacity=50_000)

    # ============================================================
    # Mask invalid JOINT actions
    # ============================================================
    def _masked_joint_q(self, q_vec, env):
        """
        q_vec: shape (n_actions,)
        Returns masked Q with invalid actions = -inf
        """

        valid = env._get_valid_actions()  # list of joint flat action ids
        mask = tf.fill([self.n_actions], -np.inf)
        valid_tf = tf.convert_to_tensor(valid, dtype=tf.int32)

        # scatter valid Q-values in place
        masked = tf.tensor_scatter_nd_update(
            mask,
            indices=tf.expand_dims(valid_tf, axis=1),
            updates=tf.gather(q_vec, valid_tf)
        )

        return masked

    # ============================================================
    #  Action Selection
    # ============================================================
    def select_action(self, X, env):
        valid = env._get_valid_actions()

        # ε-greedy exploration
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid))

        # Forward pass
        X_tf = tf.convert_to_tensor(X[None], dtype=tf.float32)
        q_vec = self.q_net(X_tf, training=False)[0]

        # Mask invalid joint actions
        masked = self._masked_joint_q(q_vec, env)

        # Select best valid action
        valid_tf = tf.convert_to_tensor(valid, dtype=tf.int32)
        q_valid = tf.gather(masked, valid_tf)

        best_idx = tf.argmax(q_valid).numpy()
        return int(valid[best_idx])


    # ============================================================
    # DDQN Target Calculation
    # ============================================================
    def _ddqn_best_next(self, next_q_online, next_q_target, r1_idx, r2_idx,
                        r1_nbrs, r2_nbrs):

        B = len(r1_idx)
        out = np.zeros(B, dtype=np.float32)

        for b in range(B):
            legal1 = [r1_idx[b]] + list(r1_nbrs[b])
            legal2 = [r2_idx[b]] + list(r2_nbrs[b])

            joint_valid = []
            for i_rel, _ in enumerate(legal1):
                for j_rel, _ in enumerate(legal2):
                    flat = i_rel * self.n_actions_r2 + j_rel
                    joint_valid.append(flat)

            joint_valid = np.array(joint_valid, np.int32)

            # ONLINE selects best
            q_online = next_q_online[b].numpy()
            q_online_valid = q_online[joint_valid]
            idx = np.argmax(q_online_valid)
            best_joint = joint_valid[idx]

            # TARGET evaluates
            q_target = next_q_target[b].numpy()
            out[b] = q_target[best_joint]

        return tf.convert_to_tensor(out, dtype=tf.float32)

    # ============================================================
    #  Training Step
    # ============================================================
    def train_step(self):
        if len(self.replay) < self.batch_size:
            return

        (Xs, actions, rewards, next_Xs, dones,
         idxs, weights,
         r1_idx, r2_idx, r1_nbrs, r2_nbrs) = self.replay.sample(
            self.batch_size, beta=self.beta
        )

        # Convert to tensors
        Xs_tf      = tf.convert_to_tensor(Xs, tf.float32)
        next_Xs_tf = tf.convert_to_tensor(next_Xs, tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards, tf.float32)
        dones_tf   = tf.convert_to_tensor(dones, tf.float32)
        weights_tf = tf.convert_to_tensor(weights, tf.float32)
        actions_tf = tf.convert_to_tensor(actions, tf.int32)

        # Flatten observations
        Xs_flat      = tf.reshape(Xs_tf, (self.batch_size, -1))
        next_Xs_flat = tf.reshape(next_Xs_tf, (self.batch_size, -1))

        # Q(s')
        # next_q_online = tf.clip_by_value(self.q_net(next_Xs_flat), -100, 100)
        # next_q_target = tf.clip_by_value(self.target_net(next_Xs_flat), -100, 100)
        next_q_online = self.q_net(next_Xs_flat)
        next_q_target = self.target_net(next_Xs_flat)

        # masked DDQN target values
        next_best = self._ddqn_best_next(
            next_q_online, next_q_target,
            r1_idx, r2_idx, r1_nbrs, r2_nbrs
        )

        # TD target
        target_raw = rewards_tf + (1.0 - dones_tf) * self.gamma * next_best
        # target = tf.clip_by_value(target_raw, -10.0, 10.0)
        target = target_raw


        # ============================================================
        # Loss + Backprop
        # ============================================================
        with tf.GradientTape() as tape:
            # q_all = tf.clip_by_value(self.q_net(Xs_flat), -100, 100)
            q_all = self.q_net(Xs_flat)

            self.last_mean_q = float(tf.reduce_mean(q_all).numpy())

            chosen_q = tf.gather(q_all, actions_tf, axis=1, batch_dims=1)

            td_raw = target_raw - chosen_q
            td     = target - chosen_q

            loss = tf.reduce_mean(weights_tf * (td ** 2))

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        # PER priority update
        self.replay.update_priorities(idxs, np.minimum(np.abs(td_raw.numpy()), 50))

        # Soft update target network
        for t, s in zip(self.target_net.variables, self.q_net.variables):
            t.assign((1 - self.tau) * t + self.tau * s)

        self.last_loss = float(loss.numpy())

    # ============================================================
    # Save / Load
    # ============================================================
    def save(self, path, episode=0, epoch=0):
        path = ensure_dir(Path(path))
        self.q_net.save_weights(str(path / "q_net.h5"))
        self.target_net.save_weights(str(path / "target_net.h5"))
        meta = dict(epsilon=self.epsilon, episode=episode, epoch=epoch)
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def load(self, path):
        path = Path(path)
        obs_dim = self.env.observation_space.shape[0]
        dummy = tf.zeros((1, obs_dim))
        self.q_net(dummy)
        self.target_net(dummy)
        self.q_net.load_weights(str(path / "q_net.h5"))
        self.target_net.load_weights(str(path / "target_net.h5"))
        meta = {"epsilon": 1.0}
        mp = path / "meta.pkl"
        if mp.exists():
            with open(mp, "rb") as f:
                meta = pickle.load(f)
        self.epsilon = meta.get("epsilon", 1.0)
        return meta

