# =====================================================================
# Warm-start MLP-Q network using heuristic: "move toward shooter"
# =====================================================================

import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm
from src.rl.envs import ShooterEnv

# --------------------------------------------------------------
# Compute heuristic Q-values for the NEW MLP AGENT
# --------------------------------------------------------------
def compute_heuristic_Q(env, ctx, agent):
    """
    Produce joint-action Q-targets for the MLP.
    Q[a] = -(d1_next + d2_next)/2

    where d1_next, d2_next are hop distances AFTER taking action a.
    """

    paths = ctx.shortest_paths
    idx_sho = env.sho.curr_idx

    nA1 = agent.n_actions_r1
    nA2 = agent.n_actions_r2
    n_actions = agent.n_actions

    Q = np.full(n_actions, -1e6, dtype=np.float32)  # invalid = large negative

    # Current indices
    cur_r1 = env.ro1.curr_idx
    cur_r2 = env.ro2.curr_idx

    nbr1 = env.ro1.nbrs_idx[cur_r1]
    nbr2 = env.ro2.nbrs_idx[cur_r2]

    # Valid relative actions
    legal1 = [0] + list(range(1, len(nbr1) + 1))
    legal2 = [0] + list(range(1, len(nbr2) + 1))

    Dmax = float(ctx.graph_diameter)

    for a1_rel in legal1:
        for a2_rel in legal2:

            flat = a1_rel * nA2 + a2_rel

            # Determine next positions
            next_r1 = cur_r1 if a1_rel == 0 else nbr1[a1_rel - 1]
            next_r2 = cur_r2 if a2_rel == 0 else nbr2[a2_rel - 1]

            # Hop distances (path length − 1)
            d1 = len(paths[next_r1][idx_sho]) - 1
            d2 = len(paths[next_r2][idx_sho]) - 1
            d1 = max(d1, 0)
            d2 = max(d2, 0)

            Q[flat] = -( (d1 + d2) / (2 * Dmax) )

    return Q


# --------------------------------------------------------------
# Warm-start loop
# --------------------------------------------------------------
def warm_start(agent, ctx, shooter_paths, n_iters=2000, strategy="nodal_means"):
    """
    Supervised warm-start for the NEW MLP-Q network.
    """

    print(f"[Warm-start] Pretraining MLP-Q for {n_iters} steps...")

    losses = []
    prev_eps = agent.epsilon
    agent.epsilon = 0.0   # deterministic

    for _ in tqdm(range(n_iters), desc="Warm-start"):

        # pick random shooter path
        path = random.choice(shooter_paths)
        env = ShooterEnv(ctx, path, strategy=strategy)
        obs, _ = env.reset()

        obs = obs.astype(np.float32)  # shape (5,)
        obs_tf = tf.convert_to_tensor(obs[None], tf.float32)

        # Compute action-value targets under heuristic
        Q_target = compute_heuristic_Q(env, ctx, agent)
        Q_target_tf = tf.convert_to_tensor(Q_target[None], tf.float32)

        # Supervised gradient step
        with tf.GradientTape() as tape:
            Q_pred = agent.q_net(obs_tf, training=True)
            loss = tf.reduce_mean((Q_pred - Q_target_tf)**2)

        grads = tape.gradient(loss, agent.q_net.trainable_variables)
        agent.optimizer.apply_gradients(zip(grads, agent.q_net.trainable_variables))

        losses.append(float(loss))

    agent.epsilon = prev_eps
    print(f"[Warm-start] Done. Final loss(last50) = {np.mean(losses[-50:]):.4f}")

    return losses
