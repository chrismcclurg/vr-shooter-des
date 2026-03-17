import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from src.des import SimContext
from src.rl.envs import ShooterEnv
from src.rl.dqn import DQNAgent, warm_start
from src.utils.rl import load_shooter_paths
from src.utils.paths import RL_MODEL_DIR, ensure_dir

def _run_episode(agent, ctx, path, strategy, train_online=True, step_update_freq=10):
    env = ShooterEnv(ctx, path, strategy=strategy)
    env.epsilon = agent.epsilon
    obs, _ = env.reset()
    obs = obs.astype(np.float32)

    done = False
    curr_reward = 0.0
    step_count = 0
    curr_term1, curr_term2, curr_r_dist, curr_r_rate = 0, 0, 0, 0

    while not done:
        action = agent.select_action(obs, env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = next_obs.astype(np.float32)
        done = terminated or truncated

        # Store transition
        agent.replay.push(
            X=obs,
            action=action,
            reward=reward,
            next_X=next_obs,
            done=float(done),
            r1_idx=info["idx_ro1"],
            r2_idx=info["idx_ro2"],
            r1_nbrs=info["nbrs_r1"],
            r2_nbrs=info["nbrs_r2"]
        )

        # Online learning
        if train_online and step_count % step_update_freq == 0:
            agent.train_step()

        obs = next_obs
        step_count += 1
        curr_reward += reward
        curr_term1 += info['term1']
        curr_term2 += info['term2']
        curr_r_dist += info['r_dist']
        curr_r_rate += info['r_rate']

    episode_time = info.get("time", ctx.final_time)
    episode_vict = info.get("victims", 0)
    episode_node = info.get("nodes", 1)
    episode_shot = info.get("shots", 0)
    episode_d1 = curr_term1 / episode_time
    episode_d2 = curr_term2 / episode_time

    episode_reward = curr_reward
    episode_r_dist = curr_r_dist / episode_time
    episode_r_rate = curr_r_rate / episode_time

    stats = {'r_tot': episode_reward,
             'r_dist': episode_r_dist,
             'r_rate': episode_r_rate,
             'd1': episode_d1,
             'd2': episode_d2,
             'v': episode_vict,
             'n': episode_node,
             's': episode_shot}

    return stats

def _offline_retrain(agent, n_updates=500):
    print(f"   [Offline retrain] Performing {n_updates} replay updates...")
    for _ in range(n_updates):
        agent.train_step()

def _log_metrics(writer, ep_idx, stats, agent):

    with writer.as_default():
        tf.summary.scalar("episode/r_tot", stats['r_tot'], step=ep_idx)
        tf.summary.scalar("episode/r_dist", stats['r_dist'], step=ep_idx)
        tf.summary.scalar("episode/r_rate", stats['r_rate'], step=ep_idx)
        tf.summary.scalar("episode/num_vict", stats['v'], step=ep_idx)
        tf.summary.scalar("episode/num_node", stats['n'], step=ep_idx)
        tf.summary.scalar("episode/num_shot", stats['s'], step=ep_idx)
        tf.summary.scalar("episode/d1", stats['d1'], step=ep_idx)
        tf.summary.scalar("episode/d2", stats['d2'], step=ep_idx)
        tf.summary.scalar("episode/epsilon", agent.epsilon, step=ep_idx)
        tf.summary.scalar("episode/beta", agent.beta, step=ep_idx)

        if getattr(agent, "last_loss", None) is not None:
            tf.summary.scalar("train/loss", agent.last_loss, step=ep_idx)

        if getattr(agent, "last_mean_q", None) is not None:
            tf.summary.scalar("train/mean_q", agent.last_mean_q, step=ep_idx)

    writer.flush()

def update_epsilon(agent, ep_idx, n_epochs, episodes_per_epoch):

    eps_start = 1.0
    eps_mid   = 0.20
    eps_end   = 0.05

    tot_eps = n_epochs * episodes_per_epoch

    # phase boundaries
    phase1_end = int(0.5 * tot_eps)   # decay 1.00 → 0.20
    phase2_end = int(0.8 * tot_eps)   # decay 0.20 → 0.05

    # ----- Phase 1 -----
    if ep_idx < phase1_end:
        frac = ep_idx / phase1_end
        agent.epsilon = eps_start - (eps_start - eps_mid) * frac
        return

    # ----- Phase 2 -----
    if ep_idx < phase2_end:
        frac = (ep_idx - phase1_end) / (phase2_end - phase1_end)
        agent.epsilon = eps_mid - (eps_mid - eps_end) * frac
        return

    # ----- Phase 3 -----
    agent.epsilon = eps_end

def update_beta(agent, ep_idx, n_epochs, episodes_per_epoch):

    beta_start = 0.40
    beta_mid1  = 0.60
    beta_mid2  = 0.90
    beta_end   = 1.00

    tot_eps = n_epochs * episodes_per_epoch
    phase1_end = int(0.5 * tot_eps)   # first 50%
    phase2_end = int(0.8 * tot_eps)   # 50%->80%

    # ----- Phase 1: 0.40 -> 0.60 -----
    if ep_idx < phase1_end:
        frac = ep_idx / phase1_end
        agent.beta = beta_start + (beta_mid1 - beta_start) * frac
        return

    # ----- Phase 2: 0.60 -> 0.90 -----
    if ep_idx < phase2_end:
        frac = (ep_idx - phase1_end) / (phase2_end - phase1_end)
        agent.beta = beta_mid1 + (beta_mid2 - beta_mid1) * frac
        return

    # ----- Phase 3: 0.90 -> 1.00 -----
    frac = (ep_idx - phase2_end) / (tot_eps - phase2_end)
    agent.beta = beta_mid2 + (beta_end - beta_mid2) * frac

def summarize_epoch(x):
    x = np.asarray(x, dtype=np.float32)
    mean = float(np.mean(x))
    std  = float(np.std(x, ddof=1))
    ci95 = 1.96 * std / np.sqrt(len(x))
    return mean, std, ci95

def train(
    n_epochs,
    offline_freq=100,
    eval_freq=10,
    offline_updates=500,
    step_update_freq=10,
    resume=True,
    resume_dir = None,
    strategy="nodal_coupling",
    ):

    # ------------------------
    # Setup context + data
    # ------------------------
    ctx = SimContext(None)                            # default: robot_test = True
    train_paths = load_shooter_paths(ctx)             # 600 generated paths

    if resume and resume_dir is not None:
        resume_dir = Path(resume_dir)
        dir_ckpt = resume_dir
        dir_log  = resume_dir.parent / "logs"
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_name = f"ddqn_run_{timestamp}"
        dir_log  = ensure_dir(RL_MODEL_DIR / run_name / "logs")
        dir_ckpt = ensure_dir(RL_MODEL_DIR / run_name / "checkpoints")

    # Print TensorBoard path for convenience
    abs_log = dir_log.resolve()
    print("\n==================== TensorBoard ====================")
    print(f"Log directory: {abs_log}")
    print("Start TensorBoard with:")
    print(f"  tensorboard --logdir \"{abs_log}\"")
    print("=====================================================\n")

    writer = tf.summary.create_file_writer(str(dir_log))
    env_template = ShooterEnv(ctx, train_paths[0], strategy=strategy)
    agent = DQNAgent(env_template, ctx)

    # ------------------------
    # Resume safety settings
    # ------------------------
    WARMUP_REPLAY_MIN = 5000
    RESET_EPSILON_ON_RESUME = False
    WARMUP_PER_ALPHA = 0.4
    POST_WARMUP_PER_ALPHA = agent.replay.alpha
    LR_COOLDOWN_STEPS = 2000
    original_lr = agent.optimizer.learning_rate.numpy()

    # ------------------------
    # Resume checkpoint
    # ------------------------
    episode_counter = 0
    start_episode = 0
    start_epoch = 0
    ckpt_qnet = dir_ckpt / "q_net.h5"

    if resume and ckpt_qnet.exists():
        try:
            meta = agent.load(dir_ckpt)
            last_episode = meta.get("episode", 0)
            last_epoch   = meta.get("epoch", 0)

            # Resume from NEXT epoch
            start_epoch = last_epoch + 1

            # Continue episode counter (do NOT reset)
            episode_counter = last_episode

            print(f"[Resume] Loaded: last_epoch={last_epoch}, starting at epoch={start_epoch}")
            print(f"[Resume] Episodes so far: {episode_counter}")

            if RESET_EPSILON_ON_RESUME:
                print("[Resume] Resetting epsilon schedule.")
                agent.epsilon = 1.0   # but NOT episode counter

            print("[Resume] Entering LR cooldown phase.")
            agent.optimizer.learning_rate.assign(original_lr * 0.3)
            lr_cooldown_remaining = LR_COOLDOWN_STEPS

            print("[Resume] Applying temporary PER warmup alpha.")
            agent.replay.alpha = WARMUP_PER_ALPHA

        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            episode_counter = 0
            start_epoch = 0

    # ------------------------
    # warm-start (supervised)
    # ------------------------
    if not resume or start_epoch == 0:
        print("[Warm-start] Pretraining MLP-Q...")
        warm_start(agent, ctx, train_paths, n_iters=200, strategy=strategy)
        agent.target_net.set_weights(agent.q_net.get_weights())
    else:
        print("[Warm-start] Skipped (resuming training).")

    # ------------------------
    # TRAIN LOOP
    # ------------------------
    best_reward = -np.inf
    ema_reward = -np.inf
    for epoch in range(start_epoch, n_epochs):
        print(f"\n===== Starting Epoch {epoch+1}/{n_epochs} =====")
        random.shuffle(train_paths)
        epoch_reward = []
        epoch_dv = []

        for path_idx, path in enumerate(train_paths):

            # Replay warmup check
            replay_ready = len(agent.replay) >= WARMUP_REPLAY_MIN

            stats = _run_episode(agent, ctx, path, strategy,
                                 train_online=replay_ready,
                                 step_update_freq=step_update_freq)

            epoch_reward.append(stats['r_tot'])
            epoch_dv.append(stats['v'])

            print(
                f"Epoch {epoch+1}, Ep {episode_counter:05d} | "
                f"n={stats['n']:.1f} | "
                f"V={stats['v']:.1f} | "
                f"R={stats['r_tot']:.3f} | "
                f"Eps={agent.epsilon:.2f} | Replay={len(agent.replay)}"
            )

            # LR cooldown removal
            if replay_ready and 'lr_cooldown_remaining' in locals():
                lr_cooldown_remaining -= 1
                if lr_cooldown_remaining <= 0:
                    print("[Resume] LR cooldown complete → restoring LR and PER.")
                    agent.optimizer.learning_rate.assign(original_lr)
                    agent.replay.alpha = POST_WARMUP_PER_ALPHA
                    del lr_cooldown_remaining

            # Logging
            _log_metrics(writer, episode_counter, stats, agent)

            # Epsilon PER params
            episodes_per_epoch = len(train_paths)
            update_epsilon(agent, episode_counter, n_epochs, episodes_per_epoch)
            update_beta(agent, episode_counter, n_epochs, episodes_per_epoch)

            episode_counter += 1

            # Periodic offline updates
            if replay_ready and episode_counter % offline_freq == 0:
                _offline_retrain(agent, n_updates=offline_updates)


        # -------------------------
        # Evaluate training
        # -------------------------
        m_r, s_r, ci_r = summarize_epoch(epoch_reward)
        m_v, s_v, ci_v = summarize_epoch(epoch_dv)

        with writer.as_default():
            tf.summary.scalar("train/reward_mean", m_r, step=epoch)
            tf.summary.scalar("train/reward_std",  s_r, step=epoch)
            tf.summary.scalar("train/reward_ci95", ci_r, step=epoch)
            tf.summary.scalar("train/victims_mean", m_v, step=epoch)
            tf.summary.scalar("train/victims_std",  s_v, step=epoch)
            tf.summary.scalar("train/victims_ci95", ci_v, step=epoch)

        # -------------------------
        # Save per epoch
        # -------------------------
        ckpt_path = ensure_dir(dir_ckpt / f"epoch{epoch+1:03d}")
        agent.save(str(ckpt_path), episode=episode_counter, epoch=epoch)
        print(f"[Checkpoint] Saved periodic checkpoint: {ckpt_path}")

        # -------------------------
        # Early stopping
        # -------------------------
        PATIENCE = 5
        EMA_ALPHA = 0.2
        ema_reward = m_r if epoch == 0 else (
            EMA_ALPHA * m_r + (1 - EMA_ALPHA) * ema_reward
            )

        with writer.as_default():
            tf.summary.scalar("train/reward_ema", ema_reward, step=epoch)
        writer.flush()

        if epoch == 0 or (ema_reward > best_reward):
            best_reward = ema_reward
            best_epoch = epoch
            stall = 0
            best_path = ensure_dir(dir_ckpt / "best")
            agent.save(str(best_path), episode=episode_counter, epoch=epoch)
        else:
            stall += 1

        if stall >= PATIENCE:
            print(f"[Early stop] Converged. Best epoch={best_epoch}")
            break

    return agent, list(epoch_reward)

if __name__ == "__main__":
    agent, rewards = train(
        n_epochs=50,
        offline_freq=200,
        offline_updates=200,
        step_update_freq=10,
        resume = False,
        resume_dir = None
    )
