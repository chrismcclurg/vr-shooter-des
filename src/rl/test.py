import os
import numpy as np
from tqdm import tqdm
from src.des import SimContext
from src.rl.envs import ShooterEnv
from src.rl.dqn import DQNAgent
from src.utils.rl import load_shooter_paths, numeric_heuristic
from src.utils.paths import RL_MODEL_DIR

def _run_episode(ctx, path, event_type, policy_fn, show_tqdm=False):
    env     = ShooterEnv(ctx, path, strategy=event_type)
    obs, _  = env.reset()
    done    = False
    final_t = ctx.final_time
    last_t  = 0.0
    with tqdm(total=final_t, desc="Episode time (s)", unit="s", leave=False,
              disable=not show_tqdm) as pbar:
        while not done:
            action = policy_fn(env)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            curr_t = info["time"]
            pbar.update(curr_t - last_t)
            last_t = curr_t

    return info

def run_policy(ctx, shooter_paths, event_type, mv, mn,
               ro1_policy=1, ro2_policy=1,
               ro1_tar_node=1, ro2_tar_node=281,
               checkpoint_dir=None,show_baseline=False,
               hidden_dim=64,alpha=None,desc=""):

    ro1_tar_idx=ctx.node_idx_map[ro1_tar_node]
    ro2_tar_idx=ctx.node_idx_map[ro2_tar_node]

    # Determine policy function
    use_tqdm = False
    dump_cache = lambda: None

    if checkpoint_dir is not None:
        # print(f"[INFO] Loading trained RL policy from: {checkpoint_dir}")
        env_template = ShooterEnv(ctx, shooter_paths[0], strategy=event_type)
        agent = DQNAgent(env_template, ctx, hidden_dim)
        agent.epsilon = 0.0
        _ = agent.load(checkpoint_dir)
        policy_fn = lambda env: agent.select_action(env._get_obs(), env)
    else:
        # print("[INFO] Using heuristic policy.")
        policy_fn = lambda env: numeric_heuristic(
            ro1_policy, ro2_policy, env, ctx, ro1_tar_idx, ro2_tar_idx
        )

    # Run evaluations
    victims_all, nodes_all = [], []
    for path in tqdm(shooter_paths, desc=f"[Policy] {desc}", leave=False):
        info = _run_episode(ctx, path, event_type, policy_fn, use_tqdm)
        victims_all.append(info.get("victims", 0))
        nodes_all.append(info.get("nodes", 0))
        dump_cache()

    # Summary stats
    mean_v, std_v = float(np.mean(victims_all)), float(np.std(victims_all))
    mean_n, std_n = float(np.mean(nodes_all)),   float(np.std(nodes_all))
    pd_v = np.round((mean_v - mv) / mv * 100., 2)
    pd_n = np.round((mean_n - mn) / mn * 100., 2)

    print(
        f"[Policy] {desc:<20} | "
        f"Victims: ${mean_v:6.2f} \pm {std_v:6.2f}$  &  ${pd_v:+5.1f}\%$ | "
        f"Nodes:   ${mean_n:6.2f} ± {std_n:6.2f}$   &  ${pd_n:+5.1f}\%"
    )

def run_baseline(ctx, shooter_paths, event_type, desc = "no robot"):
    # run evaluations
    victims_all, nodes_all = [], []
    for path in tqdm(shooter_paths, desc=f"[Policy] {desc}", leave=False):
        env = ShooterEnv(ctx, path, strategy=event_type, robot_type="ignore")
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated
        victims_all.append(info.get("victims", 0))
        nodes_all.append(info.get("nodes", 0))

    # summary stats
    mean_v, std_v = float(np.mean(victims_all)), float(np.std(victims_all))
    mean_n, std_n = float(np.mean(nodes_all)),   float(np.std(nodes_all))

    print(
        f"[Policy] {desc:<20} | "
        f"Victims: ${mean_v:6.2f} \pm {std_v:6.2f}$   {'':8} | "
        f"Nodes:   ${mean_n:6.2f} \pm {std_n:6.2f}$   {'':8}"
    )

    return mean_v, mean_n

if __name__ == "__main__":

    use_actual = True
    event_type="nodal_coupling"

    # no stairs ctx
    ns_ctx = SimContext(None, robot_test=True, robot_train=True, verbose=False, robot_stairs = False)

    # shooter paths E2 and E3
    test_ctx = SimContext(None, robot_test=False, robot_train=False, verbose=False)
    test_paths = load_shooter_paths(test_ctx, use_actual)

    # shooter paths E4 and E5
    train_ctx = SimContext(None, robot_test=True, robot_train=True, verbose=False)
    valid_paths = load_shooter_paths(train_ctx, use_actual)
    all_paths = test_paths + valid_paths

    # # run baseline
    mv, mn = run_baseline(train_ctx, all_paths, event_type)

    # # # run heuristics
    # run_policy(ns_ctx,    all_paths, event_type, mv, mn, 0, 0, 1, 281, desc="base, static")
    # run_policy(train_ctx, all_paths, event_type, mv, mn, 0, 0, 1, 281, desc="stairs, static")
    # run_policy(ns_ctx,    all_paths, event_type, mv, mn, 0, 0, 103, 215, desc="base, min-kr")
    # run_policy(train_ctx, all_paths, event_type, mv, mn, 0, 0, 103, 215, desc="stairs, min-kr")
    # run_policy(ns_ctx,    all_paths, event_type, mv, mn, 0, 0, 101, 204, desc="base, max-kr")
    # run_policy(train_ctx, all_paths, event_type, mv, mn, 0, 0, 101, 204, desc="stairs, max-kr")
    # run_policy(ns_ctx,    all_paths, event_type, mv, mn, 1, 1, desc="base, pursue")
    # run_policy(train_ctx, all_paths, event_type, mv, mn, 1, 1, desc="stairs, pursue")

    # run RL policy
    run_name = "ddqn_run_1227_1635"
    policy_dir = RL_MODEL_DIR / "checkpoints" / run_name / "best"
    run_policy(train_ctx, all_paths, event_type, mv, mn,
               checkpoint_dir=policy_dir, hidden_dim = 32,
               desc= "RL - min dist")