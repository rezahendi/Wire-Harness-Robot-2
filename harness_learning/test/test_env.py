import numpy as np
from gymnasium.utils.env_checker import check_env

from harness_learning.env import HarnessRoutingEnv


def test_env_api():
    env = HarnessRoutingEnv()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=4)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert set(env.obs_layout) >= {"tcp_pos", "wrench", "cable", "forks", "progress"}
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.isfinite(r) and not term and not trunc


def test_replay_is_deterministic():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (30, 5)).astype(np.float32)
    actions[:, 2] = np.abs(actions[:, 2])          # stay away from the board
    runs = []
    for _ in range(2):
        env = HarnessRoutingEnv()
        obs, _ = env.reset(seed=21)
        traj = [obs]
        for a in actions:
            traj.append(env.step(a)[0])
        runs.append(np.array(traj))
    assert np.array_equal(runs[0], runs[1])
