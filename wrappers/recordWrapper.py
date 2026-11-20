import time
from collections import deque

import numpy as np


class RecordEpisodeStatistics:
    """Record episodic return/length for the (custom 4-tuple) RLOR envs.

    Self-contained wrapper: it does NOT inherit gymnasium.Wrapper, because the
    gymnasium base class changed its step/reset contract and attribute
    forwarding. Here we hold the wrapped env explicitly and forward unknown
    attributes via __getattr__, keeping the original 4-tuple (obs, reward,
    done, info) behaviour unchanged.
    """

    def __init__(self, env, deque_size=100):
        self.env = env
        self.num_envs = getattr(env, "num_envs", 1)
        self.n_traj = env.n_traj
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def __getattr__(self, name):
        # Forward any attribute we don't define (observation_space,
        # action_space, metadata, seed, ...) to the wrapped env.
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)
        self.episode_returns = np.zeros((self.num_envs, self.n_traj), dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.finished = [False] * self.num_envs
        return observations

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        else:
            infos = list(infos)  # Convert infos to mutable type
        for i in range(len(dones)):
            if dones[i].all() and not self.finished[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return.copy(),
                    "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.finished[i] = True

        if self.is_vector_env:
            infos = tuple(infos)
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )
