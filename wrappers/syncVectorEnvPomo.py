from copy import deepcopy
from typing import List, Optional, Union

import numpy as np

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv:
    """Vectorized environment that serially runs multiple RLOR envs.

    Self-contained: it does NOT inherit gymnasium's VectorEnv, and does NOT use
    gymnasium.vector.utils. The original code relied on gym's internal
    concatenate / create_empty_array / iterate helpers, whose signatures changed
    across the gym -> gymnasium migration. The RLOR observation is always a dict
    of numpy arrays, so we batch it by stacking along a new leading num_envs
    axis. Behaviour (4-tuple step, dict obs batching) is unchanged.
    """

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.num_envs = len(env_fns)
        self.metadata = getattr(self.envs[0], "metadata", {})
        self.n_traj = self.envs[0].n_traj
        self.is_vector_env = True

        self.single_observation_space = (
            observation_space or self.envs[0].observation_space
        )
        self.single_action_space = action_space or self.envs[0].action_space
        # gym/gymnasium vector envs historically expose these too.
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        self._check_spaces()
        self._rewards = np.zeros((self.num_envs, self.n_traj), dtype=np.float64)
        self._dones = np.zeros((self.num_envs, self.n_traj), dtype=np.bool_)
        self._actions = None

    # ------------------------------------------------------------------ #
    # batching helper: stack a list of per-env obs dicts into one dict    #
    # with a leading num_envs axis. Replaces gym.vector.utils.concatenate #
    # ------------------------------------------------------------------ #
    def _batch_obs(self, observations):
        # Stack per-env obs dicts along a new leading num_envs axis, and cast
        # each key to the dtype declared by its (sub)observation space. The old
        # gym.vector.utils.concatenate copied into a pre-allocated array whose
        # dtype came from the space (e.g. Box -> float32); np.stack alone would
        # keep the source dtype (e.g. float64 from np.random.rand), which breaks
        # the float32 model weights. So we cast explicitly to match.
        space = self.single_observation_space
        keys = observations[0].keys()
        batched = {}
        for k in keys:
            arr = np.stack([np.asarray(obs[k]) for obs in observations], axis=0)
            subspace = space[k] if hasattr(space, "__getitem__") else None
            if subspace is not None and getattr(subspace, "dtype", None) is not None:
                arr = arr.astype(subspace.dtype, copy=False)
            batched[k] = arr
        return batched

    def seed(self, seed=None):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs
        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset(self, seed=None, options=None, return_info=False):
        return self.reset_wait(seed=seed, return_info=return_info, options=options)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._dones[:] = False
        observations = []
        data_list = []
        for env, single_seed in zip(self.envs, seed):
            if single_seed is not None:
                env.seed(single_seed)

            if not return_info:
                observation = env.reset()
                observations.append(observation)
            else:
                observation, data = env.reset()
                observations.append(observation)
                data_list.append(data)

        self.observations = self._batch_obs(observations)
        if not return_info:
            return deepcopy(self.observations) if self.copy else self.observations
        else:
            return (
                deepcopy(self.observations) if self.copy else self.observations
            ), data_list

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        # actions: array shape (num_envs, n_traj) -> iterate over leading axis
        self._actions = list(actions)

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            observations.append(observation)
            infos.append(info)
        self.observations = self._batch_obs(observations)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)
        return tuple(results)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )
        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        [env.close() for env in self.envs]

    def close(self, **kwargs):
        self.close_extras(**kwargs)

    def _check_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )
            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )
        return True
