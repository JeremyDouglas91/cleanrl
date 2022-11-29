import gym
from gym.spaces import Box, Discrete
import numpy as np

from .utils import ObsTransformer


class AttentionTaskWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.0, fp=2):
        super().__init__(env)
        self.env = env
        self.fp = fp
        self.penalty = penalty
        self.transformer = ObsTransformer(
            n_obs=self.env.observation_space.shape[0], fp=fp)

        # obs and act space
        self.new_obs_size = self.env.observation_space.shape[0] * (fp + 1)
        self.env.observation_space = Box(
            low=-1e1,
            high=1e1,
            shape=(self.new_obs_size,)  # self.new_obs_size + 1),
        )

    def reset(self):
        return self.transformer.transform(self.env.reset())

    def step(self, action):
        # action = action_mask[0].astype(int)
        # bool_mask = action_mask[1:].astype(bool)
        o, r, d, _ = self.env.step(action)

        # transform observation
        o = self.transformer.transform(o)

        # apply penalty
        # r = self._augment_reward(r, bool_mask)

        return o, r, d, _

    # def _augment_reward(self, reward, bool_mask):
    #     return reward + self.penalty * np.mean(bool_mask)


class RandomSamplingTaskWrapper(gym.Wrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.env = env
        self.n_obs = self.env.observation_space.shape[0]
        self.n_sample = n
        self.env.observation_space = Box(
            low=-1e1,
            high=1e1,
            # number of elements to sample + encoding
            shape=(self.n_sample + self.n_obs,)
        )

    def sample(self, o):
        idx = np.random.choice(self.n_obs, self.n_sample, replace=False)
        idx = np.sort(idx)
        encoding = np.zeros(self.n_obs)
        encoding[idx] = 1.
        return np.hstack([o[idx], encoding])

    def reset(self):
        o = self.sample(self.env.reset())
        return o

    def step(self, action):
        o, r, d, _ = self.env.step(action)
        return self.sample(o), r, d, _


class PositionVelocityTaskWrapper(gym.Wrapper):
    def __init__(self, env, flavour='random'):
        super().__init__(env)
        self.env = env
        self.flavour = flavour
        self.n_obs = self.env.observation_space.shape[0]
        self.env.observation_space = Box(
            low=-1e1,
            high=1e1,
            # pos/vel components + flag
            shape=(self.n_obs//2 + 1,)
        )
        self.position = 0

    def select(self, o):
        if self.flavour == 'random':
            self.position = np.random.choice([0, 1])
        elif self.flavour == 'alternate':
            self.position = abs(self.position-1)  # alternate between 0 and 1

        if self.position:
            o = o[[0, 2]]
        else:
            o = o[[1, 3]]

        return np.hstack([o, [self.position]])

    def reset(self):
        o = self.select(self.env.reset())
        return o

    def step(self, action):
        o, r, d, _ = self.env.step(action)
        return self.select(o), r, d, _


def wrapper_selection(task, fp=None, n=None, flavour=None):
    def make_env(env_id, seed, idx, capture_video, run_name):
        def thunk():
            env = gym.make(env_id)
            if task == 'attention':
                env = AttentionTaskWrapper(env, fp)
            elif task == 'random_sampling':
                env = RandomSamplingTaskWrapper(env, n)
            elif task == 'position_velocity':
                env = PositionVelocityTaskWrapper(env, flavour)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk
    return make_env


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    """
    Attention Wrapper
    """

    # params
    seed = 42
    num_envs = 2
    fp = 2

    # dummy data
    action = np.random.choice([0, 1], num_envs)
    # mask = np.random.choice([0, 1], (num_envs, 4 * (fp + 2)))
    # action_mask = np.concatenate([
    #     np.expand_dims(action, -1),
    #     mask,
    # ], axis=-1)
    print("action:", action.shape, action, sep='\n')
    # print("mask:", mask.shape, mask, sep='\n')
    # print("action_mask:", action_mask.shape, action_mask, sep='\n')

    envs = gym.vector.SyncVectorEnv(
        [
            wrapper_selection('attention', fp=fp)(
                'CartPole-v1', seed + i, i, False, 'test') for i in range(num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs = envs.reset()
    print("obs:", obs.shape, obs, sep='\n')

    obs_new, r, d, _ = envs.step(action)
    print("obs_new:", obs_new.shape, obs_new, sep='\n')
    print("reward:", r)
    print("done:", d)

    """
    Random Sampling Wrapper
    """

    # params
    seed = 42
    num_envs = 2
    n_samples = 2

    # dummy data
    action = np.random.choice([0, 1], num_envs)
    print("action:", action.shape, action, sep='\n')

    envs = gym.vector.SyncVectorEnv(
        [wrapper_selection('random_sampling', n=n_samples)(
            'CartPole-v1', seed + i, i, False, 'test') for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs = envs.reset()
    print("obs:", obs.shape, obs, sep='\n')
    obs_new, r, d, _ = envs.step(action)
    print("obs_new:", obs_new.shape, obs_new, sep='\n')
    print("reward:", r)
    print("done:", d)

    """
    Position-Velocity Wrapper
    """

    # params
    seed = 42
    num_envs = 2
    n_samples = 2

    # dummy data
    action = np.random.choice([0, 1], num_envs)
    print("action:", action.shape, action, sep='\n')

    envs = gym.vector.SyncVectorEnv(
        [wrapper_selection('position_velocity', flavour='alternate')(
            'CartPole-v1', seed + i, i, False, 'test') for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs = envs.reset()
    print("obs:", obs.shape, obs, sep='\n')
    obs_new, r, d, _ = envs.step(action)
    print("obs_new:", obs_new.shape, obs_new, sep='\n')
    print("reward:", r)
    print("done:", d)
