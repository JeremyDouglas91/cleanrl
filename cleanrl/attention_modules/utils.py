import numpy as np
from time import time


# def explode(x, precision):
#     arr = np.array(list((f'%.{precision}f' % x).replace('.','')), dtype=int)
#     # if x < 0:
#     #     arr *= -1.
#     return arr


# def transform_obs(obs, fp):
#     """
#     tranform a single observation:
#     (n_obs,)
#     -->
#     (n_obs_new, n_obs_new + 1)
#     """
#     n_obs = obs.shape[0]

#     # Placeholder for new obs
#     n_obs_new = n_obs * (2 + fp)
#     new_obs = np.empty((
#         n_obs_new,
#         n_obs_new + 1  # encoding + value --> d_model
#     ))

#     # variable encoding
#     new_obs[:, :n_obs_new] = np.eye(n_obs_new)

#     # Value
#     temp = np.empty((n_obs, 2 + fp))
#     temp[:, 0] = obs > 0.  # sign bit
#     temp[:, 1:] = np.vstack(
#         [explode(np.abs(x), fp) for x in np.round(obs, fp)])
#     new_obs[:, n_obs_new] = temp.reshape((n_obs_new,))

#     return new_obs


class ObsTransformer:
    def __init__(self, n_obs, fp):
        self.n_obs = n_obs
        self.fp = fp
        self.n_obs_new = n_obs * (1 + fp)
        self.placeholder = np.zeros((
            self.n_obs_new,
            self.n_obs_new + 1  # encoding + value --> d_model
        ))
        self.placeholder[:, :self.n_obs_new] = np.eye(self.n_obs_new)

    def explode(self, x, precision):
        digits = list(
            (f'%.{precision}f' % abs(round(x, precision))).replace('.', '')
        )
        sign = np.sign(x)
        return [
            sign * int(digit)/10**i for i, digit in enumerate(digits)
        ]

    def transform(self, obs):
        temp = np.hstack([self.explode(x, self.fp) for x in obs])
        return temp
        # self.placeholder[:, self.n_obs_new] = temp.reshape((self.n_obs_new,))
        # return self.placeholder


if __name__ == '__main__':
    transformer = ObsTransformer(n_obs=4, fp=2)

    obs = np.array([0.94641704, -0.12679185, 0.88343427, -0.04530228])
    print("obs:", obs.shape, obs, sep='\n')
    t0 = time()
    out = transformer.transform(obs)
    print("time:", time() - t0)
    print("obs_transformed", out.shape, out, sep="\n")

    obs = np.array([-0.94641704, 0.12679185, -0.88343427, 0.04530228])
    print("obs:", obs.shape, obs, sep='\n')
    t0 = time()
    out = transformer.transform(obs)
    print("time:", time() - t0)
    print("obs_transformed", out.shape, out, sep="\n")
