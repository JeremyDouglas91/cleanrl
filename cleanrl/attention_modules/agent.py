import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, hidden=64):
        super().__init__()
        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, add_bias_kv=True)
        self.network = nn.Sequential(
            layer_init(nn.Linear(n_obs, hidden)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, n_act), std=1.0),
        )
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def append_cls_token(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        return torch.cat((cls_tokens, x), dim=1)

    def get_value(self, x):
        # x = self.append_cls_token(x)
        # x, _ = self.attention(x, x, x, need_weights=False)
        # return self.critic(x[:, 0, :])
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        # x = self.append_cls_token(x)
        # x, _ = self.attention(x, x, x, need_weights=False)  # self attention
        # x = x[:, 0, :]  # (batch, obs, d_model) -> (batch, d_model)
        x = self.network(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == '__main__':
    # params
    fp = 1
    n_obs_env = 4
    n_obs = n_obs_env * (1 + fp)  # n_obs_env * (integer + fp)
    d_model = 1  # n_obs + 1
    n_act = 2

    from utils import ObsTransformer

    transformer = ObsTransformer(n_obs=n_obs_env, fp=fp)
    obs = np.array([0.94641704, -0.12679185, 0.88343427, -0.04530228])
    print("obs:", obs.shape, obs, sep='\n')
    obs_transformed = torch.Tensor(transformer.transform(obs)).unsqueeze(0)
    print("obs_transformed:", obs_transformed.shape, obs_transformed, sep='\n')

    agent = Agent(n_obs=n_obs, n_act=n_act)

    action, log_prob, entropy, value = agent.get_action_and_value(obs_transformed)

    print("action:", action)
    print("log_prob:", log_prob)
    print("entropy:", entropy)
    print("value:", value)
