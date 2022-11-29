import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli

import time


class LSTMAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model, rnn_h=32):
        super().__init__()
        # attention Base
        self.network = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
            bias=False,
            add_bias_kv=False,
        )

        # recurrent layer
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=rnn_h)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # actor
        self.actor_action = layer_init(nn.Linear(rnn_h, act_dim), std=0.01)
        self.actor_mask = layer_init(nn.Linear(rnn_h, obs_dim), std=0.01)

        # critic
        self.critic = layer_init(nn.Linear(rnn_h, 1), std=1.)

    def get_states(self, x, lstm_state, done, mask):
        # transform 2D mask into 3D mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = 1 - mask
        # https://discuss.pytorch.org/t/batch-outer-product/4025
        mask = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1)).to(torch.bool)

        # (batch, seq, d_model) -> (batch, seq, d_model)
        hidden, _ = self.network(x, x, x, attn_mask=mask)  # self attention
        hidden = hidden.sum(axis=1)  # (batch, d_model)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done, mask):
        hidden, _ = self.get_states(x, lstm_state, done, mask)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, mask, action_tuple=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done, mask)

        # (batch, seq, rnn_h) --> (batch, seq, n_acts)
        logits = self.actor_action(hidden)
        probs_action = Categorical(
            logits=logits
        )
        # (batch, seq, n_acts) --> (batch, seq, 1)
        if action_tuple is None:
            action = probs_action.sample()

        # (batch, seq, rnn_h) --> (batch, seq, n_obs)
        logits = self.actor_mask(hidden)
        probs_mask = Bernoulli(
            logits=logits,
        )
        if action_tuple is None:
            # (batch, seq, n_obs) --> (batch, seq, n_obs)
            mask = probs_mask.sample()
        else:
            action, mask = action_tuple

        # compute log probs (indepedence: log(p) = log(p_act) + sum(log_p([p_mask]))
        log_prob = probs_action.log_prob(action) + \
            probs_mask.log_prob(mask).sum(axis=1)  # (batch, 1)
        entropy = probs_action.entropy() + \
            probs_mask.entropy().sum(axis=1)  # H_total = \sum_i(H_i)

        return (action, mask), log_prob, entropy, self.critic(hidden), lstm_state


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def policy_test():
    # params
    n_obs = 2
    n_act = 2
    fp = 1  # floating point precision
    n_obs_new = n_obs * (2 + fp)
    d_model = n_obs_new+1
    rnn_h = 32
    batch_size = 2

    # agent
    policy = LSTMAgent(
        obs_dim=n_obs_new,
        act_dim=n_act,
        d_model=d_model,
        rnn_h=rnn_h,
    )

    # Dummy data
    batch = torch.rand((batch_size, n_obs_new, d_model))
    done = torch.randint(low=0, high=2, size=(batch_size,))
    next_lstm_state = (
        torch.zeros(1, 1, rnn_h),
        torch.zeros(1, 1, rnn_h),
    )
    mask = torch.Tensor([[1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1]])

    print(
        "batch_size: %d" % batch_size,
        "obs: %d" % n_obs_new,
        "d_model: %d" % d_model,
        "input shape:", batch.shape,
        sep='\n'
    )

    # warm up
    (action, mask), log_prob, entropy, value, lstm_state = policy.get_action_and_value(
        batch, next_lstm_state, done, mask
    )

    # Dummy data
    batch = torch.rand((batch_size, n_obs_new, d_model))
    done = torch.randint(low=0, high=2, size=(batch_size,))
    next_lstm_state = (
        torch.zeros(1, 1, rnn_h),
        torch.zeros(1, 1, rnn_h),
    )
    mask = torch.Tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])

    # timeit
    t0 = time.time()
    (action, mask), log_prob, entropy, value, lstm_state = policy.get_action_and_value(
        batch, next_lstm_state, done, mask
    )
    T = time.time() - t0

    print(
        f"forward pass time: {T}",
        f"action: {action.shape}",
        f"mask: {mask.shape}",
        f"log_prob: {log_prob.shape}",
        f"value: {value.shape}",
        sep="\n"
    )


if __name__ == '__main__':
    policy_test()
