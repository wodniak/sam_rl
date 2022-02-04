#!/usr/bin/env python3

# Copyright 2022 Grzegorz Wozniak (gwozniak@kth.se)
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided
# that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
#  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Deep Deterministic Policy Gradients (DDPG) for continuous control

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


# Source : https://keras.io/examples/rl/ddpg_pendulum/
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Actor(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension, max_action):
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(state_dimension, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, action_dimension)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()
        # State
        self.s1 = torch.nn.Linear(state_dimension, 16)
        self.s_out = torch.nn.Linear(16, 32)

        # Action
        self.a_out = torch.nn.Linear(action_dimension, 32)

        # common
        self.c1 = torch.nn.Linear(64, 256) # a_out + s_out
        self.c2 = torch.nn.Linear(256, 256)
        self.c_out = torch.nn.Linear(256, 1)

    def forward(self, state, action):
        # forward action
        a_out = F.relu(self.a_out(action))

        # forward state
        s_out = F.relu(self.s1(state))
        s_out = F.relu(self.s_out(s_out))

        # forward concat
        c = torch.cat([s_out, a_out], 1)
        out = F.relu(self.c1(c))
        out = F.relu(self.c2(out))
        return self.c_out(out)


class ReplayBuffer(object):

    def __init__(self, state_dimension, action_dimension, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dimension))
        self.action = np.zeros((max_size, action_dimension))
        self.next_state = np.zeros((max_size, state_dimension))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def is_ready(self, batch_size):
        return self.size > batch_size


class DDPGAgent(object):

    def __init__(self, state_dim, action_dim, max_action, device, discount, tau, actor_lr, critic_lr):
        self.device = device
        self.discount = discount
        self.tau = tau

        max_action = torch.FloatTensor(max_action).to(self.device)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    @ staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, filename):
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(),
                   filename + '_critic_optimizer')
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(),
                   filename + '_actor_optimizer')

    def load_checkpoint(self, filename):
        self.critic.load_state_dict(
            torch.load(
                filename + "_critic",
                map_location=torch.device('cpu')
            )
        )
        self.critic_optimizer.load_state_dict(
            torch.load(
                filename + "_critic_optimizer",
                map_location=torch.device('cpu')
            )
        )
        self.actor.load_state_dict(
            torch.load(
                filename + "_actor",
                map_location=torch.device('cpu')
            )
        )
        self.actor_optimizer.load_state_dict(
            torch.load(
                filename + "_actor_optimizer",
                map_location=torch.device('cpu')
            )
        )

        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)
        target_q = self.critic_target(
            next_state, self.actor_target(next_state))
        target_q = reward + (not_done * self.discount * target_q).detach()
        current_q = self.critic(state, action)

        # Critic
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update
        DDPGAgent.soft_update(self.critic, self.critic_target, self.tau)
        DDPGAgent.soft_update(self.actor, self.actor_target, self.tau)
