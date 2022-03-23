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

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

"""
Define Stablebaselines3 callbacks for training.
"""


class TensorboardCallback(BaseCallback):
    """Plotting in tensorboard"""

    def __init__(self, action_dim, state_dim, ep_length, n_envs, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.reward = 0
        self.ts = 0

        self.ep_length = ep_length
        self.actions = np.zeros([ep_length, action_dim])
        self.states = np.zeros([ep_length, state_dim])
        self.t = np.linspace(0, ep_length, ep_length).astype(int)
        self.episode = 0
        self.n_envs = n_envs

    def _on_rollout_end(self) -> None:
        self.logger.record("reward", self.reward)

        # if self.episode % self.n_envs == 0:
        #     self.plot_trim()
        #     self.plot_traj_3d()

        self.logger.dump(self.num_timesteps)
        self.reward = 0
        self.ts = 0
        self.episode += 1

    def _on_step(self) -> bool:
        reward = self.training_env.get_attr("reward")[0]
        # state = self.training_env.get_attr("full_state")[0]
        # action = self.training_env.get_attr("prev_action")[0]

        self.reward += reward
        # self.states[self.ts] = state
        # self.actions[self.ts] = action
        self.ts += 1

        return True

    def plot_trim(self) -> None:
        """dsds"""
        # fig, axs = plt.subplots(3)
        # axs[0].set_ylim([-1.1, 1.1])
        # axs[1].set_ylim([-1.1, 1.1])
        # axs[2].set_ylim([-1.1, 1.1])
        # axs[0].plot(self.t, self.actions[:, 0], label="rpm")
        # axs[1].plot(self.t, self.actions[:, 1:3], label=["de", "dr"])
        # axs[2].plot(self.t, self.actions[:, 3:5], label=["lcg", "vbs"])
        # for ax in axs:
        #     ax.legend()
        #     ax.grid()
        fig = plt.figure(1)
        plt.plot(self.t, self.actions, label=["lcg", "vbs"])
        plt.legend()
        plt.grid()
        self.logger.record(
            f"trim/{self.episode}_actions",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close()

        fig, axs = plt.subplots(2)
        axs[1].set_ylim([-1.6, 1.6])
        axs[0].plot(self.t, self.states[:, 0:3], label=["x", "y", "z"])  # z
        axs[1].plot(self.t, self.states[:, 3:6], label=["phi", "theta", "psi"])  # theta
        for ax in axs:
            ax.legend()
            ax.grid()
        self.logger.record(
            f"trim/{self.episode}_states",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close()

    def plot_traj_3d(self):
        """Plot 3D the trajectory"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            self.states[0 : self.ep_length - 2, 0],
            self.states[0 : self.ep_length - 2, 1],
            self.states[0 : self.ep_length - 2, 2],
            "k-",
            label="sim",
        )
        ax.plot(
            self.states[:1, 0],
            self.states[:1, 1],
            self.states[:1, 2],
            "go",
            label="start",
        )
        ax.plot(
            self.states[-1, 0],
            self.states[-1, 1],
            self.states[-1, 2],
            "ro",
            label="end",
        )
        ax.plot(0.0, 0.0, 0.0, "ko", label="target")

        # format
        ax.set_xlabel("$x~[m]$")
        ax.set_ylabel("$y~[m]$")
        ax.set_zlabel("$z~[m]$")
        plt.legend()
        self.logger.record(
            f"trim/{self.episode}_traj_3d",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close()


class SaveVecNormalizeCheckpoint(BaseCallback):
    """Save VecNormalize into pkl during training"""

    def __init__(
        self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0
    ):
        super(SaveVecNormalizeCheckpoint, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            print("Saving model...")
            if hasattr(self.model, "save"):
                model_path = "{}/{}_{}_steps".format(
                    self.save_path, self.name_prefix, self.num_timesteps
                )
                self.model.save(model_path)
            else:
                print(
                    f"[WARN] Model {self.name_prefix} not saved at timestep {self.num_timesteps}"
                )

            if hasattr(self.training_env, "save"):
                env_path = "{}/{}_{}_steps_env.pkl".format(
                    self.save_path, self.name_prefix, self.num_timesteps
                )
                self.training_env.save(env_path)
        return True
