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

import os

import numpy as np
import matplotlib.pyplot as plt

"""
Used for plotting test results
"""


def plot_trim_with_setpoint(
    title: str, epoch, plot_dir: str, t, states, actions, t_setpoint
):
    """Plot states and actions"""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # actions
    fig, axs = plt.subplots(3)
    axs[0].set_ylim([-1.1, 1.1])
    axs[1].set_ylim([-1.1, 1.1])
    axs[2].set_ylim([-1.1, 1.1])
    axs[0].plot(t, actions[:, 0], label="rpm")
    axs[1].plot(t, actions[:, 1:3], label=["de", "dr"])
    axs[2].plot(t, actions[:, 3:5], label=["lcg", "vbs"])
    for ax in axs:
        ax.legend()
        ax.grid()

    fig.suptitle(title)
    plt.savefig(plot_dir + f"{epoch}_actions")
    plt.close()

    # states
    fig, axs = plt.subplots(2)
    # axs[1].set_ylim([-1.6, 1.6])
    axs[0].plot(t, states[:, 0:3], label=["x", "y", "z"])  # z
    axs[0].plot(t, t_setpoint[:, 0], "--", label="x setpoint")  # x setpoint
    axs[0].plot(t, t_setpoint[:, 1], "--", label="y setpoint")  # y setpoint
    axs[0].plot(t, t_setpoint[:, 2], "--", label="z setpoint")  # z setpoint
    axs[1].plot(t, states[:, 3:6], label=["phi", "theta", "psi"])  # theta
    axs[1].plot(t, t_setpoint[:, 4], "k--", label="theta setpoint")  # theta setpoint
    for ax in axs:
        ax.legend()
        ax.grid()

    fig.suptitle(title)
    plt.savefig(plot_dir + f"{epoch}_states")
    plt.close()


def plot_traj_3d(title: str, epoch, plot_dir, t, states, setpoint):
    """3D trajectory plot"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(states[:, 0], states[:, 1], states[:, 2], "k-", label="sim")
    ax.plot(states[:1, 0], states[:1, 1], states[:1, 2], "go", label="start")
    ax.plot(states[-1, 0], states[-1, 1], states[-1, 2], "ro", label="end")
    ax.plot(setpoint[0], setpoint[1], setpoint[2], "ko", label="setpoint")

    # format
    ax.set_xlabel("$x~[m]$")
    ax.set_ylabel("$y~[m]$")
    ax.set_zlabel("$z~[m]$")
    plt.legend()
    fig.suptitle(title)
    plt.savefig(plot_dir + f"{epoch}_traj_3d")


def plot_traj_2d(title: str, epoch, plot_dir, t, states, setpoint):
    """2D XY trajectory plot"""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(states[:, 0], states[:, 1], "k-", label="sim")
    ax.plot(states[:1, 0], states[:1, 1], "go", label="start")
    ax.plot(states[-1, 0], states[-1, 1], "ro", label="end")
    # ax.plot(setpoint[0], setpoint[1], "ko", label="setpoint")

    step = 500
    print(f"states.shape[-2] : {states.shape[-2]}")
    for i in range(states.shape[-2] // step):
        print(i)
        idx = i * step
        r = 1
        x_0 = states[idx, 0]
        y_0 = states[idx, 1]
        psi_0 = states[idx, 5]  # psi
        plt.arrow(
            x_0,
            y_0,
            r * np.cos(psi_0),
            r * np.sin(psi_0),
            color="blue",
            head_length=1,
            head_width=1,
        )
    # arrow at the stop
    r = 1
    x_1 = states[-1, 0]
    y_1 = states[-1, 1]
    psi_1 = states[-1, 5]  # psi
    plt.arrow(
        x_1,
        y_1,
        r * np.cos(psi_1),
        r * np.sin(psi_1),
        color="blue",
        head_length=1,
        head_width=1,
    )

    # format
    ax.set_xlabel("$x~[m]$")
    ax.set_ylabel("$y~[m]$")
    plt.legend()
    fig.suptitle(title)
    plt.savefig(plot_dir + f"{epoch}_traj_2d")
