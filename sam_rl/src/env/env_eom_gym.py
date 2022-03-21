#!/usr/bin/env python3

# Copyright 2022 Grzegorz Wozniak (gwozniak@kth.se)
# Copyright 2020 Sriharsha Bhat (svbhat@kth.se)
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

import math
import numpy as np
import gym
import random

from scipy.integrate import solve_ivp
from typing import Optional


def T_b2ned(phi, theta, psi):
    """:return translation matrix in NED frame"""
    T = np.array(
        [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]
    )
    return T


def R_b2ned(phi, theta, psi):
    """:return rotation matrix in NED frame"""
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )
    Ry = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    Rz = np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
    )
    R = Rz @ Ry @ Rx
    return R


def unskew(mat):
    """"""
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])


def skew(vec):
    """"""
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


class EnvEOM(object):
    """
    Simplified numerical simulation of SAM for trim control
    State : (1 x 12) (x, y, z, phi, theta, psi, u, v, w, p, q, r)
        @note : xyz in NED frame
    Action : (1 x 6) (rpm1, rpm2, de, dr, lcg, vbs)
        @note : values are in different ranges, scaled in eom function
    """

    def __init__(self, init_state):
        self.timestep = 0.0
        self.dt = 0.01  # time step

        # system tolerances
        self.atol = 1e-8
        self.rtol = 1e-8
        self.solver_method = "DOP853"

        self.current_x = init_state
        assert self.current_x.shape == (12,), self.current_x

    def eom(self, state, control):
        """
        Nonlinear dynamics model function
        @param control: Action in range (-1, 1)
        """
        # state and control
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        rpm1, rpm2, de, dr, lcg, vbs = control
        # rpm1, rpm2, de, dr = control #(AUV specific)

        # position (NED) and velocity (body), rsp
        eta = np.array([x, y, z, phi, theta, psi])
        nu = np.array([u, v, w, p, q, r])

        # scaled controls (AUV specific)
        rpm1 *= 1000.0
        rpm2 *= 1000.0
        de *= 0.05
        dr *= 0.05
        vbs *= 1.0
        lcg *= 1.0

        # mass and inertia matrix : (AUV specific)
        m = 14.0
        I_o = np.diag(np.array([0.0294, 1.6202, 1.6202]))

        # centre of gravity, buoyancy, and pressure positions, resp. (AUV specific)
        r_g = np.array(
            [0.1 + lcg * 0.01, 0.0, 0.0]
        )  # Including Longitudinal C.G. trim system
        # r_g = np.array([0.1, 0.0, 0.0]) #no effect of Longitudinal C.G. trim system
        r_b = np.array([0.1, 0.0, 0.0])
        r_cp = np.array([0.1, 0.0, 0.0])

        # Buoyancy effects
        W = m * 9.81
        B = W + vbs * 1.5  # (AUV specific if a Variable Buoyancy System exists)
        # B = W #no effect of Variable buoyancy system

        # hydrodynamic coefficients (AUV specific)
        Xuu = 5.0
        Yvv = 20.0
        Zww = 50.0
        Kpp = 0.1
        Mqq = 20.0
        Nrr = 20.0

        # Thruster coefficients (AUV specific)
        K_T = np.array([0.0175, 0.0175])
        Q_T = np.array([0.001, -0.001])  # *0.0

        # mass and inertia matrix
        M = np.block([[m * np.eye(3, 3), -m * skew(r_g)], [m * skew(r_g), I_o]])
        assert M.shape == (6, 6), M

        # coriolis and centripetal matrix
        nu1 = np.array([u, v, w])
        nu2 = np.array([p, q, r])
        top_right = -m * skew(nu1) - m * skew(nu2) * skew(r_g)
        bottom_left = -m * skew(nu1) + m * skew(r_g) * skew(nu2)
        bottom_right = -skew(I_o.dot(nu2))
        C_RB = np.block([[np.zeros((3, 3)), top_right], [bottom_left, bottom_right]])
        assert C_RB.shape == (6, 6), C_RB

        # damping matrix (AUV specific - This layout is for a slender AUV)
        forces = np.diag(np.array([Xuu * np.abs(u), Yvv * np.abs(v), Zww * np.abs(w)]))
        moments = np.diag(np.array([Kpp * np.abs(p), Mqq * np.abs(q), Nrr * np.abs(r)]))
        coupling = np.matmul(skew(r_cp), forces)
        D = np.block([[forces, np.zeros((3, 3))], [-coupling, moments]])
        assert D.shape == (6, 6), D

        # rotational transform between body and NED in Euler
        T_euler = T_b2ned(phi, theta, psi)
        R_euler = R_b2ned(phi, theta, psi)
        assert R_euler.shape == (3, 3), R_euler
        J_eta = np.block([[R_euler, np.zeros((3, 3))], [np.zeros((3, 3)), T_euler]])
        assert J_eta.shape == (6, 6), J_eta

        # buoyancy in quaternions
        f_g = np.array([0, 0, W])
        f_b = np.array([0, 0, -B])
        row1 = np.linalg.inv(R_euler).dot(f_g + f_b)
        row2 = skew(r_g).dot(np.linalg.inv(R_euler)).dot(f_g) + skew(r_b).dot(
            np.linalg.inv(R_euler)
        ).dot(f_b)
        geta = np.block([row1, row2])
        assert geta.shape == (6,), geta

        # Effect of control actuators (AUV specific)
        F_T = K_T.dot(np.array([rpm1, rpm2]))
        M_T = Q_T.dot(np.array([rpm1, rpm2]))
        tauc = np.array(
            [
                F_T * np.cos(de) * np.cos(dr),
                -F_T * np.sin(dr),
                F_T * np.sin(de) * np.cos(dr),
                M_T * np.cos(de) * np.cos(dr),
                -M_T * np.sin(dr),
                M_T * np.sin(de) * np.cos(dr),
            ]
        )
        assert tauc.shape == (6,), tauc

        # rate of change of position in NED frame
        etadot = np.block([J_eta.dot(nu)])
        assert etadot.shape == (6,)

        nudot = np.linalg.pinv(M).dot(tauc - (C_RB + D).dot(nu) - geta)

        # state-space dynamics
        return np.hstack((etadot, nudot))

    def propagate(self, state, action, t0, tf, atol, rtol, method):
        """
        Propagate dynamics function (solve ODE by timestepping)
        """
        # integrate dynamics (also consider - lambda t, x: eom(x, controller(x)),)
        sol = solve_ivp(
            lambda t, x: self.eom(x, action),
            (t0, tf),
            state,
            method=method,
            rtol=rtol,
            atol=atol,
            # jac=lambda t, x: eom_jac(x, controller(x))
        )

        times, states = sol.t, sol.y.T
        return times, states

    def step(self, action):
        """One step in the simulation"""
        # propagate system
        next_ts = self.timestep + self.dt
        t_out, x_out = self.propagate(
            self.current_x,
            action,
            self.timestep,
            next_ts,
            self.atol,
            self.rtol,
            self.solver_method,
        )

        self.timestep = t_out[-1]
        self.current_x = x_out[-1, :]
        return self.timestep, self.current_x

    def get_state(self):
        """:return current state"""
        return self.current_x

    def set_state(self, state):
        """:param state - set new current state"""
        self.current_x = state


class EnvEOMGym(gym.Env):
    """
    Define common API to env and cost function for the task
    Learns to control trim (depth and pitch angle)
    """

    def __init__(self, episode_length, env_obs_states, env_actions, num_envs=1):
        super(EnvEOMGym, self).__init__()

        self.env_obs_states = env_obs_states  # defined in params
        self.env_actions = env_actions
        self.key_to_state_map = {
            "x": 0,
            "y": 1,
            "z": 2,
            "phi": 3,
            "theta": 4,
            "psi": 5,
            "u": 6,
            "v": 7,
            "w": 8,
            "p": 9,
            "q": 10,
            "r": 11,
        }

        action_size = len(env_actions)
        action_high = np.ones(action_size)
        self.action_space = gym.spaces.Box(low=-action_high, high=action_high)

        # x, y, z, phi, theta, psi, u, v, w, p, q, r
        obs_high = self._make_obs_high()
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high)
        self.ep_length = episode_length
        self.current_step = 0
        self.num_resets = 0
        self.num_envs = num_envs

        # EOM simulation
        self.init_state = self._reset_uniform()
        self.prev_action = np.zeros(action_size)
        self.dynamics = EnvEOM(self.init_state)

        # For logger, accessed with 'get_attr'
        self.reward = 0
        self.full_state = np.zeros(12)

        print(
            f"Running with Env:\n\
            Observed states:{env_obs_states.keys()}\n\
            Obs state high:{obs_high}\n\
            Actions:{env_actions.keys()}\n"
        )

    def _make_obs_high(self):
        obs_high = []
        for key in self.env_obs_states.keys():
            value = self.env_obs_states[key]
            obs_high.append(value)
        return np.array(obs_high)

    def _make_action_6d(self, input_action):
        action_6d = np.zeros(6)
        for i, key in enumerate(self.env_actions):
            pos = self.env_actions[key]
            value = input_action[i]
            action_6d[pos] = value
            if key == "rpm":
                action_6d[0] = value  # rpm is the same for both propellers
        return action_6d

    def _get_obs(self):
        """return observed state from 12d state dynamics"""
        state_12d = self.dynamics.get_state()
        obs = []
        for key in self.key_to_state_map.keys():
            if key in self.env_obs_states:
                state_pos = self.key_to_state_map[key]
                obs.append(state_12d[state_pos])
        return np.array(obs)

    def _reset_uniform(self):
        xyz = np.random.uniform(-5, 5, 3)
        rpy = np.random.uniform(-1.57, 1.57, 3)
        uvw = np.random.uniform(-2, 2, 3)
        pqr = np.random.uniform(-1, 1, 3)

        # x, y, z, phi, theta, psi, u, v, w, p, q, r
        state = np.array(
            # [xyz[0], xyz[1], 0.0, 0.0, 0.0, rpy[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            [0.0, 0.0, xyz[2], 0.0, rpy[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        return state

    def step(self, action):
        action_6d = self._make_action_6d(action)

        t, full_state = self.dynamics.step(action_6d)  # 12d
        self.full_state = full_state  # for plotting in tensorboard

        observed_state = self._get_obs()

        # self.reward, reward_info = self._calculate_reward(observed_state, action)
        # self.reward, reward_info = self._calculate_reward_xy(observed_state, action)
        self.reward, reward_info = self._calculate_reward_trim(observed_state, action)

        self.prev_action = action
        self.current_step += 1
        done = self.current_step >= self.ep_length

        info_state = {
            "xyz": full_state[0:3].round(2).tolist(),
            "rpy": full_state[3:6].round(2).tolist(),
            "uvw": full_state[6:9].round(2).tolist(),
            "pqr": full_state[9:12].round(2).tolist(),
        }
        action_6d = action_6d.round(2).tolist()
        info_actions = {
            "rpm": action_6d[1],
            "de": action_6d[2],
            "dr": action_6d[3],
            "lcg": action_6d[4],
            "vbs": action_6d[5],
        }
        info = {"rewards": reward_info, "state": info_state, "actions": info_actions}

        return observed_state, self.reward, done, info

    def set_init_state(self, init_state):
        self.init_state = init_state

    def reset(self):
        self.current_step = 0
        self.num_resets += 1

        # state = self.init_state
        state = self._reset_uniform()
        self.dynamics.set_state(state)
        return self._get_obs()

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _calculate_reward_xy(self, state, action):
        """
        :param state: 12d state vector
            x, y, z,
            phi, theta, psi,
            u, v, w,
            p, q, r
        :param action: 5d action vector
        """
        a_diff = action - self.prev_action

        e_position = np.log(np.linalg.norm(state[0:3]))
        e_orientation = 0.2 * np.linalg.norm(state[3:6])
        e_lin_vel = 0.1 * np.linalg.norm(state[6:9])
        e_ang_vel = 0.1 * np.linalg.norm(state[9:12])
        e_action = 0.1 * np.linalg.norm(action)
        e_action_rate = 0.1 * np.linalg.norm(a_diff)

        dt = 0.01
        e_total = -dt * np.sum(
            [e_position, e_orientation, e_lin_vel, e_ang_vel, e_action, e_action_rate]
        )

        e_info = {
            "e_total": round(e_total, 3),
            "e_position": round(-e_position, 3),
            "e_orientation": round(-e_orientation, 3),
            "e_lin_vel": round(-e_lin_vel, 3),
            "e_ang_vel": round(-e_ang_vel, 3),
            "e_action": round(-e_action, 3),
            "e_action_rate": round(-e_action_rate, 3),
        }

        return e_total, e_info

    def _calculate_reward_xy2(self, state, action):
        Q = np.diag(
            [0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01]
        )
        R = np.diag([0.003, 0.003, 0.003, 0.003, 0.003])  # weights on controls
        R_r = np.diag([0.003, 0.003, 0.003, 0.003, 0.003])  # weights on rates

        a_diff = action - self.prev_action

        e_s = np.linalg.norm(state * Q * state)  # error on state, setpoint is all 0's
        e_a = np.linalg.norm(action * R * action)  # error on actions
        e_r = np.linalg.norm(a_diff * R_r * a_diff)  # error on action rates
        e = e_s + e_a + e_r
        error = np.maximum(0, 1.0 - e_s) - e_a - e_r

        e_info = {
            "e_total": round(error, 3),
            "e_state": round(-e_s, 3),
            "e_action": round(-e_a, 3),
            "e_action_rate": round(-e_r, 3),
        }

        return error, e_info

    def _calculate_reward_xy3(self, state, action):
        """
        :param state: 12d state vector
            x, y, z,
            phi, theta, psi,
            u, v, w,
            p, q, r
        :param action: 5d action vector
        """
        a_diff = action - self.prev_action

        e_position = 1 / (np.linalg.norm(state[0:3]) + 1)
        e_orientation = 1 / (np.linalg.norm(state[0:6]) + 1)
        e_lin_vel = 1 / (np.linalg.norm(state[6:9]) + 1)
        e_ang_vel = 1 / (np.linalg.norm(state[9:12]) + 1)
        e_action = 1 / (np.linalg.norm(action) + 1)
        e_action_rate = 1 / (np.linalg.norm(a_diff) + 1)

        e_total = (
            0.56 * e_position
            + 0.2 * e_orientation
            + 0.02 * e_lin_vel
            + 0.02 * e_ang_vel
            + 0.1 * e_action
            + 0.1 * e_action_rate
        )

        e_info = {
            "e_total": round(e_total, 3),
            "e_position": round(e_position, 3),
            "e_orientation": round(e_orientation, 3),
            "e_lin_vel": round(e_lin_vel, 3),
            "e_ang_vel": round(e_ang_vel, 3),
            "e_action": round(e_action, 3),
            "e_action_rate": round(e_action_rate, 3),
        }

        return e_total, e_info

    def _calculate_reward_trim(self, state, action):
        # x, y, z, phi, theta, psi, u, v, w, p, q, r)
        Q = np.diag(
            # [0.0, 0.1, 0.3, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
        )  # z, pitch, v, q
        R = np.diag([0.03, 0.03])  # weights on controls
        R_r = np.diag([0.3, 0.3])  # weights on rates

        a_diff = action - self.prev_action

        e_s = np.linalg.norm(state * Q * state)  # error on state, setpoint is all 0's
        e_a = np.linalg.norm(action * R * action)  # error on actions
        e_r = np.linalg.norm(a_diff * R_r * a_diff)  # error on action rates
        e = e_s + e_a + e_r
        error = np.maximum(0, 1.0 - e_s) - e_a - e_r

        e_info = {
            "e_total": round(error, 3),
            "e_state": round(-e_s, 3),
            "e_action": round(-e_a, 3),
            "e_action_rate": round(-e_r, 3),
        }

        return error, e_info


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ep_length = 7000
    env = EnvEOMGym(episode_length=ep_length, num_envs=1)
    action = np.array([-1, 0.0, 1.0, 0.0, 0.0])

    states = np.zeros([ep_length, 12])
    for ts in range(ep_length):
        state, reward, done, info = env.step(action)
        states[ts] = state

        print(
            "[{}] {}\n{}\n{}\n".format(
                ts, info["state"], info["actions"], info["rewards"]
            )
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(states[:, 0], states[:, 1], "k-", label="sim")
    ax.plot(states[:1, 0], states[:1, 1], "go", label="start")
    ax.plot(states[-1, 0], states[-1, 1], "ro", label="end")

    # format
    ax.set_xlabel("$x~[m]$")
    ax.set_ylabel("$y~[m]$")
    plt.legend()
    fig.suptitle("test")
    plt.show()
