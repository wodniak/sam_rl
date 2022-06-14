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

from env.utils import normalize_angle_rad

# from utils import normalize_angle_rad


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

    def __init__(self, dt, init_state):
        self.timestep = 0.0
        self.dt = dt  # time step

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

    def __init__(
        self,
        episode_length,
        dt,
        env_obs_states,
        env_obs_state_reset,
        env_actions,
        env_reward_fn_type,
        weights_Q,
        weights_R,
        weights_R_r,
        num_envs=1,
    ):
        super(EnvEOMGym, self).__init__()

        self.env_obs_states = env_obs_states  # defined in params
        self.env_obs_state_reset = env_obs_state_reset
        self.env_actions = env_actions
        self.env_reward_fn_type = env_reward_fn_type
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

        if env_reward_fn_type == "trim":
            self.reward_fn = RewardFnTrim(
                env_obs_states, env_actions, weights_Q, weights_R, weights_R_r
            )
        elif env_reward_fn_type == "xy":
            self.reward_fn = RewardFnXY(
                env_obs_states, env_actions, weights_Q, weights_R, weights_R_r
            )
        elif env_reward_fn_type == "pendulum":
            self.reward_fn = RewardFnInvertedPendulum(
                env_obs_states, env_actions, weights_Q, weights_R, weights_R_r
            )
        elif env_reward_fn_type == "tight_turn":
            self.reward_fn = RewardFnTightTurn(
                env_obs_states, env_actions, weights_Q, weights_R, weights_R_r
            )
        else:
            self.reward_fn = None

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
        self.dynamics = EnvEOM(dt, self.init_state)

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
                value = state_12d[state_pos]

                # normalize angle to [-pi, pi]
                if key in ["phi", "theta", "psi"]:
                    value = normalize_angle_rad(value)

                obs.append(value)
        return np.array(obs)

    def _reset_uniform(self):
        # x, y, z, phi, theta, psi, u, v, w, p, q, r
        state = np.zeros(12)
        for key in self.env_obs_state_reset.keys():
            value = self.env_obs_state_reset[key]
            pos = self.key_to_state_map[key]
            state[pos] = np.random.uniform(value["min"], value["max"], 1)
        return state

    def _is_done(self, state):
        # return self.reward_fn.is_done(state) or self.current_step >= self.ep_length
        return self.current_step >= self.ep_length

    def step(self, action):
        action_6d = self._make_action_6d(action)
        # action_6d = np.array([action[0], action[0], 0.0, action[2], 0.0, 0.0])
        t, full_state = self.dynamics.step(action_6d)  # 12d
        full_state[3] = normalize_angle_rad(full_state[3])
        full_state[4] = normalize_angle_rad(full_state[4])
        full_state[5] = normalize_angle_rad(full_state[5])
        self.full_state = full_state  # for plotting in tensorboard

        observed_state = self._get_obs()
        self.reward, reward_info = self.reward_fn.calculate_reward(
            observed_state, action
        )

        self.prev_action = action
        self.current_step += 1
        done = self._is_done(observed_state)

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
        # print(info)
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


class RewardFnBase(object):
    """Define reward function weight matrices"""

    def __init__(self, env_obs_states, env_actions, weights_Q, weights_R, weights_R_r):
        # variables
        self.state_dim = len(env_obs_states)
        self.action_dim = len(env_actions)

        self.Q = np.zeros(self.state_dim)
        self.R = np.zeros(self.action_dim)
        self.R_r = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)

        # Fill Q matrix
        for i, key in enumerate(env_obs_states):
            if key in weights_Q:
                value = weights_Q[key]
                self.Q[i] = value

        # Fill R and R_r matrices
        for i, key in enumerate(env_actions):
            if key in weights_R:
                value_R = weights_R[key]
                value_R_r = weights_R_r[key]
                self.R[i] = value_R
                self.R_r[i] = value_R_r

        print(
            f"Reward function:\n\
            Q:{self.Q}\n\
            R:{self.R}\n\
            R_r:{self.R_r}\n"
        )

    def calculate_reward(self, state, action):
        raise NotImplementedError()


class RewardFnTrim(RewardFnBase):
    def __init__(self, env_obs_states, env_actions, weights_Q, weights_R, weights_R_r):
        super().__init__(env_obs_states, env_actions, weights_Q, weights_R, weights_R_r)
        # sanity checks
        assert "z" in env_obs_states, "Missing env states for TRIM cost function"
        assert "theta" in env_obs_states, "Missing env states for TRIM cost function"
        assert "w" in env_obs_states, "Missing env states for TRIM cost function"
        assert "q" in env_obs_states, "Missing env states for TRIM cost function"

    def calculate_reward(self, state, action):
        """Reward function for Trim"""
        assert len(state) == self.state_dim
        assert len(action) == self.action_dim

        a_diff = action - self.prev_action
        e_s = np.linalg.norm(
            state * self.Q * state
        )  # error on state, setpoint is all 0's
        e_a = np.linalg.norm(action * self.R * action)  # error on actions
        e_r = np.linalg.norm(a_diff * self.R_r * a_diff)  # error on action rates
        e = e_s + e_a + e_r
        error = np.maximum(0, 1.0 - e_s) - e_a - e_r

        e_info = {
            "e_total": round(error, 3),
            "e_state": round(-e_s, 3),
            "e_action": round(-e_a, 3),
            "e_action_rate": round(-e_r, 3),
        }

        return error, e_info


class RewardFnXY(RewardFnBase):
    """Reward function for XY"""

    def __init__(self, env_obs_states, env_actions, weights_Q, weights_R, weights_R_r):
        super().__init__(env_obs_states, env_actions, weights_Q, weights_R, weights_R_r)
        # sanity checks
        assert "x" in env_obs_states, "Missing env states for XY cost function"
        assert "y" in env_obs_states, "Missing env states for XY cost function"
        assert "psi" in env_obs_states, "Missing env states for XY cost function"
        assert "r" in env_obs_states, "Missing env states for XY cost function"

    def calculate_reward(self, state, action):
        assert len(state) == self.state_dim
        assert len(action) == self.action_dim

        a_diff = action - self.prev_action

        if self.state_dim == 12:
            e_position = np.log(np.linalg.norm(state[0:3]))
            e_orientation = 0.2 * np.linalg.norm(state[3:6])
            e_lin_vel = 0.1 * np.linalg.norm(state[6:9])
            e_ang_vel = 0.1 * np.linalg.norm(state[9:12])
            e_action = 0.1 * np.linalg.norm(action)
            e_action_rate = 0.1 * np.linalg.norm(a_diff)
        # assumed states: x,y,psi,u,v,r
        elif self.state_dim == 6:
            e_position = np.linalg.norm(state[0:2])
            e_orientation = 0.2 * np.linalg.norm(state[2:3])
            e_lin_vel = 0.1 * np.linalg.norm(state[3:5])
            e_ang_vel = 0.1 * np.linalg.norm(state[5:6])
            e_action = 0.1 * np.linalg.norm(action)
            e_action_rate = 0.1 * np.linalg.norm(a_diff)
        else:
            raise NotImplementedError("Other cases are not implemented.")

        dt = 0.2
        e_total = -dt * np.sum(
            [e_position, e_orientation, e_lin_vel, e_ang_vel, e_action, e_action_rate]
        )
        # e_total = -dt * e_position

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


class RewardFnInvertedPendulum(RewardFnBase):
    def __init__(self, env_obs_states, env_actions, weights_Q, weights_R, weights_R_r):
        super().__init__(env_obs_states, env_actions, weights_Q, weights_R, weights_R_r)
        # sanity checks
        name = "Inverted Pendulum"
        assert "theta" in env_obs_states, f"Missing env states for {name} cost function"

        self.target_theta = -1.5708
        # x, y, z, phi, theta, psi, u, v, w, p, q, r
        self.target_state = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                self.target_theta,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        # rpm, de, dr, lcg, vbs
        # self.target_actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.target_actions = np.array([0.0, 0.0, 0.0])

    def calculate_reward(self, state, action):
        """Reward function for Inverted pendulum"""
        assert len(state) == self.state_dim
        assert len(action) == self.action_dim

        a_diff = action - self.prev_action

        pos = state[0:3]
        theta = state[4]
        theta_dt = state[7]
        theta_acc = state[10]

        e_s = np.minimum(np.linalg.norm(pos**2 * self.Q[0:3]), 10.0)
        e_theta = (
            (theta - self.target_theta) ** 2
            + 0.1 * theta_dt**2
            + 0.001 * theta_acc**2
        )
        e_a = np.linalg.norm(
            (action - self.target_actions) ** 2 * self.R
        )  # error on actions
        e_r = np.linalg.norm(a_diff**2 * self.R_r)  # error on action rates

        e = e_s + e_theta + e_a + e_r
        error = -e

        e_info = {
            "e_total": round(error, 3),
            "e_state": round(-e_s, 3),
            "e_theta": round(-e_theta, 3),
            "e_action": round(-e_a, 3),
            "e_action_rate": round(-e_r, 3),
        }

        return error, e_info

    def is_done(self, state):
        """
        :return True if SAM loses balance
        """
        theta = state[4]
        eps = 0.3
        return theta < (self.target_theta - eps) or theta > (self.target_theta + eps)


class RewardFnTightTurn(RewardFnBase):
    def __init__(self, env_obs_states, env_actions, weights_Q, weights_R, weights_R_r):
        super().__init__(env_obs_states, env_actions, weights_Q, weights_R, weights_R_r)
        # sanity checks
        name = "Tight Turn"
        assert "x" in env_obs_states, f"Missing env states for {name} cost function"
        assert "y" in env_obs_states, f"Missing env states for {name} cost function"
        assert "psi" in env_obs_states, f"Missing env states for {name} cost function"
        assert "u" in env_obs_states, f"Missing env states for {name} cost function"
        assert "v" in env_obs_states, f"Missing env states for {name} cost function"
        assert "r" in env_obs_states, f"Missing env states for {name} cost function"

        self.psi_dt_target = 0.5  # manual max
        # rpm, de, dr, lcg, vbs

    def calculate_reward(self, state, action):
        """Reward function for tight turn"""
        assert len(state) == self.state_dim
        assert len(action) == self.action_dim

        # Cost on position
        r = 3
        if self.state_dim == 6:
            pos = state[0:2]
            e_s = np.minimum(np.abs(np.linalg.norm(pos) - r), 7)  # stay on the circle
        else:
            pos = state[0:3]
            e_s = 0.5 * np.minimum(
                np.linalg.norm(pos * self.Q[0:3]), 3.14
            )  # max cost as for heading

        # Cost on velocity
        # min = 0, max = 5 (when not moving)
        psi_dt = state[5] if self.state_dim == 6 else state[11]
        psi_dt = np.abs(psi_dt)
        e_psi_dt = np.linalg.norm((psi_dt - self.psi_dt_target)) * 20

        # # Cost on actions
        # a_diff = action - self.prev_action
        # e_a = np.linalg.norm(action**2 * self.R)  # error on actions
        # e_r = np.linalg.norm(a_diff**2 * self.R_r)  # error on action rates

        # Cost on heading
        # Reward on keeping the heading toward the origin, min = 0, max = 3.14
        # NOTE: Vectors are in SAM body frame, i.e, when SAM heads toward origin angle = 0 rad.
        psi = state[2] if self.state_dim == 6 else state[5]
        sam_to_origin_v = state[0:2]  # x,y
        sam_to_origin_v_unit = sam_to_origin_v / np.linalg.norm(sam_to_origin_v)
        sam_heading_v = np.array([-np.cos(psi), -np.sin(psi)])
        sam_heading_v_unit = sam_heading_v / np.linalg.norm(sam_heading_v)
        e_heading = np.arccos(
            np.clip(np.dot(sam_to_origin_v_unit, sam_heading_v_unit), -1.0, 1.0)
        )  # https://stackoverflow.com/a/13849249
        # print(f"np.arctan2: {np.arctan2(state[1],state[0])}")

        # total cost
        # error = -(e_s + e_psi_dt + e_a + e_r + e_heading)
        error = -(e_s + e_psi_dt + e_heading)

        e_info = {
            "e_total": round(error, 3),
            "e_state": round(-e_s, 3),
            "e_psi_dt": round(-e_psi_dt, 3),
            # "e_action": round(-e_a, 3),
            # "e_action_rate": round(-e_r, 3),
            "e_heading": round(-e_heading, 3),
        }

        return error, e_info

    def is_done(self, state):
        """ """
        theta = state[4]
        eps = 0.3
        return theta < (self.target_theta - eps) or theta > (self.target_theta + eps)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import yaml

    config_path = "/home/gwozniak/catkin_ws/src/smarc_rl_controllers/sam_rl/src/config/tight_turn_6d.yaml"
    with open(config_path) as file:
        print(f"Loading config file : {config_path}")
        params = yaml.load(file, Loader=yaml.FullLoader)

    env = EnvEOMGym(
        episode_length=params["episode_length"],
        dt=params["env_dt"],
        env_obs_states=params["env_state"],
        env_obs_state_reset=params["env_state_reset"],
        env_actions=params["env_actions"],
        env_reward_fn_type=params["env_reward_fn_type"],
        weights_Q=params["env_state_weights_Q"],
        weights_R=params["env_actions_weights_R"],
        weights_R_r=params["env_actions_weights_R_r"],
        num_envs=params["num_cpu"],
    )
    action = np.array([-1, 1.0])

    test_length = 1
    states = np.zeros([test_length, 12])
    for ts in range(test_length):
        state, reward, done, info = env.step(action)
        states[ts] = env.full_state

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
    ax.plot(0.0, 0.0, "ko", label="origin")

    # arrows
    r = 0.2
    x_0 = states[-1, 0]
    y_0 = states[-1, 1]
    psi_0 = states[-1, 5]  # psi
    plt.arrow(
        x_0,
        y_0,
        r * np.cos(psi_0),
        r * np.sin(psi_0),
        color="blue",
        head_length=0.1,
        head_width=0.1,
    )
    plt.arrow(
        x_0,
        y_0,
        -x_0,
        -y_0,
        color="blue",
        head_length=0.1,
        head_width=0.1,
    )  # In SAM frame

    # format
    ax.set_xlabel("$x~[m]$")
    ax.set_ylabel("$y~[m]$")
    plt.legend()
    fig.suptitle("test")
    plt.show()
