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

from scipy.integrate import solve_ivp

def T_b2ned(phi, theta, psi):
    T = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)],
    ])
    return T

def R_b2ned(phi, theta, psi):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    R = Rz@Ry@Rx
    return R

def unskew(mat):
    return np.array([
        mat[2,1],
        mat[0,2],
        mat[1,0]
    ])

def skew(vec):
    return np.array([
    [0, -vec[2], vec[1]],
    [vec[2], 0, -vec[0]],
    [-vec[1], vec[0], 0]
    ])

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
        self.dt = 0.2 # time step

        # system tolerances
        self.atol = 1e-8
        self.rtol = 1e-8
        self.solver_method='DOP853'

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
        r_g = np.array([0.1 + lcg*0.01, 0.0, 0.0]) # Including Longitudinal C.G. trim system
        # r_g = np.array([0.1, 0.0, 0.0]) #no effect of Longitudinal C.G. trim system
        r_b = np.array([0.1, 0.0, 0.0])
        r_cp = np.array([0.1, 0.0, 0.0])

        # Buoyancy effects
        W = m*9.81
        B = W + vbs*1.5  #(AUV specific if a Variable Buoyancy System exists)
        # B = W #no effect of Variable buoyancy system

        # hydrodynamic coefficients (AUV specific)
        Xuu = 5.
        Yvv = 20.
        Zww = 50.
        Kpp = 0.1
        Mqq = 20.
        Nrr = 20.

        # Thruster coefficients (AUV specific)
        K_T = np.array([0.0175, 0.0175])
        Q_T = np.array([0.001, -0.001])#*0.0

        # mass and inertia matrix
        M = np.block([
            [m*np.eye(3,3), -m*skew(r_g)],
            [m*skew(r_g), I_o]
        ])
        assert M.shape == (6,6), M

        # coriolis and centripetal matrix
        nu1 = np.array([u, v, w])
        nu2 = np.array([p, q, r])
        top_right = -m*skew(nu1) - m*skew(nu2)*skew(r_g)
        bottom_left = -m*skew(nu1) + m*skew(r_g)*skew(nu2)
        bottom_right = -skew(I_o.dot(nu2))
        C_RB = np.block([
            [np.zeros((3,3)), top_right],
            [bottom_left, bottom_right]
        ])
        assert C_RB.shape == (6, 6), C_RB

        # damping matrix (AUV specific - This layout is for a slender AUV)
        forces = np.diag(np.array([Xuu*np.abs(u), Yvv*np.abs(v), Zww*np.abs(w)]))
        moments = np.diag(np.array([Kpp*np.abs(p), Mqq*np.abs(q), Nrr*np.abs(r)]))
        coupling = np.matmul(skew(r_cp), forces)
        D = np.block([[forces, np.zeros((3, 3))], [-coupling, moments]])
        assert D.shape == (6, 6), D

        # rotational transform between body and NED in Euler
        T_euler = T_b2ned(phi, theta, psi)
        R_euler = R_b2ned(phi, theta, psi)
        assert R_euler.shape == (3,3), R_euler
        J_eta = np.block([
            [R_euler, np.zeros((3,3))],
            [np.zeros((3,3)), T_euler]
        ])
        assert J_eta.shape == (6,6), J_eta

        # buoyancy in quaternions
        f_g = np.array([0, 0, W])
        f_b = np.array([0, 0, -B])
        row1 = np.linalg.inv(R_euler).dot(f_g + f_b)
        row2 = skew(r_g).dot(np.linalg.inv(R_euler)).dot(f_g) + \
            skew(r_b).dot(np.linalg.inv(R_euler)).dot(f_b)
        geta = np.block([row1, row2])
        assert geta.shape == (6,), geta

        # Effect of control actuators (AUV specific)
        F_T = K_T.dot(np.array([rpm1, rpm2]))
        M_T = Q_T.dot(np.array([rpm1, rpm2]))
        tauc = np.array([
            F_T*np.cos(de)*np.cos(dr),
            -F_T*np.sin(dr),
            F_T*np.sin(de)*np.cos(dr),
            M_T*np.cos(de)*np.cos(dr),
            -M_T*np.sin(dr),
            M_T*np.sin(de)*np.cos(dr)
        ])
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
                #jac=lambda t, x: eom_jac(x, controller(x))
            )

        times, states = sol.t, sol.y.T
        return times, states

    def step(self, action):
        # propagate system
        next_ts = self.timestep + self.dt
        t_out, x_out = self.propagate(self.current_x, action, self.timestep, next_ts, self.atol, self.rtol, self.solver_method)

        self.timestep = t_out[-1]
        self.current_x = x_out[-1,:]
        return self.timestep, self.current_x

    def get_state(self):
        return self.current_x

    def set_state(self, state):
        self.current_x = state


class EnvEOM_Task_Trim(object):
    """
    Define common API to env and cost function for the task
    Learns to control trim (depth and pitch angle)
    """
    def __init__(self):
        # self.init_state = np.array([0., 0.2, 0.18, 0., 0., 0.]) # x, z, theta, u, w, q
        self.init_state = np.array([0., 0., 0.17, 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # x, y, z, phi, theta, psi, u, v, w, p, q, r
        self.prev_action = []
        self.env = EnvEOM(self.init_state)

        self.state_dim = 6 # x, z, theta, u, w, q
        self.action_dim = 2  # lcg, vbs
        self.max_action = np.array([1, 1])  # max LCG = 1, max VBS = 1

        self.setpoint_6d = np.array([0., 5.0, 0.3, 0., 0., 0.])    # (x, z, theta, u, w, q)

    def step(self, action : np.array):
        """
        @param action : np.array (6 x 1) : (rpm1, rpm2, de, dr, lcg, vbs)
                        every action is in range (-1, 1) and scaled appropriately in the EnvEOM.eom
        """
        # propagate system
        t, state = self.env.step(action)
        self.prev_action = action

        state_6d = self.get_observation() # temporary 6d for full trim test
        reward = self._calculate_reward(state_6d, self.setpoint_6d, action)
        # done = self._is_done(state_6d, self.setpoint_6d)
        done = False
        return state_6d, reward, done

    def get_observation(self):
        """
        @return state (6 x 1) : (x, z, theta, u, w, q)
        """
        state = self.env.get_state() # get full state 12d
        # 12d - x, y, z, phi, theta, psi, u, v, w, p, q, r
        x = state[0]
        z = state[2]
        theta = state[4]
        u = state[6]
        w = state[8]
        q = state[10]
        state_6d = np.array([x, z, theta, u, w, q])
        return state_6d

    def reset(self):
        self.env.set_state(self.init_state)
        return self.get_observation()

    def _calculate_reward(self, state, target, action):
        """
        @param state (6 x 1) (x, z, theta, u, w, q)
        @param target (6 x 1)
        """
        Q = np.diag([0., 10., 1000., 0., 1000., 0.])  # weights on states - z, theta, w
        R = np.diag([0., 0., 0., 0., 10., 10.]) # weights on controls - lcg, vbs

        s_diff = state - target
        a_diff = action - self.prev_action

        e_s = np.linalg.norm(s_diff * Q * s_diff) # error on state
        e_a = np.linalg.norm(action * R * action) # error on actions
        e_r = np.linalg.norm(a_diff * R * a_diff) # error on action rates
        return -(e_s + e_a + e_r)

    def _is_done(self, state, target):
        """
        @param state (6 x 1)
        @param target (6 x 1)
        """
        eps = 1e-2
        return np.isclose(state, target, rtol=eps)


class EnvEOM_Task_XY(object):
    """
    Define common API to env and cost function for the task
    Learns to control RPM and DR to go to point in XY plane
    """
    def __init__(self):
        # x, y, z, phi, theta, psi, u, v, w, p, q, r
        self.init_state = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.prev_action = []
        self.env = EnvEOM(self.init_state)

        self.state_dim = 6 # x, y, psi, u, v, r
        self.action_dim = 2  # rpm (both), dr
        self.max_action = np.array([1, 1])  # max rpm = 1, max dr = 1

        self.setpoint_6d = np.array([50., 0.0, 0.0, 0., 0., 0.])    # x, y, psi, u, v, r

    def step(self, action : np.array):
        """
        @param action : np.array (6 x 1) : (rpm1, rpm2, de, dr, lcg, vbs)
                        every action is in range (-1, 1) and scaled appropriately in the EnvEOM.eom
        """
        # propagate system
        t, state = self.env.step(action)
        self.prev_action = action

        state_6d = self.get_observation() # temporary 6d for full trim test
        reward = self._calculate_reward(state_6d, self.setpoint_6d, action)
        # done = self._is_done(state_6d, self.setpoint_6d)
        done = False
        return state_6d, reward, done

    def get_observation(self):
        """
        @return state (6 x 1) : (x, z, theta, u, w, q)
        """
        state = self.env.get_state() # get full state 12d
        # 12d - x, y, z, phi, theta, psi, u, v, w, p, q, r
        x = state[0]
        y = state[1]
        psi = state[5]
        u = state[6]
        v = state[7]
        r = state[11]
        state_6d = np.array([x, y, psi, u, v, r])
        return state_6d

    def reset(self):
        self.env.set_state(self.init_state)
        return self.get_observation()

    def _calculate_reward(self, state, target, action):
        """
        @param state (6 x 1) (x, y, psi, u, v, r)
        @param target (6 x 1)
        """
        Q = np.diag([100., 100., 100., 100., 100., 100.])  # weights on states
        R = np.diag([10., 10., 0., 10., 0., 0.]) # weights on controls - rpm, dr

        s_diff = state - target
        a_diff = action - self.prev_action

        e_s = np.linalg.norm(s_diff * Q * s_diff) # error on state
        e_a = np.linalg.norm(action * R * action) # error on actions
        e_r = np.linalg.norm(a_diff * R * a_diff) # error on action rates
        return -(e_s + e_a + e_r)

    def _is_done(self, state, target):
        """
        @param state (6 x 1)
        @param target (6 x 1)
        """
        eps = 1e-2
        return np.isclose(state, target, rtol=eps)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.set_printoptions(precision=2, suppress=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    init_state = np.array([0., 0., 0.17, 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # x, y, z, phi, theta, psi, u, v, w, p, q, r
    env = EnvEOM(init_state)

    action_6d = np.array([0.1, 0.1, 0., 1, 0., 0.]) #rpm1, rpm2, de, dr, lcg, vbs

    error = 0   # max 31.5k for 400 timesteps (80s sim)
    ts = 1000
    traj = np.zeros([ts, 12])
    for i in range(ts):
        t, x = env.step(action_6d)
        error += -np.power(x[2] - 5., 2)
        print(f't = {t.round(2)}   x = {x.round(3)}   error = {error:.2f}')
        traj[i] = x # save for plots

    ax.plot(traj[:,0], traj[:,1], traj[:,2], 'k-', label='sim')
    ax.plot(traj[:1,0], traj[:1,1], traj[:1,2], 'ko', label=None)

    # format
    ax.set_xlabel('$x~[m]$')
    ax.set_ylabel('$y~[m]$')
    ax.set_zlabel('$z~[m]$')
    plt.legend()
    plt.show()
