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
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

class EnvEOM(object):
    """
    Simplified numerical simulation of SAM for trim control
    State : (x, z, theta, u, w, q)
        @note : Z axis point UP, invert 'z' for being consistend with NED frame
    Action : (vbs, lcg)
        @note : vbs and lcg are in range (-1, 1)
    """
    def __init__(self, init_state):
        self.timestep = 0.0
        self.dt = 0.2 # time step

        # system tolerances
        self.atol = 1e-8
        self.rtol = 1e-8
        self.solver_method='DOP853'

        self.current_x = init_state


    def eom(self, state, control):
        """
        Nonlinear dynamics model function
        """
        # extract states and controls
        x, z, theta, u, w, q = state
        vbs, lcg = control

        #scale controls from -1 to 1 to the correct ranges
        vbs_scale = 1.
        lcg_scale = 1.

        vbs = vbs * vbs_scale
        lcg = lcg * lcg_scale

        eta= np.array([[x], [z], [theta]])
        nu= np.array([[u], [w], [q]])

        # assign parameters
        m = 15.4 # mass
        Iyy = 1.6202

        #cg position
        x_g = 0.0 + lcg*0.01
        z_g = 0.

        #cb position
        x_b = 0.0
        z_b = 0.

        #center of pressure position
        x_cp = 0.1
        z_cp = 0.

        W = m*9.81
        B = W + vbs*1.5 #include VBS effect

        #Hydrodynamic coefficients
        Xuu = 3 #0.8 #1.
        Zww = 50.0 #100.
        Mqq = 40.0 #100.

        # Control actuators
        KT = 0.0175

        # Mass and inertia matrix
        M = np.array([[m, 0., m*z_g],
                    [0, m, -m*x_g],
                    [m*z_g, -m*x_g, Iyy]])
        assert M.shape == (3,3), M

        # Coriolis and centripetal matrix
        C_RB = np.array([[0., 0., -m*(x_g*q-w)],
                        [0., 0., -m*(z_g*q+u)],
                        [m*(x_g*q-w), m*(z_g*q+u), 0.]])
        assert C_RB.shape == (3,3), C_RB

        #Damping matrix
        D = np.array([[Xuu*abs(u), 0., 0.],
                    [0, Zww*abs(w), 0],
                    [-z_cp*Xuu*abs(u), x_cp*Zww*abs(w), Mqq*abs(q)]])

        assert D.shape == (3,3), D


        #rotational transform for kinematics
        J_eta = np.array([[np.cos(theta), np.sin(theta), 0.],
                        [-np.sin(theta), np.cos(theta), 0.],
                        [0., 0.,  1.]])
        assert J_eta.shape == (3,3), J_eta

        # buoyancy in quaternions
        f_g = W
        f_b = -B
        geta = np.array([[(W-B)*np.sin(theta)],
                        [-(W-B)*np.cos(theta)],
                        [(z_g*W-z_b*B)*np.sin(theta)+(x_g*W-x_b*B)*np.cos(theta)]])
        assert geta.shape == (3,1), geta

        # controls
        #F_T= KT*rpm
        #r_LCG = np.array([lcg*0.01, 0, 0])
        #M_LCG = skew(r_LCG).dot(np.linalg.inv(R_q)).dot(f_g)


        tauc = np.block([ [0.],
                        [0.],
                        #[W*np.cos(theta)*lcg*0.01]]) #including LCG as a moment
                        [0.] ])

        #tau_LCG = -np.block([[np.zeros([3,1])],
        #                    [M_LCG]])

        #tauc = tauc + tau_LCG

        assert tauc.shape == (3,1), tauc
        # Kinematics
        etadot = np.block([J_eta.dot(nu)])

        assert etadot.shape == (3,1), etadot

        # Dynamics
        invM = np.linalg.inv(M)
        crbd = C_RB+D
        other = crbd.dot(nu)
        other2 = tauc-other-geta
        nudot = invM.dot(other2)

        assert nudot.shape == (3,1), nudot

        sdot= np.block([ [etadot],
                        [nudot] ])

        return sdot.flatten()

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
    """
    def __init__(self):
        self.init_state = np.array([0., 0.2, 0.18, 0., 0., 0.]) # x, z, theta, u, w, q
        self.env = EnvEOM(self.init_state)

        self.current_setpoint = np.array([0.0, 5.0, 0.15, 0, 0, 0])

        self.state_dim = 2
        self.action_dim = 1  # VBS
        self.max_action = np.array([1])  # max LCG = 1, max VBS = 1

    def step(self, action):
        # propagate system
        action[0] = action[0] * 2 - 1 # normalize (0,1) to (-1,1) for VBS
        action = np.array([action[0], 0.1])
        t, state = self.env.step(action)

        state_2d = np.array([-state[1], state[4]]) # z, w
        reward = self._calculate_reward(state_2d, self.current_setpoint)
        # done = self._is_done(state, self.current_setpoint)
        done = False
        return state_2d, reward, done

    def get_observation(self):
        """Return state"""
        state = self.env.get_state()
        state_2d = np.array([-state[1], state[4]]) # z, w
        return state_2d

    def reset(self):
        self.env.set_state(self.init_state)
        return self.get_observation()

    def _calculate_reward(self, state, target):
        error = np.power(state[0] - target[0], 2)
        return -error

    def _is_done(self, state, target):
        eps = 1e-3
        return np.isclose(state[1], target[1], rtol=eps)



if __name__ == '__main__':
    init_state = np.array([0., -10., 0.1, 0., 0., 0.]) # x, z, theta, u, w, q
    env = EnvEOM(init_state)

    action = np.array([-1, 0.0]) # vbs, lcg

    for i in range(200):
        t, x = env.step(action)

        print(f't = {t.round(2)}   x = {x.round(3)}')
