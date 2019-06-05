#   nengo_bio -- Extensions to Nengo for more biological plausibility
#   Copyright (C) 2019  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def compile_simulator_python(params_som, params_den, dt=1e-3, ss=10):
    # Some handy aliases
    pS, pD = params_som, params_den
    dt_, ss_ = dt, ss

    class Simulator:
        n_comp = params_den.n_comp
        n_inputs = params_den.n_inputs
        dt = dt_
        ss = ss_

        def __init__(self, n_neurons):
            # Copy the number of neurons
            self.n_neurons = n_neurons

            # Initialize the state matrix
            self.state = np.empty((n_neurons, self.n_comp + 1),
                                  order='C',
                                  dtype=np.float64)
            self.state[:, :-1] = params_som.v_reset
            self.state[:, -1] = 0

            # Initialize the output vector
            self.out = np.zeros((n_neurons), order='C', dtype=np.float64)

        def step_math(self, xs):
            # Make sure the input has the correct size and datatype
            xs = np.asarray(xs, order='C', dtype=np.float64)
            assert xs.size >= self.n_inputs * self.n_neurons

            # Iterate over all neurons
            for i in range(self.n_neurons):
                # Fetch the input data
                x = xs[i]

                # Compute the A-matrix and the b-vector for this sample
                A = pD.C + np.diag(pD.a_const + pD.A @ x)
                b = pD.b_const + pD.B @ x

                # Access to the current state: membrane poential and
                # refractoriness
                S = self.state[i]

                # Write the initial output value
                self.out[i] = 0.

                # Advance the simulation for the given number of subsamples
                for s in range(ss):
                    # Compute the membrane potentials in dt / ss
                    S[:-1] += (A @ S[:-1] + b) * (dt / ss)

                    # Handle refractoriness
                    if S[-1] > 0.0:
                        S[-1] -= dt / ss
                        S[0] = pS.v_spike if S[-1] > pS.tau_ref else pS.v_reset

                    # Handle spikes
                    if S[0] > pS.v_th and S[-1] <= 0.0:
                        S[-1] = pS.tau_ref + pS.tau_spike
                        S[0] = pS.v_spike if pS.tau_spike > 0 else pS.v_reset
                        self.out[i] = 1. / dt

            # Return the otuput
            return self.out, self.state

    return Simulator

