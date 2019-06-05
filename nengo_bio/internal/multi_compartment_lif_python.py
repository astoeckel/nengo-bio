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

from nengo_bio.internal.multi_compartment_lif_sim import make_simulator_class

def compile_simulator_python(params_som, params_den, dt=1e-3, ss=10):
    # Some handy aliases
    pS, pD = params_som, params_den

    def step_math(self, out, *xs):
        # Iterate over all neurons
        for i in range(self.n_neurons):
            # Fetch the input data
            x = np.array([xs[j][i] for j in range(self.n_inputs)])

            # Compute the A-matrix and the b-vector for this sample
            A = pD.C + np.diag(pD.a_const + pD.A @ x)
            b = pD.b_const + pD.B @ x

            # Access to the current state: membrane poential and
            # refractoriness
            S = self.state[i]

            # Write the initial output value
            out[i] = 0.

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
                    out[i] = 1. / dt

    return make_simulator_class(step_math, params_som, params_den, dt, ss)

