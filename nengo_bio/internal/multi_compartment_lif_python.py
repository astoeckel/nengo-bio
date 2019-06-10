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

    # Compute v_min, v_max
    v_min, v_max = params_den.vEq_extreme(params_som)

    class PythonImpl:
        @staticmethod
        def run_step_from_memory(self, out, *xs):
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

                    # Clamp the potentials
                    S[:-1] = np.clip(S[:-1], v_min, v_max)

                    # Handle refractoriness
                    if S[-1] > 0.0:
                        S[-1] -= dt / ss
                        S[0] = pS.v_spike if S[-1] > pS.tau_ref else pS.v_reset

                    # Handle spikes
                    if S[0] > pS.v_th and S[-1] <= 0.0:
                        S[-1] = pS.tau_ref + pS.tau_spike
                        S[0] = pS.v_spike if pS.tau_spike > 0 else pS.v_reset
                        out[i] = 1. / dt

        @staticmethod
        def run_single_with_constant_input(self, out, xs):
            for i in range(out.shape[0]):
                PythonImpl.run_step_from_memory(self, out[i:i + 1], *xs)

        @staticmethod
        def run_single_with_poisson_sources(self, out, sources):
            n_samples = out.size
            n_inputs = len(sources)

            # Initialize the individual random engines for the input channels,
            # pre-compute some filter constants
            rngs = [None] * n_inputs
            dist_exp, dist_gain = [None] * n_inputs, [None] * n_inputs
            filt, xs, offs, T = np.zeros((4, n_inputs, 1))
            for j, src in enumerate(sources):
                # Initialize the random engine for this input with the seed
                # specified by the user
                rngs[j] = np.random.RandomState(src.seed)

                # Compute the filter coefficient
                filt[j] = 1.0 - self.dt / src.tau

                # Setup the poisson and uniform distribution
                def mk_dists(j, src):
                    scale = 1.0 / (src.tau * src.rate)
                    dist_exp = lambda: rngs[j].exponential(1. / src.rate)
                    dist_gain = lambda: rngs[j].uniform(src.gain_min * scale, src.gain_max * scale)
                    return dist_exp, dist_gain

                dist_exp[j], dist_gain[j] = mk_dists(j, src)

                # Draw the first spike time
                T[j] = dist_exp[j]()

                # Setup the uniform gain distribution and initialize xs to the
                # average value
                xs[j] = 0.5 * (src.gain_min + src.gain_max)

                # Copy the offset
                offs[j] = src.offs

            # Implement the Poisson source and run the simulation
            for i in range(n_samples):
                # Apply the exponential filter
                xs *= filt

                # Simulate the Poisson spike source
                curT = i * self.dt
                for j in range(n_inputs):
                    while T[j] < curT:
                        # Feed a Delta pulse into the input
                        xs[j] += dist_gain[j]()

                        # Compute the next spike time
                        T[j] += dist_exp[j]()

                # Advance the simulation by one step
                PythonImpl.run_step_from_memory(self, out[i:i + 1],
                                                *(xs + offs))

        @staticmethod
        def run_single_with_gaussian_sources(self, out, sources):
            n_samples = out.size
            n_inputs = len(sources)

            # Initialize the individual random engines for the input channels,
            # pre-compute some filter constants
            rngs = [None] * n_inputs
            dist_norm = [None] * n_inputs
            filt, xs, offs = np.zeros((3, n_inputs, 1))
            for j, src in enumerate(sources):
                # Initialize the random engine for this input with the seed
                # specified by the user
                rngs[j] = np.random.RandomState(src.seed)

                # Compute the filter coefficient
                filt[j] = 1.0 - self.dt / src.tau

                # Setup the Gaussian distribution
                def mk_dists(j, src):
                    scale = self.dt / src.tau
                    dist_norm = lambda: rngs[j].normal(scale * src.mu, scale * src.sigma)
                    return dist_norm

                dist_norm[j] = mk_dists(j, src)

                # Setup the uniform gain distribution and initialize xs to the
                # average value
                xs[j] = src.mu

                # Copy the offset
                offs[j] = src.offs

            # Implement the Gaussian source and run the simulation
            for i in range(n_samples):
                for j in range(n_inputs):
                    xs[j] = xs[j] * filt[j] + dist_norm[j]()
                inp = np.maximum(0.0, xs + offs)
                PythonImpl.run_step_from_memory(self, out[i:i + 1], *inp)

    return make_simulator_class(PythonImpl, params_som, params_den, dt, ss)

