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


def make_simulator_class(step_math, params_som_, params_den_, dt_, ss_):
    """
    The make_simulator_class creates a new Simulator class as used by both the
    Python and C++ simulator backend. It injects the given step_math function
    into a wrapper that performs additional type checking.

    Parameters
    ==========

    step_math:
        The actual step_math function provided by the simulator backend.
    params_som_:
        The somatic parameters that should be stored in the Simulator class.
    params_den_:
        The dendritic parameters that should be stored in the Simulator class.
    dt_:
        The simulation timestep.
    ss_:
        The number of subsamples per timestep used by the simulator.
    """

    class Simulator:
        params_som = params_som_
        params_den = params_den_
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
            self.state[:, :-1] = params_som_.v_reset
            self.state[:, -1] = 0

        def step_math(self, out, *xs):

            def check(a, size):
                return (a.flags.c_contiguous and (a.dtype == np.float64)
                        and a.size >= size)

            # Make sure the output array has the correct layout and length
            assert check(out, self.n_neurons)

            # Make sure the input arrays have the correct layout and length
            assert len(xs) == self.n_inputs
            for x in xs:
                check(x, self.n_neurons)

            # Call the actual step_math function
            step_math(self, out, *xs)

    return Simulator
