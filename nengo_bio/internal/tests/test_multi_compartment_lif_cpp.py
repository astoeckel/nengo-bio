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

from nengo_bio.internal.multi_compartment_lif_parameters import (
    DendriticParameters,
    SomaticParameters
)
from nengo_bio.internal.multi_compartment_lif_cpp import compile_simulator_cpp

def test_compile_simulator_cpp():
    # Simulation parameters
    n_neurons = 1
    T = 0.1
    dt = 1e-4
    ts = np.arange(0, T, dt)

    # Generate some somatic and dendritic parameters
    params_som = SomaticParameters()
    params_den = DendriticParameters.make_two_comp_lif(input_mul=1e-9)

    # Compile a simulator for these parameters
    sim_class_ss1 = compile_simulator_cpp(params_som, params_den, dt=dt, ss=1)
    sim_class_ss10 = compile_simulator_cpp(params_som, params_den, dt=dt, ss=10)

    # Run the simulation
    N = ts.size
    trace_v_ss1 = np.empty((N, n_neurons, params_den.n_comp))
    trace_v_ss10 = np.empty((N, n_neurons, params_den.n_comp))
    sim_ss1 = sim_class_ss1(n_neurons)
    sim_ss10 = sim_class_ss10(n_neurons)
    for i in range(N):
        sim_ss1.step_math(np.tile(((50, 0),), (n_neurons, 1)))
        sim_ss10.step_math(np.tile(((50, 0),), (n_neurons, 1)))
        trace_v_ss1[i] = sim_ss1.state[:, :-1]
        trace_v_ss10[i] = sim_ss10.state[:, :-1]

    assert np.sqrt(np.mean(np.square(trace_v_ss1 - trace_v_ss10)[:, :, 1])) < 1e-3
