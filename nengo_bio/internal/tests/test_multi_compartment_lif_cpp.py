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

def test_compile_simulator_cpp_subsample():
    # Simulation parameters
    n_neurons = 1
    T = 1.0
    dt = 1e-4
    ts = np.arange(0, T, dt)

    # Generate some somatic and dendritic parameters
    params_som = SomaticParameters()
    params_den = DendriticParameters.make_two_comp_lif()

    # Compile a simulator for these parameters
    sim_class_ss1 = compile_simulator_cpp(params_som, params_den, dt=dt, ss=1)
    sim_class_ss10 = compile_simulator_cpp(params_som, params_den, dt=dt, ss=10)

    # Run the simulation
    N = ts.size
    sim_ss1, sim_ss10 = sim_class_ss1(n_neurons), sim_class_ss10(n_neurons)
    spikes_ss1, spikes_ss10 = np.zeros((2, N))
    gE, gI = 50e-9, 0e-9
    for i in range(N):
        x_exc = np.asarray((gE,) * n_neurons, order='C', dtype=np.float64)
        x_inh = np.asarray((gI,) * n_neurons, order='C', dtype=np.float64)
        sim_ss1.run_step_from_memory(spikes_ss1[i:i+1], x_exc, x_inh)
        sim_ss10.run_step_from_memory(spikes_ss10[i:i+1], x_exc, x_inh)

    spike_times_ss1 = np.where(spikes_ss1 != 0)[0] * dt
    spike_times_ss10 = np.where(spikes_ss10 != 0)[0] * dt

    assert spike_times_ss1.size == spike_times_ss10.size == 33
    assert np.sqrt(np.mean((spike_times_ss1 - spike_times_ss10) ** 2)) < 2e-3

