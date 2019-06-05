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

import nengo
import numpy as np

from nengo_bio.neurons import TwoCompLIF

class ReferenceTwoCompLIFSimulator:

    def __init__(self, nrn, dt, ss):
        self.nrn = nrn
        self.dt, self.ss = dt, ss
        self.v_som, self.v_den, self.tref = nrn.v_reset, nrn.v_reset, 0.

    def __call__(self, x):
        nrn = self.nrn
        v_som, v_den, tref, out = self.v_som, self.v_den, self.tref, 0.

        for s in range(self.ss):
            d_v_som = \
                 (nrn.E_rev_leak - v_som) * nrn.g_leak_som + \
                 (v_den - v_som) * nrn.g_couple
            d_v_den = \
                 (nrn.E_rev_leak - v_den) * nrn.g_leak_den + \
                 (nrn.E_rev_exc - v_den) * x[0] + \
                 (nrn.E_rev_inh - v_den) * x[1] + \
                 (v_som - v_den) * nrn.g_couple

            v_som += (self.dt / self.ss) * d_v_som / nrn.C_som
            v_den += (self.dt / self.ss) * d_v_den / nrn.C_den

            if tref > 0.:
                tref -= self.dt / self.ss
                v_som = nrn.v_spike if tref > nrn.tau_ref else nrn.v_reset

            if v_som > nrn.v_th and tref <= 0.:
                tref = nrn.tau_ref + nrn.tau_spike
                v_som = nrn.v_spike if nrn.tau_spike > 0. else nrn.v_reset
                out = 1. / self.dt

        self.v_som, self.v_den, self.tref = v_som, v_den, tref
        return out, (self.v_som, self.v_den)

def do_test_neuron(nrn, T=1.0, dt=1e-3):
    import time
    # Generate the input signals
    white_noise_process = nengo.processes.FilteredNoise(synapse=0.1, seed=3198)
    xs = np.array((
        white_noise_process.run(T, dt=dt)[:, 0],
        white_noise_process.run(T, dt=dt)[:, 0]
    )).T
    xs -= np.min(xs)
    xs[:, 0] *= 200e-9
    xs[:, 1] *= 100e-9

    # Construct the three simulators
    sim_ref = ReferenceTwoCompLIFSimulator(nrn, dt=dt, ss=nrn.subsample)
    sim_py = nrn.compile(dt, 1, None, force_python_sim=True)
    sim_cpp = nrn.compile(dt, 1, None)

    times = [[], [], []]

    # Run the simulation
    t = 0
    for i in range(xs.shape[0]):
        X = (xs[i],)
        t += dt
        o1 = sim_ref(xs[i])[0]
        o2 = sim_py(X)[0][0]
        o3 = sim_cpp(X)[0][0]
        assert o1 == o2 == o3

def test_two_comp_lif_simulators_default():
    nrn = TwoCompLIF()
    do_test_neuron(nrn)

def test_two_comp_lif_simulators_asym_membrane():
    nrn = TwoCompLIF(C_den=0.5e-9, C_som=2e-9)
    do_test_neuron(nrn)

def test_two_comp_lif_simulators_asym_leak():
    nrn = TwoCompLIF(g_leak_den=20e-9, g_leak_som=100e-9)
    do_test_neuron(nrn)
