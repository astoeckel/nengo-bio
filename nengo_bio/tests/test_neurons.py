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

from nengo_bio.internal.multi_compartment_lif_sim import PoissonSource
from nengo_bio.neurons import TwoCompLIF


class ReferenceTwoCompLIFSimulator:
    def __init__(self, nrn, dt, ss):
        self.nrn = nrn
        self.dt, self.ss = dt, ss
        self.v_som, self.v_den, self.tref = nrn.v_reset, nrn.v_reset, 0.

    def __call__(self, out, gE, gI):
        nrn = self.nrn
        v_som, v_den, tref = self.v_som, self.v_den, self.tref

        for s in range(self.ss):
            d_v_som = \
                 (nrn.E_rev_leak - v_som) * nrn.g_leak_som + \
                 (v_den - v_som) * nrn.g_couple
            d_v_den = \
                 (nrn.E_rev_leak - v_den) * nrn.g_leak_den + \
                 (nrn.E_rev_exc - v_den) * gE + \
                 (nrn.E_rev_inh - v_den) * gI + \
                 (v_som - v_den) * nrn.g_couple

            v_som += (self.dt / self.ss) * d_v_som / nrn.C_som
            v_den += (self.dt / self.ss) * d_v_den / nrn.C_den

            if tref > 0.:
                tref -= self.dt / self.ss
                v_som = nrn.v_spike if tref > nrn.tau_ref else nrn.v_reset

            if v_som > nrn.v_th and tref <= 0.:
                tref = nrn.tau_ref + nrn.tau_spike
                v_som = nrn.v_spike if nrn.tau_spike > 0. else nrn.v_reset
                out[...] = 1. / self.dt

        self.v_som, self.v_den, self.tref = v_som, v_den, tref


def do_test_neuron(nrn, T=1.0, dt=1e-3):
    # Generate the input signals
    white_noise_process = nengo.processes.FilteredNoise(synapse=0.1, seed=3198)
    x_exc = white_noise_process.run(T, dt=dt)[:, 0] * 200e-9
    x_inh = white_noise_process.run(T, dt=dt)[:, 0] * 100e-9

    # Construct the three simulators
    sim_ref = ReferenceTwoCompLIFSimulator(nrn, dt=dt, ss=nrn.subsample)
    sim_py = nrn.compile(dt, 1, None, force_python_sim=True)
    sim_cpp = nrn.compile(dt, 1, None)

    # Run the simulation
    t = 0
    N = x_exc.size
    for i in range(N):
        t += dt
        out = np.zeros(3)
        sim_ref(out[0:1], x_exc[i:i + 1], x_inh[i:i + 1])
        sim_py(out[1:2], x_exc[i:i + 1], x_inh[i:i + 1])
        sim_cpp(out[2:3], x_exc[i:i + 1], x_inh[i:i + 1])
        assert out[0] == out[1] == out[2]


def test_two_comp_lif_simulators_default():
    nrn = TwoCompLIF()
    do_test_neuron(nrn)


def test_two_comp_lif_simulators_asym_membrane():
    nrn = TwoCompLIF(C_den=0.5e-9, C_som=2e-9)
    do_test_neuron(nrn)


def test_two_comp_lif_simulators_asym_leak():
    nrn = TwoCompLIF(g_leak_den=20e-9, g_leak_som=100e-9)
    do_test_neuron(nrn)


def do_test_poisson_sources(nrn, sources, T=10.0, dt=1e-3):
    # Construct the C++ and Python simulator
    sim_py = nrn.compile(dt, 1, None, force_python_sim=True, get_class=True)(1).run_poisson
    sim_cpp = nrn.compile(dt, 1, None, get_class=True)(1).run_poisson

    # Run the simulation
    ts = np.arange(0, T, dt)
    out_py, out_cpp = np.zeros((2, len(ts)))
    sim_py(out_py, sources)
    sim_cpp(out_cpp, sources)

    return ts, out_py, out_cpp

def test_two_comp_lif_poisson_simulators():
    nrn = TwoCompLIF()
    ts, out_py, out_cpp = do_test_poisson_sources(nrn, [
        PoissonSource(seed=3812, rate=100, gain_min=0.0, gain_max=1e-6, tau=50e-3, offs=0.0),
        PoissonSource(seed=3812, rate=50, gain_min=0.0, gain_max=0.5e-6, tau=50e-3, offs=0.0),
    ])
    times_py = ts[out_py > 0.0]
    times_cpp = ts[out_cpp > 0.0]
    rate_py = 1.0 / np.mean(times_py[1:] - times_py[:-1])
    rate_cpp = 1.0 / np.mean(times_cpp[1:] - times_cpp[:-1])
    assert np.abs(rate_py - rate_cpp) < 2.0

def test_two_comp_lif_poisson_simulators_constant():
    nrn = TwoCompLIF()
    _, out_py, out_cpp = do_test_poisson_sources(nrn, [
        PoissonSource(seed=3812, rate=100, gain_min=0.0, gain_max=0.0, tau=100e-3, offs=1e-6),
        PoissonSource(seed=3812, rate=100, gain_min=0.0, gain_max=0.0, tau=100e-3, offs=0.5e-6),
    ])
    assert np.all(out_py == out_cpp)

