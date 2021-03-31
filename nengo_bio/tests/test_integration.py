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
import nengo_bio as bio
import numpy as np

PROBE_SYNAPSE = 0.1
T = 10.0
T_SKIP = 1.0

def run_and_compute_relative_rmse(model, probe, expected_fns):
    # Run the simulation for the specified time
    with nengo.Simulator(model, progress_bar=None) as sim:
        sim.run(T)

    # Fetch the time and the probe data
    ts = sim.trange()
    expected = np.array([f(ts - PROBE_SYNAPSE) for f in expected_fns]).T
    actual = sim.data[probe]

    # Compute the slice over which to compute the error
    slice_ = slice(int(T_SKIP / sim.dt), int(T / sim.dt))

    # Compute the RMSE and the RMSE
    rms = np.sqrt(np.mean(np.square(expected)))
    rmse = np.sqrt(np.mean(np.square(expected[slice_] - actual[slice_])))

    return rmse / rms

def test_communication_channel():
    f1, f2 = lambda t: np.sin(t), lambda t: np.cos(t)
    with nengo.Network(seed=5892) as model:
        inp_a = nengo.Node(f1)
        inp_b = nengo.Node(f2)

        ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8)
        ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.8)
        ens_c = bio.Ensemble(n_neurons=103, dimensions=2)

        nengo.Connection(inp_a, ens_a)
        nengo.Connection(inp_b, ens_b)

        bio.Connection((ens_a, ens_b), ens_c)

        prb_output = nengo.Probe(ens_c, synapse=PROBE_SYNAPSE)

    assert run_and_compute_relative_rmse(model, prb_output, (f1, f2)) < 0.1


def test_communication_channel_with_post_radius():
    f1, f2 = lambda t: np.sin(t), lambda t: np.cos(t)
    with nengo.Network(seed=5892) as model:
        inp_a = nengo.Node(f1)
        inp_b = nengo.Node(f2)

        ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8)
        ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.8)
        ens_c = bio.Ensemble(n_neurons=103, dimensions=2, radius=2)

        nengo.Connection(inp_a, ens_a)
        nengo.Connection(inp_b, ens_b)

        bio.Connection((ens_a, ens_b), ens_c)

        probe = nengo.Probe(ens_c, synapse=PROBE_SYNAPSE)

    assert run_and_compute_relative_rmse(model, probe, (f1, f2)) < 0.1


def test_communication_channel_with_pre_radius():
    f1, f2 = lambda t: np.sin(t), lambda t: np.cos(t)
    with nengo.Network(seed=5892) as model:
        inp_a = nengo.Node(f1)
        inp_b = nengo.Node(f2)

        ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8, radius=2)
        ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.8, radius=2)
        ens_c = bio.Ensemble(n_neurons=103, dimensions=2)

        nengo.Connection(inp_a, ens_a)
        nengo.Connection(inp_b, ens_b)

        bio.Connection((ens_a, ens_b), ens_c)

        prb_output = nengo.Probe(ens_c, synapse=PROBE_SYNAPSE)

    assert run_and_compute_relative_rmse(model, prb_output, (f1, f2)) < 0.25


def _test_communication_channel_bias_modes(bias_mode):
    f1 = lambda t: np.sin(t)
    with nengo.Network(seed=5892) as model:
        inp_a = nengo.Node(f1)

        ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.5)
        ens_b = bio.Ensemble(n_neurons=102, dimensions=1)

        nengo.Connection(inp_a, ens_a)

        bio.Connection(ens_a, ens_b, bias_mode=bias_mode)

        prb_output = nengo.Probe(ens_b, synapse=PROBE_SYNAPSE)

    assert run_and_compute_relative_rmse(model, prb_output, (f1,)) < 0.1


def test_communication_channel_bias_mode_decode():
    _test_communication_channel_bias_modes(bio.Decode)


def test_communication_channel_bias_mode_jbias():
    _test_communication_channel_bias_modes(bio.JBias)


def test_communication_channel_bias_mode_exc_jbias():
    _test_communication_channel_bias_modes(bio.ExcJBias)


def test_communication_channel_bias_mode_inh_jbias():
    _test_communication_channel_bias_modes(bio.InhJBias)


def test_parisien():
    with nengo.Network(seed=5892) as model:
        inp_a = nengo.Node(lambda t: np.sin(t))

        # Excitatory source population
        ens_source = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=1.0)

        # Inhibitory inter-neuron population
        ens_inhint = bio.Ensemble(n_neurons=102, dimensions=1, p_inh=1.0)

        # Target population
        ens_target = bio.Ensemble(n_neurons=103, dimensions=1)

        nengo.Connection(inp_a, ens_source)
        bio.Connection(ens_source, ens_inhint)
        bio.Connection({ens_source, ens_inhint}, ens_target,
                       function=lambda x: np.mean(x)**2)

        probe = nengo.Probe(ens_target, synapse=PROBE_SYNAPSE)

    assert run_and_compute_relative_rmse(
        model, probe, (lambda t: np.sin(t)**2,)) < 0.25


def test_parisien_relax():
    with nengo.Network(seed=5892) as model:
        inp_a = nengo.Node(lambda t: np.sin(t))

        # Excitatory source population
        ens_source = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=1.0)

        # Inhibitory inter-neuron population
        ens_inhint = bio.Ensemble(n_neurons=102, dimensions=1, p_inh=1.0)

        # Target population
        ens_target = bio.Ensemble(n_neurons=103, dimensions=1)

        nengo.Connection(inp_a, ens_source)
        bio.Connection(ens_source, ens_inhint,
                       solver=bio.solvers.QPSolver(relax=True))
        bio.Connection({ens_source, ens_inhint}, ens_target,
                       function=lambda x: np.mean(x)**2,
                       solver=bio.solvers.QPSolver(relax=True))

        probe = nengo.Probe(ens_target, synapse=PROBE_SYNAPSE)

    assert run_and_compute_relative_rmse(
        model, probe, (lambda t: np.sin(t)**2,)) < 0.05


def test_multi_channel_lif_communication_channel():
    f1, f2 = lambda t: np.sin(t), lambda t: np.cos(t)
    with nengo.Network(seed=5892) as model:
        inp_a = nengo.Node(f1)
        inp_b = nengo.Node(f2)

        ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8)
        ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.8)
        ens_c = bio.Ensemble(n_neurons=103, dimensions=2,
                         neuron_type=bio.neurons.LIF(),
                         max_rates=nengo.dists.Uniform(100, 200))

        nengo.Connection(inp_a, ens_a)
        nengo.Connection(inp_b, ens_b)

        bio.Connection((ens_a, ens_b), ens_c)

        prb_output = nengo.Probe(ens_c, synapse=PROBE_SYNAPSE)

    assert run_and_compute_relative_rmse(model, prb_output, (f1, f2)) < 0.25
