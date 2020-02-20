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

import pytest

import nengo
import numpy as np

from nengo_bio.ensemble import Ensemble
from nengo_bio.connection import MultiEnsemble, Connection, ConstrainedConnectivity
from nengo_bio.builder.connection import \
    get_multi_ensemble_eval_points, get_multi_ensemble_synapse_types,\
    get_connection_matrix

@pytest.fixture
def nengo_ensembles_and_model():
    with nengo.Network() as net:
        ens_a = Ensemble(n_neurons=101, dimensions=1, p_exc=1.0,
                         eval_points=nengo.dists.Choice([[1]]),
                         n_eval_points=10)
        ens_b = Ensemble(n_neurons=102, dimensions=1, p_inh=1.0,
                         eval_points=nengo.dists.Choice([[2]]),
                         n_eval_points=11)
        ens_c = Ensemble(n_neurons=103, dimensions=2, p_exc = 0.5,
                         eval_points=nengo.dists.Choice([[3, 4]]),
                         n_eval_points=12)
        ens_d = Ensemble(n_neurons=102, dimensions=1,
                         eval_points=nengo.dists.Choice([[5]]),
                          n_eval_points=13)
    with nengo.Simulator(net, progress_bar=None) as sim:
        pass
    return (ens_a, ens_b, ens_c, ens_d, sim.model)

def test_get_eval_points(nengo_ensembles_and_model):
    ens_a, ens_b, ens_c, ens_d, model = nengo_ensembles_and_model

    rng = np.random.RandomState(4871)

    mens = MultiEnsemble(ens_a)
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 10
    assert eval_points.shape[1] == 1
    assert np.all((eval_points == [1.0]).flatten())

    mens = MultiEnsemble((ens_a, ens_b))
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 11
    assert eval_points.shape[1] == 2
    assert np.all((eval_points == [1.0, 2.0]).flatten())

    mens = MultiEnsemble((ens_a, ens_d))
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 13
    assert eval_points.shape[1] == 2
    assert np.all((eval_points == [1.0, 5.0]).flatten())

    mens = MultiEnsemble((ens_a, ens_d), operator=MultiEnsemble.OP_JOIN)
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 13
    assert eval_points.shape[1] == 1
    assert np.all(np.logical_or(eval_points == 1.0, eval_points == 5.0))

    mens = MultiEnsemble(((ens_a, ens_b), ens_c),
                         operator=MultiEnsemble.OP_JOIN)
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 12
    assert eval_points.shape[1] == 2
    assert np.all((np.logical_or(
        np.prod(eval_points == [1.0, 2.0], axis=1),
        np.prod(eval_points == [3.0, 4.0], axis=1)
    )))

    mens = MultiEnsemble(((ens_a, ens_b), ens_c),
                         operator=MultiEnsemble.OP_STACK)
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 12
    assert eval_points.shape[1] == 4
    assert np.all((eval_points == [1.0, 2.0, 3.0, 4.0]).flatten())

    mens = MultiEnsemble(((ens_a, ens_b), ens_d),
                         operator=MultiEnsemble.OP_STACK)
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 13
    assert eval_points.shape[1] == 3
    assert np.all((eval_points == [1.0, 2.0, 5.0]).flatten())

    mens = MultiEnsemble((
               MultiEnsemble(((ens_a, ens_b), (ens_a, ens_b), ens_c), 
                   operator=MultiEnsemble.OP_JOIN),
               ens_d))
    eval_points = get_multi_ensemble_eval_points(model, mens, rng=rng)
    assert eval_points.shape[0] == 13
    assert eval_points.shape[1] == 3
    assert np.all((np.logical_or(
        np.prod(eval_points == [1.0, 2.0, 5.0], axis=1),
        np.prod(eval_points == [3.0, 4.0, 5.0], axis=1)
    )))

def test_get_eval_points_fixed(nengo_ensembles_and_model):
    ens_a, ens_b, ens_c, ens_d, model = nengo_ensembles_and_model

    rng = np.random.RandomState(4871)

    for N in [0, 5, 20]:
        mens = MultiEnsemble(ens_a)
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 1
        assert np.all((eval_points == [1.0]).flatten())

        mens = MultiEnsemble((ens_a, ens_b))
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 2
        assert np.all((eval_points == [1.0, 2.0]).flatten())

        mens = MultiEnsemble((ens_a, ens_d))
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 2
        assert np.all((eval_points == [1.0, 5.0]).flatten())

        mens = MultiEnsemble((ens_a, ens_d), operator=MultiEnsemble.OP_JOIN)
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 1
        assert np.all(np.logical_or(eval_points == 1.0, eval_points == 5.0))

        mens = MultiEnsemble(((ens_a, ens_b), ens_c),
                             operator=MultiEnsemble.OP_JOIN)
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 2
        assert np.all((np.logical_or(
            np.prod(eval_points == [1.0, 2.0], axis=1),
            np.prod(eval_points == [3.0, 4.0], axis=1)
        )))

        mens = MultiEnsemble(((ens_a, ens_b), ens_c),
                             operator=MultiEnsemble.OP_STACK)
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 4
        assert np.all((eval_points == [1.0, 2.0, 3.0, 4.0]).flatten())

        mens = MultiEnsemble(((ens_a, ens_b), ens_d),
                             operator=MultiEnsemble.OP_STACK)
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 3
        assert np.all((eval_points == [1.0, 2.0, 5.0]).flatten())

        mens = MultiEnsemble((
                   MultiEnsemble(((ens_a, ens_b), (ens_a, ens_b), ens_c), 
                       operator=MultiEnsemble.OP_JOIN),
                   ens_d))
        eval_points = get_multi_ensemble_eval_points(model, mens, N, rng=rng)
        assert eval_points.shape[0] == N
        assert eval_points.shape[1] == 3
        assert np.all((np.logical_or(
            np.prod(eval_points == [1.0, 2.0, 5.0], axis=1),
            np.prod(eval_points == [3.0, 4.0, 5.0], axis=1)
        )))

def test_standard_pre_post_objs(nengo_ensembles_and_model):
    with nengo.Network() as net:
        ens_a = nengo.Ensemble(n_neurons=11, dimensions=1)
        ens_b = nengo.Ensemble(n_neurons=12, dimensions=1)
        conn = Connection(ens_a, ens_b)
    with nengo.Simulator(net, progress_bar=None) as sim:
        pass

    synapse_types = get_multi_ensemble_synapse_types(sim.model, conn.pre_obj)
    connection_matrix = get_connection_matrix(sim.model, conn, synapse_types)
    assert np.prod(connection_matrix.flatten()) == 1

def test_no_exc_inh(nengo_ensembles_and_model):
    with nengo.Network() as net:
        ens_a = Ensemble(n_neurons=11, dimensions=1)
        ens_b = Ensemble(n_neurons=12, dimensions=1)
        conn = Connection(ens_a, ens_b)
    with nengo.Simulator(net, progress_bar=None) as sim:
        pass

    synapse_types = get_multi_ensemble_synapse_types(sim.model, conn.pre_obj)
    connection_matrix = get_connection_matrix(sim.model, conn, synapse_types)
    assert np.prod(connection_matrix.flatten()) == 1

def test_get_connection_matrix(nengo_ensembles_and_model):
    ens_a, ens_b, ens_c, ens_d, model = nengo_ensembles_and_model

    # ens_a is excitatory, ens_b is inhibitory
    with model.toplevel:
        conn = Connection((ens_a, ens_b), ens_c)
    synapse_types = get_multi_ensemble_synapse_types(model, conn.pre_obj)
    connection_matrix = get_connection_matrix(model, conn, synapse_types)

    assert np.prod(connection_matrix[0, :101, :].flatten()) == 1
    assert np.sum(connection_matrix[0, 101:, :].flatten()) == 0
    assert np.sum(connection_matrix[1, :101, :].flatten()) == 0
    assert np.prod(connection_matrix[1, 101:, :].flatten()) == 1

    # ens_d allows all connections
    with model.toplevel:
        conn = Connection(ens_d, ens_d)
    synapse_types = get_multi_ensemble_synapse_types(model, conn.pre_obj)
    connection_matrix = get_connection_matrix(model, conn, synapse_types)
    assert np.prod(connection_matrix.flatten()) == 1

    # ens_a is excitatory, ens_b is inhibitory
    with model.toplevel:
        conn = Connection(
            (ens_a, ens_b), ens_c,
            connectivity=ConstrainedConnectivity(convergence=10))
    synapse_types = get_multi_ensemble_synapse_types(model, conn.pre_obj)
    connection_matrix = get_connection_matrix(model, conn, synapse_types)

    for i in range(connection_matrix.shape[2]):
        assert np.sum(connection_matrix[:, :, i]) == 20

    # ens_a is excitatory, ens_b is inhibitory
    with model.toplevel:
        conn = Connection(
            (ens_a, ens_b), ens_c,
            connectivity=ConstrainedConnectivity(divergence=10))
    synapse_types = get_multi_ensemble_synapse_types(model, conn.pre_obj)
    connection_matrix = get_connection_matrix(model, conn, synapse_types)

    for i in range(connection_matrix.shape[1]):
        assert np.sum(connection_matrix[:, i, :]) == 10


