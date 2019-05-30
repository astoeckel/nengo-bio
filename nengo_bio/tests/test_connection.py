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
from nengo_bio.connection import MultiEnsemble

@pytest.fixture
def nengo_ensembles():
    with nengo.Network() as net:
        ens_a = nengo.Ensemble(n_neurons=101, dimensions=1)
        ens_b = nengo.Ensemble(n_neurons=102, dimensions=1)
        ens_c = nengo.Ensemble(n_neurons=103, dimensions=2)
    return (ens_a, ens_b, ens_c)

def test_multi_ensemble_dims(nengo_ensembles):
    ens_a, ens_b, ens_c = nengo_ensembles

    mens = MultiEnsemble(ens_a)
    assert len(mens) == 1
    assert mens.dimensions == 1
    assert mens.size_in == 1
    assert mens.size_out == 1
    assert mens.n_neurons == 101

    mens = MultiEnsemble((ens_a,))
    assert len(mens) == 1
    assert mens.dimensions == 1
    assert mens.size_in == 1
    assert mens.size_out == 1
    assert mens.n_neurons == 101

    mens = MultiEnsemble((ens_a, ens_b))
    assert len(mens) == 2
    assert mens.dimensions == 2
    assert mens.size_in == 2
    assert mens.size_out == 2
    assert mens.n_neurons == 203

    mens = MultiEnsemble((ens_a, ens_b, (ens_c,)))
    assert len(mens) == 4
    assert mens.dimensions == 4
    assert mens.size_in == 4
    assert mens.size_out == 4
    assert mens.n_neurons == 306

    mens = MultiEnsemble({ens_a,})
    assert len(mens) == 1
    assert mens.dimensions == 1
    assert mens.size_in == 1
    assert mens.size_out == 1
    assert mens.n_neurons == 101

    mens = MultiEnsemble({ens_a, ens_b})
    assert len(mens) == 1
    assert mens.dimensions == 1
    assert mens.size_in == 1
    assert mens.size_out == 1
    assert mens.n_neurons == 203

    mens = MultiEnsemble({(ens_a, ens_b), ens_c})
    assert len(mens) == 2
    assert mens.dimensions == 2
    assert mens.size_in == 2
    assert mens.size_out == 2
    assert mens.n_neurons == 306


def test_multi_ensemble_flatten(nengo_ensembles):
    ens_a, ens_b, ens_c = nengo_ensembles

    # Single ensemble without operator
    mens = MultiEnsemble(ens_a)
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 1
    assert ens[0] == ens_a
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ds[0].start == 0 and ds[0].stop == 1
    assert js[0] == 0

    # Single ensemble with Stack operator
    mens = MultiEnsemble((ens_a,))
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 1
    assert ens[0] == ens_a
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ds[0].start == 0 and ds[0].stop == 1
    assert js[0] == 0

    # Single ensemble with Join operator
    mens = MultiEnsemble({ens_a,})
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 1
    assert ens[0] == ens_a
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ds[0].start == 0 and ds[0].stop == 1
    assert js[0] == 0

    # Two ensembles with Join operator
    mens = MultiEnsemble((ens_a, ens_b), operator=MultiEnsemble.OP_JOIN)
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 2
    assert (ens_a in ens) and (ens_b in ens)
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ns[1].start == 101 and ns[1].stop == 203
    assert ds[0].start == 0 and ds[0].stop == 1
    assert ds[1].start == 0 and ds[1].stop == 1
    assert js[0] == 0
    assert js[1] == 0

    # Two ensembles with Stack operator
    mens = MultiEnsemble((ens_a, ens_b), operator=MultiEnsemble.OP_STACK)
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 2
    assert (ens_a in ens) and (ens_b in ens)
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ns[1].start == 101 and ns[1].stop == 203
    assert ds[0].start == 0 and ds[0].stop == 1
    assert ds[1].start == 1 and ds[1].stop == 2
    assert js[0] == 0
    assert js[1] == 0

    # Three ensembles with Stack/Join operator
    mens = MultiEnsemble(((ens_a, ens_b), ens_c), 
                         operator=MultiEnsemble.OP_JOIN)
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 3
    assert ens == (ens_a, ens_b, ens_c)
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ns[1].start == 101 and ns[1].stop == 203
    assert ns[2].start == 203 and ns[2].stop == 306
    assert ds[0].start == 0 and ds[0].stop == 1
    assert ds[1].start == 1 and ds[1].stop == 2
    assert ds[2].start == 0 and ds[2].stop == 2
    assert js[0] == 0
    assert js[1] == 0
    assert js[2] == 1

    # Four ensembles with Stack/Join operator
    mens = MultiEnsemble((
               MultiEnsemble(((ens_a, ens_b), ens_c), 
                   operator=MultiEnsemble.OP_JOIN),
               ens_c))
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 4
    assert ens == (ens_a, ens_b, ens_c, ens_c)
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ns[1].start == 101 and ns[1].stop == 203
    assert ns[2].start == 203 and ns[2].stop == 306
    assert ns[3].start == 306 and ns[3].stop == 409
    assert ds[0].start == 0 and ds[0].stop == 1
    assert ds[1].start == 1 and ds[1].stop == 2
    assert ds[2].start == 0 and ds[2].stop == 2
    assert ds[3].start == 2 and ds[3].stop == 4
    assert js[0] == 0
    assert js[1] == 0
    assert js[2] == 1
    assert js[3] == 1

    # Five ensembles with Stack/Join operator
    mens = MultiEnsemble((
               MultiEnsemble(((ens_a, ens_b), ens_c, ens_c), 
                   operator=MultiEnsemble.OP_JOIN),
               ens_c))
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 5
    assert ens == (ens_a, ens_b, ens_c, ens_c, ens_c)
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ns[1].start == 101 and ns[1].stop == 203
    assert ns[2].start == 203 and ns[2].stop == 306
    assert ns[3].start == 306 and ns[3].stop == 409
    assert ns[4].start == 409 and ns[4].stop == 512
    assert ds[0].start == 0 and ds[0].stop == 1
    assert ds[1].start == 1 and ds[1].stop == 2
    assert ds[2].start == 0 and ds[2].stop == 2
    assert ds[3].start == 0 and ds[3].stop == 2
    assert ds[4].start == 2 and ds[4].stop == 4
    assert js[0] == 0
    assert js[1] == 0
    assert js[2] == 1
    assert js[3] == 1
    assert js[4] == 1

    # Six ensembles with Stack/Join operator
    mens = MultiEnsemble((
               MultiEnsemble(((ens_a, ens_b), (ens_a, ens_b), ens_c), 
                   operator=MultiEnsemble.OP_JOIN),
               ens_c))
    ens, ns, ds, js = mens.flatten()
    assert len(ens) == len(ns) == len(ds) == len(js) == 6
    assert ens == (ens_a, ens_b, ens_a, ens_b, ens_c, ens_c)
    assert ns[0].start == 0 and ns[0].stop == 101
    assert ns[1].start == 101 and ns[1].stop == 203
    assert ns[2].start == 203 and ns[2].stop == 304
    assert ns[3].start == 304 and ns[3].stop == 406
    assert ns[4].start == 406 and ns[4].stop == 509
    assert ns[5].start == 509 and ns[5].stop == 612
    assert ds[0].start == 0 and ds[0].stop == 1
    assert ds[1].start == 1 and ds[1].stop == 2
    assert ds[2].start == 0 and ds[2].stop == 1
    assert ds[3].start == 1 and ds[3].stop == 2
    assert ds[4].start == 0 and ds[4].stop == 2
    assert ds[5].start == 2 and ds[5].stop == 4
    assert js[0] == 0
    assert js[1] == 0
    assert js[2] == 1
    assert js[3] == 1
    assert js[4] == 2
    assert js[5] == 2

def test_multi_ensemble_exceptions(nengo_ensembles):
    ens_a, ens_b, ens_c = nengo_ensembles

    # Empty MultiEnsemble
    with pytest.raises(ValueError) as _:
        mens = MultiEnsemble(set())
    with pytest.raises(ValueError) as _:
        mens = MultiEnsemble(tuple())
    with pytest.raises(ValueError) as _:
        mens = MultiEnsemble((ens_a, tuple()))

    # Not the same dimension
    with pytest.raises(ValueError) as _:
        mens = MultiEnsemble({ens_a, ens_c})

    # Not an ensemble
    with pytest.raises(ValueError) as _:
        mens = MultiEnsemble(None)

