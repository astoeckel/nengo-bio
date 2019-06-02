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

import collections
import numpy as np

from nengo_bio.common import Excitatory, Inhibitory
from nengo_bio.ensemble import Ensemble

import nengo.builder.ensemble

built_attrs = nengo.builder.ensemble.built_attrs + ["synapse_types"]
class BuiltEnsemble(collections.namedtuple('BuiltEnsemble', built_attrs)):
    pass

@nengo.builder.Builder.register(Ensemble)
def build_ensemble(model, ens):
    # Call the original build_ensemble function
    nengo.builder.ensemble.build_ensemble(model, ens)

    # Select the neuron types
    if ens._p_exc is None:
        # Mark all neurons as both excitatory and inhibitory if no probabilities
        # are given
        synapse_types_exc = np.ones(ens.n_neurons, dtype=np.bool)
        synapse_types_inh = np.ones(ens.n_neurons, dtype=np.bool)
    else:
        # Otherwise select a random neuron type per neuron
        rng = np.random.RandomState(model.seeds[ens])
        synapse_types_exc = rng.choice(
            [True, False], ens.n_neurons, p=(ens.p_exc, ens.p_inh))
        synapse_types_inh = ~synapse_types_exc

    # Store the model parameters in the extended BuiltEnsemble named tuple
    synapse_types = {
        Excitatory: synapse_types_exc,
        Inhibitory: synapse_types_inh
    }
    model.params[ens] = BuiltEnsemble(*model.params[ens], synapse_types)

