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

def _coerce_synapse_type_probabilities(p_exc, p_inh):
    has_p_exc, has_p_inh = not p_exc is None, not p_inh is None
    if has_p_exc or has_p_inh:
        # Both p_exc and p_inh are given must sum to one
        if has_p_exc and has_p_inh:
            if abs(p_exc + p_inh - 1.0) > 1e-3:
                raise BuildError(
                    "p_exc={} and p_inh={} do not add up to one for {}"
                    .format(p_exc, p_inh, ens)
                )

        # At least p_exc is given, check range
        if has_p_exc:
            if p_exc < 0.0 or p_exc > 1.0:
                raise BuildError(
                    "p_exc={} must be between 0.0 and 1.0 for {}"
                    .format(p_exc, ens))
            p_inh = 1.0 - p_exc
        # At least p_inh is given, check range
        elif has_p_inh:
            if p_inh < 0.0 or p_inh > 1.0:
                raise BuildError(
                    "p_inh={} must be between 0.0 and 1.0 for {}"
                    .format(p_inh, ens))
            p_exc = 1.0 - p_inh
    return p_exc, p_inh

built_attrs = nengo.builder.ensemble.built_attrs + ["synapse_types"]
class BuiltEnsemble(collections.namedtuple('BuiltEnsemble', built_attrs)):
    pass

@nengo.builder.Builder.register(Ensemble)
def build_ensemble(model, ens):
    # Call the original build_ensemble function
    nengo.builder.ensemble.build_ensemble(model, ens)

    # Validate p_exc/p_inh
    p_exc, p_inh = _coerce_synapse_type_probabilities(ens.p_exc, ens.p_inh)

    # Select the neuron types
    if (p_exc is None) and (p_inh is None):
        # Mark all neurons as both excitatory and inhibitory if no probabilities
        # are given
        synapse_types_exc = np.ones(ens.n_neurons, dtype=np.bool)
        synapse_types_inh = np.ones(ens.n_neurons, dtype=np.bool)
    else:
        # Otherwise select a random neuron type per neuron
        rng = np.random.RandomState(model.seeds[ens])
        synapse_types_exc = rng.choice(
            [True, False], ens.n_neurons, p=(p_exc, p_inh))
        synapse_types_inh = ~synapse_types_exc

    # Store the model parameters in the extended BuiltEnsemble named tuple
    synapse_types = {
        Excitatory: synapse_types_exc,
        Inhibitory: synapse_types_inh
    }
    model.params[ens] = BuiltEnsemble(*model.params[ens], synapse_types)

