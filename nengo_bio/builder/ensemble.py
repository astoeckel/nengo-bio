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
import nengo.utils.numpy as npext

from nengo_bio.common import Excitatory, Inhibitory
from nengo_bio.ensemble import Ensemble
from nengo_bio.neurons import MultiChannelNeuronType

from nengo.builder.operator import Reset
import nengo.dists
import nengo.builder.ensemble
import nengo.builder.signal


built_attrs = nengo.builder.ensemble.built_attrs + ["synapse_types", "tuning"]
class BuiltEnsemble(collections.namedtuple('BuiltEnsemble', built_attrs)):
    pass

@nengo.builder.Builder.register(Ensemble)
def build_ensemble(model, ens):
    # Fetch the random number generator
    rng = np.random.RandomState(model.seeds[ens])

    # Select the synapse types
    if ens._p_exc is None:
        # Mark all neurons as both excitatory and inhibitory if no probabilities
        # are given
        synapse_types_exc = np.ones(ens.n_neurons, dtype=np.bool)
        synapse_types_inh = np.ones(ens.n_neurons, dtype=np.bool)
    else:
        # Otherwise select a random neuron type per neuron
        synapse_types_exc = rng.choice(
            [True, False], ens.n_neurons, p=(ens.p_exc, ens.p_inh))
        synapse_types_inh = ~synapse_types_exc

    # Store the model parameters in the extended BuiltEnsemble named tuple
    synapse_types = {
        Excitatory: synapse_types_exc,
        Inhibitory: synapse_types_inh
    }

    # If this ensemble uses a default neuron type, just call the default
    # ensemble build function.
    if not isinstance(ens.neuron_type, MultiChannelNeuronType):
        nengo.builder.ensemble.build_ensemble(model, ens)
        model.params[ens] = BuiltEnsemble(*model.params[ens], synapse_types, None)
        return

    # Otherwise generate the evaluation points, encoders, gains, biases manually
    eval_points = nengo.builder.ensemble.gen_eval_points(
        ens, ens.eval_points, rng=rng)
    if isinstance(ens.encoders, nengo.dists.Distribution):
        encoders = nengo.dists.get_samples(
            ens.encoders, ens.n_neurons, ens.dimensions, rng=rng)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
    if ens.normalize_encoders:
        encoders /= npext.norm(encoders, axis=1, keepdims=True)
    gain, bias, max_rates, intercepts = \
        nengo.builder.ensemble.get_gain_bias(ens, rng)
    scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    # Store the parameters in the BuiltEnsemble instance
    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias,
                                      synapse_types=synapse_types,
                                      tuning=None)

    # Call the "tune" function of the neuron type to give the neuron type
    # the opportunity to find some optimal model parameters
    tuning = ens.neuron_type.tune(model.dt, model, ens)
    model.params[ens] = model.params[ens]._replace(tuning=tuning)

    # Setup dummy input signals causing the original nengo code to work
    n_neurons, n_inputs = ens.n_neurons, ens.neuron_type.n_inputs
    sig = nengo.builder.signal.Signal(
        np.zeros(0), name="{}.neuron_in".format(ens))
    model.sig[ens.neurons]['in'] = sig
    model.sig[ens]['in'] =  None

    # Setup the actual input signals
    for i in range(n_inputs):
        sig = nengo.builder.signal.Signal(
            np.zeros(n_neurons), name="{}.neuron_in".format(ens))
        model.sig[ens.neurons]['in_{}'.format(i)] = sig
        model.add_op(Reset(sig))

    # Output signal
    sig = nengo.builder.signal.Signal(
        np.zeros(n_neurons), name="{}.neuron_out".format(ens))
    model.sig[ens.neurons]['out'] = sig
    model.sig[ens]['out'] = sig

    # This adds the neuron's operator and sets other signals
    model.build(ens.neuron_type, ens.neurons)

