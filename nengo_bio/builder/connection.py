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

import warnings
import numpy as np

from nengo_bio.common import Excitatory, Inhibitory
from nengo_bio.connection import ConnectionPart, Connection
from nengo_bio.solvers import SolverWrapper, ExtendedSolver
from nengo_bio.neurons import MultiChannelNeuronType

from nengo.exceptions import NengoWarning, BuildError
from nengo.ensemble import Ensemble, Neurons

import nengo.builder
from nengo.builder.operator import Copy, Reset

class BuiltConnection:
    def __init__(self):
        self.weights = {
            Excitatory: None,
            Inhibitory: None
        }

def get_multi_ensemble_eval_points(
    model, mens, n_eval_points=None, scale_eval_points=True, rng=np.random):
    """
    This function generates the evaluation points for the given MultiEnsemble.
    """

    def choice(A, n):
        return A[rng.randint(0, A.shape[0], n)]

    if mens.operator == mens.OP_NONE:
        # Recursion base case. The MultiEnsemble encapsulates a single Ensemble
        # instance -- just return the evaluation points associated with this
        # ensemble.
        pnts = model.params[mens.objs[0]].eval_points
        pnts.setflags(write=False)

        # Scale the evaluation points in case this was requested by the
        # connection
        if scale_eval_points:
            pnts = pnts * mens.objs[0].radius

        # In case a specific number of eval points is requested, create a random
        # selection of eval points
        if not n_eval_points is None:
            pnts = choice(pnts, n_eval_points)
        return pnts
    elif (mens.operator == mens.OP_STACK) or (mens.operator == mens.OP_JOIN):
        # For each MultiEnsemble object in the stack/join, fetch the evaluation
        # points associated with that MultiEnsemble. Track the maximum number
        # of evaluation points.
        pnts_per_obj, n_pnts = [None] * len(mens.objs), [0] * len(mens.objs)
        for i, obj in enumerate(mens.objs):
            pnts = get_multi_ensemble_eval_points(
                model, mens.objs[i], n_eval_points, scale_eval_points, rng)
            pnts_per_obj[i] = pnts
            n_pnts[i] = pnts.shape[0]
        max_n_pnts = max(n_pnts)

        # Either select n_eval_points or the maximum number of eval points
        # as the number of evaluation points to generate.
        if n_eval_points is None:
            n_eval_points = max_n_pnts

        if mens.operator == mens.OP_STACK:
            # Write the evaluation points to a contiguous array.
            pnts = np.empty((max_n_pnts, mens.dimensions))
            d = 0
            for i, p in enumerate(pnts_per_obj):
                n_pnts, n_dims = p.shape
                if n_eval_points >= max_n_pnts:
                    # Write the points to the resulting array, fill the
                    # remaining space with randomly selected samples
                    pnts[:n_pnts, d:(d+n_dims)] = p
                    pnts[n_pnts:, d:(d+n_dims)] = \
                        choice(p, n_eval_points - n_pnts)
                else:
                    # Ranomly select n_eval_points points and write them to
                    # the target array
                    pnts[:, d:(d+n_dims)] = choice(p, n_eval_points)

                # Increment the dimension counter
                d += n_dims
            return pnts
        elif mens.operator == mens.OP_JOIN:
            # Write the evaluation points to a contiguous array and select
            # max_n_pnts of those
            return choice(np.concatenate(pnts_per_obj, axis=0), n_eval_points)
    else:
        raise ValueError("Invalid MultiEnsemble operator")


def get_eval_points(model, conn, rng=np.random):
    # In case no eval_points object has been specified
    if conn.eval_points is None:
        return get_multi_ensemble_eval_points(
            model, conn.pre_obj, conn.n_eval_points, conn.scale_eval_points,
            rng)
    else:
        if conn.scale_eval_points:
            warnings.warn(NengoWarning(
                "Setting scale_eval_points to True has no effect on {} if "
                "eval_points are specified manually.".format(conn.pre_obj)))
        return nengo.builder.ensemble.gen_eval_points(
            conn, conn.eval_points, rng, False)


def get_multi_ensemble_activities(model, mens, eval_points):
    # Create the empty activity matrix
    n_eval_points = eval_points.shape[0]
    n_neurons = mens.n_neurons
    activities = np.empty((n_eval_points, n_neurons))

    # Iterate over all ensembles and ask Nengo to compute the activities. Write
    # the results to the activity matrix.
    arr_ens, arr_ns, arr_ds = mens.flatten()
    for ens, ns, ds in zip(arr_ens, arr_ns, arr_ds):
        activities[:, ns] = nengo.builder.ensemble.get_activities(
            model.params[ens], ens, eval_points[:, ds])

    return activities


def get_multi_ensemble_synapse_types(model, mens):
    # Create the empty synapse type map
    n_neurons = mens.n_neurons
    synapse_types = np.empty((2, n_neurons), dtype=np.bool)

    # Iterate over all ensembles and write the synapse types to the map
    arr_ens, arr_ns, _ = mens.flatten()
    for ens, ns in zip(arr_ens, arr_ns):
        for i, type_ in enumerate((Excitatory, Inhibitory)):
            built_ens = model.params[ens]
            if hasattr(built_ens, "synapse_types"):
                synapse_types[i, ns] = built_ens.synapse_types[type_]
            else:
                synapse_types[i, ns] = np.ones(ens.n_neurons, dtype=np.bool)
    return synapse_types


def get_connectivity(conn, synapse_types, rng=np.random):
    # Create some convenient aliases
    Npre, Npost = conn.pre_obj.n_neurons, conn.post_obj.n_neurons
    mps = conn.max_n_post_synapses
    mps_E, mps_I = conn.max_n_post_synapses_exc, conn.max_n_post_synapses_inh
    has_mps = not (mps is None)
    has_mps_E, has_mps_I = not (mps_E is None), not (mps_I is None)

    # If no restrictions were given, just allow all-to-all connections
    if not (has_mps or has_mps_E or has_mps_I):
        return np.array((
            np.tile(synapse_types[0, :, None], Npost),
            np.tile(synapse_types[1, :, None], Npost)),
            dtype=np.bool)
    if has_mps and has_mps_E and has_mps_I:
        raise BuildError(
            "Specifying max_n_post_synapses as well as both "
            "max_n_post_synapses_exc and max_n_post_synapses_inh is invalid.")

    # Limit mps_E and mps_I to mps
    if has_mps:
        if has_mps_E:
            mps_E = min(mps, mps_E)
        if has_mps_I:
            mps_I = min(mps, mps_I)

    connectivity = np.zeros((2, Npre, Npost), dtype=np.bool)
    for i_post in range(Npost):
        # Get the indices of possible excitatory connection sites
        i_exc, i_inh = np.where(synapse_types[0])[0], np.where(synapse_types[1])[0]
        n_exc, n_inh = i_exc.size, i_inh.size

        # Select mps_E excitatory/mps_I inhibitory connections
        i_exc_sel, i_inh_sel = np.zeros((2, 0), dtype=np.int32)
        if has_mps_E:
            i_exc_sel = rng.choice(i_exc, size=min(mps_E, n_exc), replace=False)
        if has_mps_I:
            i_inh_sel = rng.choice(i_inh, size=min(mps_I, n_inh), replace=False)

        # We're done if both has_mps_E and has_mps_I are true. Otherwise, select
        # more neurons up to mps.
        if not (has_mps_E and has_mps_I):
            # If no maximum number of synapses is set, set it to the maximum
            # number of synapses that are still available
            if not has_mps:
                mps_rem = n_exc + n_inh
            else:
                mps_rem = mps
            mps_rem = max(0, mps_rem - i_exc_sel.size - i_inh_sel.size)

            if has_mps_E: # Need to select inhibitory neurons
                i_inh_sel = rng.choice(
                    i_inh, size=min(mps_rem, n_inh), replace=False)
            elif has_mps_I: # Need to select excitatory neurons
                i_exc_sel = rng.choice(
                    i_exc, size=min(mps_rem, n_exc), replace=False)
            else: # Need to select both excitatory and inhibitory neurons
                idcs = rng.choice(
                    np.arange(0, n_inh + n_exc, dtype=np.int32),
                    size=mps_rem, replace=False)
                i_exc_sel = i_exc[idcs[idcs < n_exc]]
                i_inh_sel = i_inh[idcs[idcs >= n_exc] - n_exc]

        # Set the corresponding entries in the connectivity matrix to true
        connectivity[0, i_exc_sel, i_post] = True
        connectivity[1, i_inh_sel, i_post] = True
    return connectivity

def remove_bias_current(model, ens):
    if not 'bias' in model.sig[ens.neurons]:
        return

    sig_post_bias = model.sig[ens.neurons]['bias']
    sig_post_in = model.sig[ens.neurons]['in']
    for i, op in enumerate(model.operators):
        if isinstance(op, Copy):
            if (op.src is sig_post_bias) and (op.dst is sig_post_in):
                # Delete the copy operator and instead add a reset operator
                del model.operators[i]
                model.add_op((Reset(sig_post_in)))


@nengo.builder.Builder.register(SolverWrapper)
def build_solver(model, solver, _, rng, *args, **kwargs):
    # Fetch the high-level connection
    conn = solver.connection # Note: this is the nengo_bio.Connection object
                             # and NOT the nengo.Connection object

    # If the high-level connection object has not been built, build it
    if not conn in model.params:
        ### TODO: Move to build_connection
        model.params[conn] = built_connection = BuiltConnection()

        # Remove the bias current from the target ensemble
        if conn.decode_bias:
            remove_bias_current(model, conn.post_obj)
        elif isinstance(conn.post_obj.neuron_type, MultiChannelNeuronType):
            raise BuildError(
                "decode_bias=False on connection {} invalid for post objects "
                "with multi-channel neurons.".format(conn))

        # Fetch the evaluation points, activites, and synapse types for the
        # entire MultiEnsemble
        eval_points = get_eval_points(
            model, conn, rng)
        activities = get_multi_ensemble_activities(
            model, conn.pre_obj, eval_points)
        synapse_types = get_multi_ensemble_synapse_types(
            model, conn.pre_obj)

        # Fetch the target values in representation space
        targets = nengo.builder.connection.get_targets(conn, eval_points)

        # Transform the target values
        if hasattr(nengo.connection, 'Dense'): # Nengo 2.8 compat
            transform = conn.transform.sample(rng=rng)
        else:
            transform = conn.transform
        targets = np.dot(targets, transform.T)

        # For the target population, fetch the gains and biases
        built_post_ens = model.params[conn.post_obj]
        encoders = built_post_ens.encoders / conn.post.radius
        gain = built_post_ens.gain
        bias = built_post_ens.bias

        # Compute the target currents
        target_currents = (targets @ encoders.T) * gain
        if conn.decode_bias:
            target_currents += bias

        # Construct the connectivity matrix for this connection
        connectivity = get_connectivity(conn, synapse_types, rng)

        # LIF neuron model parameters
        tuning = None
        if hasattr(built_post_ens, 'tuning'):
            tuning = built_post_ens.tuning

        i_th = 1.0
        if hasattr(conn.post_obj.neuron_type, 'threshold_current'):
            i_th = conn.post_obj.neuron_type.threshold_current
        WE, WI = solver(activities, target_currents, connectivity, i_th, tuning, rng)

        # If we're not targeting a MultiChannelNeuronType there really isn't
        # a distinction between excitatory and inhibitory input weights. Hence,
        # we must negate the inhibitory weights
        if not isinstance(conn.post_obj.neuron_type, MultiChannelNeuronType):
            WI = -WI

        built_connection.weights[Excitatory] = WE
        built_connection.weights[Inhibitory] = WI
    else:
        built_connection = model.params[conn]

    W = np.copy(
        built_connection.weights[solver.synapse_type][solver.neuron_indices].T)

    return None, W, None

@nengo.builder.Builder.register(ConnectionPart)
def build_connection(model, conn):
    # Run the original build_connection. This will trigger the above
    # build_solver function
    nengo.builder.connection.build_connection(model, conn)

    # Fetch the ensemble the connection part is connected to. Abort if the
    # target is not an ensemble
    if isinstance(conn.post_obj, Neurons):
        ens = conn.post_obj.ensemble
    elif isinstance(conn.post_obj, Ensemble):
        ens = conn.post_obj
    else:
        return

    # Nothing to do if the post neuron type is not a MultiChannelNeuronType
    if not isinstance(ens.neuron_type, MultiChannelNeuronType):
        return

    # Make sure the connection is actually connection to the "neurons" instance
    # and not the ensembles directly
    assert model.sig[conn]['out'] == model.sig[conn.post_obj.neurons]['in']

    # Determine which channel to connect to
    channel_idx = ens.neuron_type.inputs.index(conn.synapse_type)

    # Fetch the corresponding signal
    dst = model.sig[conn.post_obj.neurons]['in_{}'.format(channel_idx)]

    # Search for the copy operator that connects the connection to the unused
    # "in" signal and delete it. Create a new operator that connects the
    # connection to the right neuron channel.
    for i, op in enumerate(model.operators):
        if isinstance(op, Copy):
            if (op.dst is model.sig[conn.post_obj.neurons]['in']):
                del model.operators[i]
                model.add_op(Copy(
                    op.src, dst, None, None, inc=True,
                    tag="{}.{}".format(conn, conn.kind)))
                break

