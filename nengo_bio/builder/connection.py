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
import warnings
import numpy as np

from nengo_bio.common import \
    Excitatory, \
    Inhibitory, \
    Decode, \
    JBias, \
    ExcJBias, \
    InhJBias
from nengo_bio.connection import \
    ConnectionPart, \
    Connection, \
    Connectivity, \
    UnconstrainedConnectivity, \
    ConstrainedConnectivity, \
    DefaultConnectivity
from nengo_bio.solvers import SolverWrapper, ExtendedSolver
from nengo_bio.neurons import MultiChannelNeuronType

from nengo.exceptions import NengoWarning, BuildError
from nengo.ensemble import Ensemble, Neurons
from nengo.utils.numpy import is_array_like

import nengo.builder
from nengo.builder.operator import Copy, Reset

built_attrs = ["weights", "connectivity"]


class BuiltConnection(collections.namedtuple('BuiltConnection', built_attrs)):
    pass


def get_multi_ensemble_eval_points(model,
                                   mens,
                                   n_eval_points=None,
                                   scale_eval_points=True,
                                   rng=np.random):
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
            pnts = get_multi_ensemble_eval_points(model, mens.objs[i],
                                                  n_eval_points,
                                                  scale_eval_points, rng)
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
                    pnts[:n_pnts, d:(d + n_dims)] = p
                    pnts[n_pnts:, d:(d+n_dims)] = \
                        choice(p, n_eval_points - n_pnts)
                else:
                    # Ranomly select n_eval_points points and write them to
                    # the target array
                    pnts[:, d:(d + n_dims)] = choice(p, n_eval_points)

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
        return get_multi_ensemble_eval_points(model, conn.pre_obj,
                                              conn.n_eval_points,
                                              conn.scale_eval_points, rng)
    else:
        if conn.scale_eval_points:
            warnings.warn(
                NengoWarning(
                    "Setting scale_eval_points to True has no effect on {} if "
                    "eval_points are specified manually.".format(
                        conn.pre_obj)))
        return nengo.builder.ensemble.gen_eval_points(conn, conn.eval_points,
                                                      rng, False)


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


def get_connection_matrix(model, conn, synapse_types, rng=np.random):
    # Create the empty synapse type map
    pre_obj, post_obj = conn.pre_obj, conn.post_obj
    n_pre, n_post = pre_obj.n_neurons, post_obj.n_neurons

    # Create the connectivity matrix
    connectivity = np.empty((2, n_pre, n_post), dtype=np.bool)

    # Iterate over all pre-ensembles in the multi-ensemble
    arr_ens, arr_ns, _ = conn.pre_obj.flatten()
    connectivity_cache = {}
    for ens, ns in zip(arr_ens, arr_ns):
        # If the connectivity object is a dictionary, check whether an entry for
        # exactly this connection exists
        connectivity_descr = None
        connectivity_key = (ens, post_obj)
        if isinstance(conn.connectivity, dict):
            if connectivity_key in conn.connectivity:
                connectivity_descr = conn.connectivity[connectivity_key]
        elif isinstance(conn.connectivity, Connectivity):
            connectivity_descr = conn.connectivity

        # The connectivity descriptor should default to default_connectivity
        if connectivity_descr is None:
            connectivity_descr = DefaultConnectivity()

        # Build the connectivity matrix according to the connectivity
        # descriptor. Use already computed connectivity from the connectivity
        # cache if the same neurons are connected multiple times.
        if connectivity_key in connectivity_cache:
            C = connectivity_cache[connectivity_key]
        else:
            C = model.build(connectivity_descr, ens, post_obj, rng)
            connectivity_cache[connectivity_key] = C

        # If the connectivity descriptor returns "None", just build all-to-all
        # connections
        if C is None:
            connectivity[:, ns, :] = np.ones((2, ens.n_neurons, n_post))
        else:
            connectivity[:, ns, :] = C

    # Issue warnings for unused elements in the connectivity dictionary
    if isinstance(conn.connectivity, dict):
        for key in conn.connectivity.keys():
            if not key in connectivity_cache:
                warnings.warn(
                    NengoWarning(
                        "Connectivity descriptor {} not used".format(key)))

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


def override_intrinsic_bias(model, ens, tar):
    if not 'bias' in model.sig[ens.neurons]:
        return

    sig_post_bias = model.sig[ens.neurons]['bias']
    sig_post_bias._initial_value.setflags(write=True)
    sig_post_bias._initial_value[...] = tar
    sig_post_bias._initial_value.setflags(write=False)


@nengo.builder.Builder.register(UnconstrainedConnectivity)
def build_unconstrained_connectivity(model, cty, pre_obj, post_obj, rng):
    # Put no constraints on the connections whatsoever
    return None


@nengo.builder.Builder.register(DefaultConnectivity)
def build_default_connectivity(model, cty, pre_obj, post_obj, rng):
    # Fetch the number of pre- and post neurons
    n_pre, n_post = pre_obj.n_neurons, post_obj.n_neurons

    # Fetch the constructed pre object
    built_pre_obj = model.params[pre_obj]
    connectivity = np.zeros((2, n_pre, n_post), dtype=np.bool)
    for i, type_ in enumerate((Excitatory, Inhibitory)):
        if hasattr(built_pre_obj, "synapse_types"):
            connectivity[i] = built_pre_obj.synapse_types[type_][:, None]
        else:
            connectivity[i] = np.ones((n_pre, n_post), dtype=np.bool)

    return connectivity


@nengo.builder.Builder.register(ConstrainedConnectivity)
def build_constrained_connectivity(model, cty, pre_obj, post_obj, rng):
    # Create some convenient aliases
    n_cov, n_div = cty.convergence, cty.divergence
    n_pre, n_post = pre_obj.n_neurons, post_obj.n_neurons

    # Create the default connectivity matrix
    cs = build_default_connectivity(model, cty,
                                    pre_obj, post_obj, rng)

    # Fetch the connection probabilities
    ps = None
    if not cty.probabilities is None:
        if callable(cty.probabilities):
            ps = cty.probabilities(n_pre, n_post, pre_obj, post_obj, model.params)
        elif is_array_like(cty.probabilities):
            ps = cty.probabilities

        if (not ps is None) and ((ps.ndim != 2) or (ps.shape[0] != n_pre) or (ps.shape[1] != n_post)):
            raise ValueError(
                "Invalid connection probability matrix. Expected matrix of "
                "shape {} x {} (n_pre x n_post)".format(n_pre, n_post))

    # Helper function used for restricting both the convergence and divergence
    def apply_constraints(n, cs, ps):
        if not ps is None:
            assert cs.shape[1] == ps.shape[0]
            assert cs.shape[2] == ps.shape[1]

        for i in range(cs.shape[1]):
            # Compute the pre-neuron indices that are available at all
            idcs_syn, idcs = np.where(cs[:, i, :] != 0)
            n_available = min(n, idcs.size)

            # If the "p" matrix is given, compute the probabilities for any of
            # these synapses to be connected
            probabilities = None
            if not ps is None:
                probabilities = ps[i, idcs] / np.sum(ps[i, idcs])

            # Select the pre-synapses
            sel = np.random.choice(np.arange(idcs.size, dtype=np.int),
                                   size=n_available,
                                   replace=False,
                                   p=probabilities)
            cs[:, i, :] = False
            cs[idcs_syn[sel], i, idcs[sel]] = True

    # Restrict the convergence numbers
    if not n_cov is None:
        apply_constraints(n_cov,
                          np.transpose(cs, (0, 2, 1)),
                          None if ps is None else ps.T)

    # Restrict the divergence numbers
    if not n_div is None:
        apply_constraints(n_div, cs, ps)

    return cs

@nengo.builder.Builder.register(SolverWrapper)
def build_solver(model, solver, _, rng, *args, **kwargs):
    # Fetch the high-level connection
    conn = solver.connection  # Note: this is the nengo_bio.Connection object
    # and NOT the nengo.Connection object

    # If the high-level connection object has not been built, build it
    if not conn in model.params:
        # For the target population, fetch the gains and biases
        built_post_ens = model.params[conn.post_obj]
        encoders = built_post_ens.encoders / conn.post.radius
        gain = built_post_ens.gain
        bias = built_post_ens.bias
        intrinsic_bias = np.copy(bias)

        # Remove the bias current from the target ensemble, if the bias mode
        # is set to "Decode". This is the only valid setting for multi-channel
        # post-neurons.
        if conn.bias_mode is Decode:
            remove_bias_current(model, conn.post_obj)
            intrinsic_bias = np.zeros_like(intrinsic_bias)
        elif isinstance(conn.post_obj.neuron_type, MultiChannelNeuronType):
            raise BuildError(
                "bias_mode != Decode on connection {} invalid for post objects "
                "with multi-channel neurons.".format(conn))

        # Modify the bias currents if the bias mode is ExcJBias or InhJBias
        if conn.bias_mode is ExcJBias:
            intrinsic_bias = np.maximum(0, intrinsic_bias)
            override_intrinsic_bias(model, conn.post_obj, intrinsic_bias)
        elif conn.bias_mode is InhJBias:
            intrinsic_bias = np.minimum(0, intrinsic_bias)
            override_intrinsic_bias(model, conn.post_obj, intrinsic_bias)

        # Fetch the evaluation points, activites, and synapse types for the
        # entire MultiEnsemble
        eval_points = get_eval_points(model, conn, rng)
        activities = get_multi_ensemble_activities(model, conn.pre_obj,
                                                   eval_points)
        synapse_types = get_multi_ensemble_synapse_types(model, conn.pre_obj)

        # Fetch the target values in representation space
        targets = nengo.builder.connection.get_targets(conn, eval_points)

        # Transform the target values
        if hasattr(nengo.connection, 'Dense'):  # Nengo 2.8 compat
            transform = conn.transform.sample(rng=rng)
        else:
            transform = conn.transform
        targets = np.dot(targets, transform.T)

        # Compute the target currents. We need to add the difference between the
        # desired target currents, and the bias currents intrinsic to each
        # neuron
        target_currents = (targets @ encoders.T) * gain
        target_currents += bias - intrinsic_bias

        # Construct the connection matrix for this connection
        connectivity = get_connection_matrix(
            model, conn, synapse_types, rng)

        # LIF neuron model parameters
        tuning = None
        if hasattr(built_post_ens, 'tuning'):
            tuning = built_post_ens.tuning

        i_th = 0.0
        if hasattr(conn.post_obj.neuron_type, 'threshold_current'):
            i_th = conn.post_obj.neuron_type.threshold_current
        WE, WI = solver(activities, target_currents, connectivity, i_th,
                        tuning, rng)

        # If we're not targeting a MultiChannelNeuronType there really isn't
        # a distinction between excitatory and inhibitory input weights. Hence,
        # we must negate the inhibitory weights
        if not isinstance(conn.post_obj.neuron_type, MultiChannelNeuronType):
            WI = -WI

        built_connection = model.params[conn] = BuiltConnection({
            Excitatory: WE, Inhibitory: WI
        }, connectivity)
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
                model.add_op(
                    Copy(op.src,
                         dst,
                         None,
                         None,
                         inc=True,
                         tag="{}.{}".format(conn, conn.kind)))
                break
