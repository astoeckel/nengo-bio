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
from nengo_bio.solvers import SolverWrapper, ExtendedSolver

from nengo.exceptions import NengoWarning

import nengo.builder

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


def get_eval_points(model, conn, rng):
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
            synapse_types[i, ns] = model.params[ens].synapse_types[type_]
    return synapse_types


def remove_bias_current(model, ens):
    sig_post_bias = model.sig[ens.neurons]['bias']
    sig_post_in = model.sig[ens.neurons]['in']
    for i, op in enumerate(model.operators):
        if isinstance(op, nengo.builder.operator.Copy):
            if (op.src is sig_post_bias) and (op.dst is sig_post_in):
                # Delete the copy operator and instead add a reset operator
                del model.operators[i]
                model.add_op((nengo.builder.operator.Reset(sig_post_in)))
                return True
    return False


@nengo.builder.Builder.register(SolverWrapper)
def build_solver(model, solver, _, rng):
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
        if not isinstance(conn.transform, nengo.connection.Dense):
            raise nengo.exceptions.BuildError(
                "Non-compositional solvers only work with Dense transforms")
        transform = conn.transform.sample(rng=rng)
        targets = np.dot(targets, transform.T)

        # For the target population, fetch the gains and biases
        built_post_ens = model.params[conn.post_obj]
        encoders = built_post_ens.encoders
        gain = built_post_ens.gain
        bias = built_post_ens.bias

        # Compute the target currents
        target_currents = (targets @ encoders.T) * gain
        if conn.decode_bias:
            target_currents += bias

        # LIF neuron model parameters
        WE, WI = solver(activities, target_currents, synapse_types, rng)

#        RMS = np.sqrt(np.mean(np.square(target_currents)))
#        RMSE = np.sqrt(np.mean(np.square(target_currents -
#               (activities @ WE - activities @ WI))))
#        print(conn.label, RMS, RMSE / RMS, np.mean(WE + WI))

        built_connection.weights[Excitatory] =  WE
        built_connection.weights[Inhibitory] = -WI
    else:
        built_connection = model.params[conn]

    W = np.copy(
        built_connection.weights[solver.synapse_type][solver.neuron_indices].T)

    return None, W, None
