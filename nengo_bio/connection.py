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

import numpy as np

from .solvers import Excitatory, Inhibitory, QPSolver

import nengo.base
import nengo.config
import nengo.connection
import nengo.exceptions
import nengo.params
import nengo.synapses
import nengo.builder

class PreParam(nengo.params.Parameter):
    """
    The PreParam class is used by arbour.Connection to describe the list of
    pre-objects that are involved in a certain connection.
    """

    def __init__(self, name):
        super().__init__(name, default=nengo.params.Unconfigurable,
                         optional=False, readonly=True)

    def coerce(self, instance, nengo_obj):
        # List of supported objects
        SUPPORTED_NENGO_OBJS = (
            nengo.base.NengoObject,
            nengo.base.ObjView,
            nengo.ensemble.Neurons,
        )

        # Make sure nengo_obj is a list or tuple
        if not isinstance(nengo_obj, (tuple, list)):
            nengo_obj = (nengo_obj,)

        # For each object, check whether it is in the list of supported nengo
        # objects
        for obj in nengo_obj:
            if not isinstance(obj, SUPPORTED_NENGO_OBJS):
                raise ValidationError("'{}' is not a Nengo object".format(obj),
                                      attr=self.name, obj=instance)
            if obj.size_in < 1:
                raise ValidationError("'{}' must have size_in > 0.".format(obj),
                                      attr=self.name, obj=instance)
        return super().coerce(instance, nengo_obj)

class ConnectionFunctionParam(nengo.connection.ConnectionFunctionParam):
    """Connection-specific validation for functions."""

    coerce_defaults = False

    def check_function_can_be_applied(self, conn, function_info):
        function, size = function_info
        type_pre = type(conn.pre_obj).__name__

        if function is not None:
            for pre_ in conn.pre_obj:
                if not isinstance(pre_, nengo.ensemble.Ensemble):
                    raise ValidationError(
                        "function can only be set for connections from an Ensemble"
                        "(got type %r)" % type_pre,
                        attr=self.name, obj=conn)

class Connection(nengo.config.SupportDefaultsMixin):

    label = nengo.params.StringParam(
        'label', default=None, optional=True)
    seed = nengo.params.IntParam(
        'seed', default=None, optional=True)

    pre = PreParam('pre')
    post = nengo.connection.PrePostParam('post', nonzero_size_in=True)

    synapse_exc = nengo.connection.SynapseParam(
        'synapse_exc', default=nengo.synapses.Lowpass(tau=0.005))
    synapse_inh = nengo.connection.SynapseParam(
        'synapse_inh', default=nengo.synapses.Lowpass(tau=0.005))

    function_info = ConnectionFunctionParam(
        'function', default=None, optional=True)

    transform = nengo.connection.ConnectionTransformParam(
        'transform', default=1.0)

    eval_points = nengo.connection.EvalPointsParam(
        'eval_points', default=None, optional=True, 
        sample_shape=('*', 'size_in'))
    scale_eval_points = nengo.params.BoolParam(
        'scale_eval_points', default=True)

    _param_init_order = [
        'pre', 'post', 'synapse_exc', 'synapse_inh', 'function_info'
    ]

    def __init__(self, pre, post,
                 synapse_exc=nengo.params.Default,
                 synapse_inh=nengo.params.Default,
                 function=nengo.params.Default,
                 transform=nengo.params.Default,
                 eval_points=nengo.params.Default,
                 scale_eval_points=nengo.params.Default,
                 label=nengo.params.Default,
                 seed=nengo.params.Default):
        super().__init__()

        # Copy the parameters
        self.label = label
        self.seed = seed
        self.pre = pre
        self.post = post
        self.synapse_exc = synapse_exc
        self.synapse_inh = synapse_inh
        self.function_info = function
        self.eval_points = eval_points
        self.scale_eval_points = scale_eval_points
        self.transform = transform

        # For each pre object add two actual nengo connections: an excitatory
        # path and an inhibitory path
        self.connections = []
        for i, pre_ in enumerate(self.pre):
            def build_connection(weight_type, synapse):
                return nengo.connection.Connection(
                    pre=pre_,
                    post=self.post,
                    transform=np.zeros((self.post.size_in, pre_.size_out)),
                    seed=self.seed,
                    synapse=synapse,
                    solver=QPSolver(
                        pre=self.pre,
                        pre_idx=i,
                        post=self.post,
                        connection=self,
                        weight_type=weight_type
                    ))
            self.connections.append((
                build_connection(Excitatory, synapse_exc),
                build_connection(Inhibitory, synapse_inh)))


    def __str__(self):
        return "<ArbourConnection {}>".format(self._str)

    def __repr__(self):
        return "<ArbourConnection at {:06X} {}>".format(id(self), self._str)

    @property
    def _str(self):
        if self.label is not None:
            return self.label

        desc = "" if self.function is None else " computing '{}'".format(
            function_name(self.function))
        return "from {} to {}{}".format(self.pre, self.post, desc)

    @property
    def function(self):
        return self.function_info.function

    @function.setter
    def function(self, function):
        self.function_info = function

    @property
    def is_decoded(self):
        return not (self.solver.weights or (
            isinstance(self.pre_obj, Neurons)
            and isinstance(self.post_obj, Neurons)))

    @property
    def _label(self):
        if self.label is not None:
            return self.label

        return "from %s to %s%s" % (
            self.pre, self.post,
            " computing '%s'" % function_name(self.function)
            if self.function is not None else "")

    @property
    def post_obj(self):
        return self.post.obj if isinstance(self.post, nengo.base.ObjView) else self.post

    @property
    def pre_obj(self):
        return self.pre.obj if isinstance(self.pre, nengo.base.ObjView) else self.pre

    @property
    def pre_slice(self):
        return slice(None)

    @property
    def post_slice(self):
        return slice(None)

    @property
    def size_in(self):
        return sum(map(lambda x: x.size_out, self.pre))

    @property
    def size_mid(self):
        size = self.function_info.size
        return self.size_in if size is None else size

    @property
    def size_out(self):
        return self.post.size_in


class BuiltConnection:

    def __init__(self):
        self.weights = {
            Excitatory: None,
            Inhibitory: None
        }
        self.pre_idx_dim_map = []
        self.pre_idx_neurons_map = []

def remove_bias_current(model, ens):
    sig_post_bias = model.sig[ens.neurons]['bias']
    sig_post_in = model.sig[ens.neurons]['in']
    for i, op in enumerate(model.operators):
        if isinstance(op, nengo.builder.operator.Copy):
            if (op.src is sig_post_bias) and (op.dst is sig_post_in):
                del model.operators[i]
                return True
    return False

@nengo.builder.Builder.register(QPSolver)
def build_solver(model, solver, _, rng):
    # Fetch the high-level connection
    conn = solver.connection

    if not conn in model.params:
        model.params[conn] = built_connection = BuiltConnection()

        # Remove the bias current from the target ensemble
        #remove_bias_current(model, conn.post_obj)

        # For each pre-ensemble, fetch the evaluation points and the activities
        d0, d1, n0, n1 = 0, 0, 0, 0

        N = len(conn.pre)
        eval_points_list = [None] * N
        activities_list = [None] * N
        pre_idx_dim_map = [(0, 0)] * N
        pre_idx_neurons_map = [(0, 0)] * N

        for pre_idx, pre_ in enumerate(conn.pre):
            d0, d1, n0, n1 = d1, d1 + pre_.size_out, n1, n1 + pre_.neurons.size_out
            if conn.eval_points is None:
                eval_points = model.params[pre_].eval_points.view()
                eval_points.setflags(write=False)
            else:
                eval_points = conn.eval_points[d0:d1]

            activities = nengo.builder.ensemble.get_activities(
                model.params[pre_], pre_, eval_points)

            eval_points_list[pre_idx] = eval_points
            activities_list[pre_idx] = activities
            pre_idx_dim_map[pre_idx] = (d0, d1)
            pre_idx_neurons_map[pre_idx] = (n0, n1)

        # Make sure each pre-population has the same number of evaluation points
        if len(set(map(lambda x: x.shape[0], eval_points))) > 1:
            raise nengo.exceptions.BuildError(
                "The number of evaluation points must be the same for all " +
                "pre-objects in connection {}".format(conn))

        # Build the evaluation points and activities encompassing all source
        # ensembles
        eval_points = np.concatenate(eval_points_list, axis=1)
        activities = np.concatenate(activities_list, axis=1)

        # Fetch the target values in representation space
        targets = nengo.builder.connection.get_targets(conn, eval_points)

        # Transform the target values
        if not isinstance(conn.transform, nengo.connection.Dense):
            raise nengo.exceptions.BuildError(
                "Non-compositional solvers only work with Dense transforms")
        transform = conn.transform.sample(rng=rng)
        targets = np.dot(targets, transform.T)

        # For the target population, fetch the gains and biases
        encoders = model.params[conn.post_obj].encoders
        gain = model.params[conn.post_obj].gain
        bias = model.params[conn.post_obj].bias

        # Compute the target currents (gains are rolled into the encoders)
        target_current = (targets @ encoders.T) * gain #+ bias

        sigma = (0.1 * np.max(activities))
        A = activities.T @ activities + np.eye(activities.shape[1]) * (sigma**2)
        b = activities.T @ target_current
        weights = np.linalg.lstsq(A, b, rcond=None)[0]

#        J = target_current[:, 0]
#        RMS = np.sqrt(np.mean((target_current) ** 2))
#        RMSE = np.sqrt(np.mean((J - (activities @ weights)[:, 0]) ** 2))
#        print(RMSE / RMS, RMSE, RMS)

        built_connection.weights[Excitatory] = np.clip(weights, 0, None)
        built_connection.weights[Inhibitory] = np.clip(weights, None, 0)
        built_connection.pre_idx_dim_map = pre_idx_dim_map
        built_connection.pre_idx_neurons_map = pre_idx_neurons_map
    else:
        built_connection = model.params[conn]

    n_neurons_pre = solver.pre[solver.pre_idx].neurons.size_out
    n_neurons_post = solver.post.neurons.size_in

    bc = built_connection
    n0, n1 = bc.pre_idx_neurons_map[solver.pre_idx]
    W = np.copy(bc.weights[solver.weight_type][n0:n1].T)

    return None, W, None
