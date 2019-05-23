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

from .common import Excitatory, Inhibitory
from .solvers import ExtendedSolver, QPSolver

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

    class SolverWrapper(ExtendedSolver):
        # The SolverWrapper class is used to append more information to t

        def __init__(self, solver, pre_idx, connection, synapse_type):
            super().__init__()
            self.solver = solver
            self.pre_idx = pre_idx
            self.connection = connection
            self.synapse_type = synapse_type

        def __call__(self, A, J, neuron_types, rng=np.random):
            return self.solver(A, J, neuron_types, rng)

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

    solver = nengo.solvers.SolverParam('solver', default=QPSolver())

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
                 solver=nengo.params.Default,
                 eval_points=nengo.params.Default,
                 scale_eval_points=nengo.params.Default,
                 label=nengo.params.Default,
                 seed=nengo.params.Default):
        super().__init__()

        # Copy the parameters
        self.label = label
        self.seed = seed
        self.synapse_exc = synapse_exc
        self.synapse_inh = synapse_inh
        self.eval_points = eval_points
        self.scale_eval_points = scale_eval_points
        self.pre = pre
        self.post = post
        self.function_info = function
        self.transform = transform
        self.solver = solver

        # For each pre object add two actual nengo connections: an excitatory
        # path and an inhibitory path
        self.connections = []
        for i, pre_ in enumerate(self.pre):
            def mkcon(synapse_type, synapse):
                return nengo.connection.Connection(
                    pre=pre_,
                    post=self.post,
                    transform=np.zeros((self.post.size_in, pre_.size_out)),
                    seed=self.seed,
                    synapse=synapse,
                    solver=Connection.SolverWrapper(self.solver, i, self, synapse_type))
            self.connections.append((
                mkcon(Excitatory, synapse_exc),
                mkcon(Inhibitory, synapse_inh)))

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

