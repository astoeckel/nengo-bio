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

from nengo_bio.common import steal_param, Excitatory, Inhibitory
from nengo_bio.solvers import SolverWrapper, QPSolver

from nengo.params import Parameter, BoolParam, IntParam, StringParam, \
                         Default, Unconfigurable

import nengo.base
import nengo.config
import nengo.connection
import nengo.dists
import nengo.exceptions
import nengo.synapses
import nengo.builder

class ConnectionPart(nengo.connection.Connection):

    dales_principle = True

    kind = StringParam('kind', default=None, optional=True)

    def __init__(self, *args, **kw_args):
        self.kind = steal_param('kind', kw_args, Default)
        super(ConnectionPart, self).__init__(*args, **kw_args)

class MultiEnsemble(nengo.base.SupportDefaultsMixin):
    """
    The MultiEnsemble class represents a view at a list of/set of ensembles.
    Given a descriptor `descr` the represented value of the ensembles is either
    stacked, or joined. "Stack" operations are denoted as a tuple

        (ens_a, ens_b)

    where ens_a is a 1D ensemble and ens_b is a 2D ensemble will result in a 3D
    multi ensemble. "Join" operations are denoted as a set:

        {ens_a, ens_b}

    Here, both ens_a and ens_b have to have the same dimensionality -- the
    resulting multi-ensemble will provide a view on ens_a and ens_b where ens_a
    and ens_b represent the same value at the same time.
    """

    OP_NONE = 0
    OP_STACK = 1
    OP_JOIN = 2

    n_neurons = IntParam('n_neurons', low=1, readonly=True)
    dimensions = IntParam('dimensions', low=1, readonly=True)

    def __init__(self, descr, operator=None):
        # Nengo objects that can be base cases for the multi ensembles
        SUPPORTED_NENGO_OBJS = (
            nengo.ensemble.Ensemble
        )

        # Determine the operator type depending on whether the given descriptor
        # is a tuple, set, or just an ensemble
        if isinstance(descr, SUPPORTED_NENGO_OBJS) and \
           ((operator is None) or (operator == OP_NONE)):
            self.operator = MultiEnsemble.OP_NONE
            self.objs = (descr,)
        elif isinstance(descr, tuple) and (operator is None):
            self.operator = MultiEnsemble.OP_STACK
            self.objs = descr
        elif isinstance(descr, set) and (operator is None):
            self.operator = MultiEnsemble.OP_JOIN
            self.objs = descr
        elif isinstance(descr, MultiEnsemble):
            if not (operator is None):
                raise ValueError(
                    "\"operator\" must be None when initialising a "
                    "MultiEnsemble with a MultiEnsemble.")
            self.operator = descr.operator
            self.objs = descr.objs
        elif (operator in [MultiEnsemble.OP_STACK,
                           MultiEnsemble.OP_JOIN]) and hasattr(descr, '__len__'):
            self.operator = operator
            self.objs = tuple(descr)
        else:
            raise ValueError(
                "Pre-object must either be a tuple, set, or an Ensemble")

        # Recursively turn the individual objects into MultiEnsemble objects
        if self.operator != MultiEnsemble.OP_NONE:
            self.objs = tuple(map(MultiEnsemble, self.objs))

        # In case this is a "join" operator, make sure the objects have the same
        # dimensionality
        if self.operator == MultiEnsemble.OP_JOIN:
            if len(set(map(lambda x: x.size_out, self.objs))) > 1:
                raise ValueError(
                    "Ensembles must have the same dimensionality to be joined.")

        # Accumulate the number of dimensions and neurons
        self.dimensions = self._get_accu_dim_attr('dimensions')
        self.n_neurons = sum(map(lambda x: x.n_neurons, self.objs))

    def _get_accu_dim_attr(self, attr):
        """
        Internal function to accumulate a quantity with the name `attr`
        according to whether this ensemble implements a stack or a join
        operation.
        """
        if self.operator == MultiEnsemble.OP_JOIN:
            return 0 if len(self.objs) == 0 else getattr(self.objs[0], attr)
        else:
            return sum(map(lambda x: getattr(x, attr), self.objs))

    def __len__(self):
        return self.dimensions

    def __repr__(self):
        if self.operator == MultiEnsemble.OP_NONE:
            return str(self.objs[0])
        else:
            d0 = '{' if self.operator == MultiEnsemble.OP_JOIN else '('
            d1 = '}' if self.operator == MultiEnsemble.OP_JOIN else ')'
            return "{}{}{}".format(d0, ', '.join(map(str, self.objs)), d1)

    @property
    def size_in(self):
        return self._get_accu_dim_attr("size_in")

    @property
    def size_out(self):
        return self._get_accu_dim_attr("size_out")

    def flatten(self):
        """
        Returns a flat tuple of ensembles, a tuple containing the 1D activity
        slice for each ensemble (assigning a neuron ID to each neuron), and a
        tuple containing the 2D eval_points slice.
        """

        if self.operator == MultiEnsemble.OP_NONE:
            ens = self.objs
            ns = (slice(0, self.n_neurons),) # neuron indices
            ds = (slice(0, self.dimensions),) # dimension indicies
        elif (self.operator == MultiEnsemble.OP_STACK) or\
             (self.operator == MultiEnsemble.OP_JOIN):
            arr_ens, arr_ns, arr_ds = [], [], []
            nn, dn, en = 0, 0, 0
            for i, obj in enumerate(self.objs):
                ens, ns, ds = obj.flatten()

                # Increment the neuron numbers by nn
                ns = tuple(slice(nn + x.start, nn + x.stop) for x in ns)
                nn = ns[-1].stop

                # Increment the dimension/eval points index
                ds = tuple(slice(dn + x.start, dn + x.stop) for x in ds)
                if self.operator == MultiEnsemble.OP_STACK:
                    dn = ds[-1].stop

                # Append the lists to the arrays
                arr_ens.append(ens); arr_ns.append(ns); arr_ds.append(ds)

            # Merge the arrays into a single tuple
            ens, ns, ds = sum(arr_ens, ()), sum(arr_ns, ()), sum(arr_ds, ())
        return ens, ns, ds


class PreParam(Parameter):
    """
    The PreParam class is used by nengo_bio.Connection to describe the list
    of pre-objects that are involved in a certain connection.
    """

    def __init__(self, name):
        super().__init__(name, default=Unconfigurable,
                         optional=False, readonly=True)

    def coerce(self, instance, nengo_obj):
        # Try to convert the given ensemble into a MultiEnsemble
        try:
            obj = MultiEnsemble(nengo_obj)
        except ValueError as e:
            raise nengo.exceptions.ValidationError(
                e.msg, attr=self.name, obj=instance)

        return super().coerce(instance, obj)


class ConnectionFunctionParam(nengo.connection.ConnectionFunctionParam):
    """Connection-specific validation for functions."""

    coerce_defaults = False

    def check_function_can_be_applied(self, conn, function_info):
        function, size = function_info
        type_pre = type(conn.pre_obj).__name__

# Nengo 2.8 compatibility
if hasattr(nengo.connection, 'ConnectionTransformParam'):
    ConnectionTransformParam = nengo.connection.ConnectionTransformParam
else:
    ConnectionTransformParam = nengo.connection.TransformParam


class Connection(nengo.config.SupportDefaultsMixin):

    label = StringParam(
        'label', default=None, optional=True)
    seed = IntParam(
        'seed', default=None, optional=True)

    pre = PreParam(
        'pre')
    post = nengo.connection.PrePostParam(
        'post', nonzero_size_in=True)

    synapse_exc = nengo.connection.SynapseParam(
        'synapse_exc', default=nengo.synapses.Lowpass(tau=0.005))
    synapse_inh = nengo.connection.SynapseParam(
        'synapse_inh', default=nengo.synapses.Lowpass(tau=0.005))

    function_info = ConnectionFunctionParam(
        'function', default=None, optional=True)

    transform = ConnectionTransformParam(
        'transform', default=1.0)

    solver = nengo.solvers.SolverParam(
        'solver', default=QPSolver())
    eval_points = nengo.dists.DistOrArrayParam(
        'eval_points', default=None, optional=True, 
        sample_shape=('*', 'size_in'))
    scale_eval_points = BoolParam(
        'scale_eval_points', default=True)
    n_eval_points = IntParam(
        'n_eval_points', default=None, optional=True)
    decode_bias = BoolParam(
        'decode_bias', default=True)

    max_n_post_synapses = IntParam(
        'max_n_post_synapses', low=0, default=None, optional=True)
    max_n_post_synapses_exc = IntParam(
        'max_n_post_synapses_exc', low=0, default=None, optional=True)
    max_n_post_synapses_inh = IntParam(
        'max_n_post_synapses_inh', low=0, default=None, optional=True)

    _param_init_order = [
        'pre', 'post', 'synapse_exc', 'synapse_inh', 'function_info'
    ]

    def __init__(self, pre, post,
                 synapse_exc=Default,
                 synapse_inh=Default,
                 function=Default,
                 transform=Default,
                 solver=Default,
                 eval_points=Default,
                 scale_eval_points=Default,
                 n_eval_points=Default,
                 decode_bias=Default,
                 max_n_post_synapses=Default,
                 max_n_post_synapses_exc=Default,
                 max_n_post_synapses_inh=Default,
                 label=Default,
                 seed=Default):
        super().__init__()

        # Copy the parameters
        self.label = label
        self.seed = seed

        self.pre = pre
        self.post = post
        self.synapse_exc = synapse_exc
        self.synapse_inh = synapse_inh
        self.eval_points = eval_points
        self.scale_eval_points = scale_eval_points
        self.n_eval_points = n_eval_points
        self.decode_bias = decode_bias
        self.function_info = function
        self.transform = transform
        self.solver = solver

        self.max_n_post_synapses = max_n_post_synapses
        self.max_n_post_synapses_exc = max_n_post_synapses_exc
        self.max_n_post_synapses_inh = max_n_post_synapses_inh

        # For each pre object add two actual nengo connections: an excitatory
        # path and an inhibitory path
        self.connections = []
        arr_ens, arr_ns, _ = self.pre.flatten()
        for i, (ens, ns) in enumerate(zip(arr_ens, arr_ns)):
            def mkcon(synapse_type, synapse):
                return ConnectionPart(
                    pre=ens,
                    post=self.post,
                    transform=np.zeros((self.post.size_in, ens.size_out)),
                    seed=self.seed,
                    synapse=synapse,
                    solver=SolverWrapper(
                        self.solver, i, self, ns, synapse_type),
                    kind=str(synapse_type))
            self.connections.append((
                mkcon(Excitatory, synapse_exc),
                mkcon(Inhibitory, synapse_inh)))

    def __str__(self):
        return "<Connection {}>".format(self._str)

    def __repr__(self):
        return "<Connection at {:06X} {}>".format(id(self), self._str)

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
        return self.post

    @property
    def pre_obj(self):
        return self.pre

    @property
    def pre_slice(self):
        return slice(None)

    @property
    def post_slice(self):
        return slice(None)

    @property
    def size_in(self):
        return self.pre.size_out

    @property
    def size_mid(self):
        size = self.function_info.size
        return self.size_in if size is None else size

    @property
    def size_out(self):
        return self.post.size_in

