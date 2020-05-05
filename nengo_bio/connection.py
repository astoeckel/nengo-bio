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
from nengo_bio.internal.sequences import hilbert_curve

from nengo.utils.numpy import is_array_like
from nengo.params import Parameter, BoolParam, IntParam, NumberParam, \
                         StringParam, NdarrayParam, Default, Unconfigurable, \
                         FrozenObject

import nengo.base
import nengo.config
import nengo.connection
import nengo.dists
import nengo.exceptions
import nengo.synapses
import nengo.builder


class ConnectionPart(nengo.connection.Connection):

    dales_principle = True

    def __init__(self, *args, **kw_args):
        self._synapse_type = steal_param('synapse_type', kw_args, Default)
        super(ConnectionPart, self).__init__(*args, **kw_args)

    @property
    def synapse_type(self):
        return self._synapse_type

    @property
    def kind(self):
        return str(self._synapse_type)


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
        SUPPORTED_NENGO_OBJS = (nengo.ensemble.Ensemble)

        # Determine the operator type depending on whether the given descriptor
        # is a tuple, set, or just an ensemble
        if isinstance(descr, SUPPORTED_NENGO_OBJS) and \
           ((operator is None) or (operator == MultiEnsemble.OP_NONE)):
            self.operator = MultiEnsemble.OP_NONE
            self.objs = (descr, )
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
        elif (operator in [MultiEnsemble.OP_STACK, MultiEnsemble.OP_JOIN
                           ]) and hasattr(descr, '__len__'):
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
                    "Ensembles must have the same dimensionality to be joined."
                )

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
            ns = (slice(0, self.n_neurons), )  # neuron indices
            ds = (slice(0, self.dimensions), )  # dimension indicies
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
                arr_ens.append(ens)
                arr_ns.append(ns)
                arr_ds.append(ds)

            # Merge the arrays into a single tuple
            ens, ns, ds = sum(arr_ens, ()), sum(arr_ns, ()), sum(arr_ds, ())
        return ens, ns, ds


class PreParam(Parameter):
    """
    The PreParam class is used by nengo_bio.Connection to describe the list
    of pre-objects that are involved in a certain connection.
    """
    def __init__(self, name):
        super().__init__(name,
                         default=Unconfigurable,
                         optional=False,
                         readonly=True)

    def coerce(self, instance, nengo_obj):
        # Try to convert the given ensemble into a MultiEnsemble
        try:
            obj = MultiEnsemble(nengo_obj)
        except ValueError as e:
            raise nengo.exceptions.ValidationError(e.msg,
                                                   attr=self.name,
                                                   obj=instance)

        return super().coerce(instance, obj)


class ConnectionFunctionParam(nengo.connection.ConnectionFunctionParam):
    """Connection-specific validation for functions."""

    coerce_defaults = False

    def check_function_can_be_applied(self, conn, function_info):
        function, size = function_info
        type_pre = type(conn.pre_obj).__name__


class ConnectivityProbabilitiesParam(Parameter):
    """
    The ConnectivityProbabilitiesParam class describes the data that can be
    passed to the "probabilities" property of the ConstrainedConnectivity class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def coerce(self, instance, obj):
        # Do nothing if no object is given -- that's fine
        if obj is None:
            return None

        # Make sure obj is either callable or an array
        if (not callable(obj)) and (not is_array_like(obj)):
            raise nengo.exceptions.ValidationError(
                "connectivity must either be a Npost x Npre array or a callable",
                attr=self.name,
                obj=instance)

        # Convert the given array to the right format
        if is_array_like(obj):
            obj = np.array(obj, copy=False, dtype=np.float64)
            if obj.ndim != 2:
                raise nengo.exceptions.ValidationError(
                    "connectivity must be a 2D array",
                    attr=self.name,
                    obj=instance)

        return obj


class Connectivity(FrozenObject):
    """
    Base class for all connectivity implementations. Deriving classes must
    implement the "__call__" method.
    """
    @property
    def _argreprs(self):
        return []


class UnconstrainedConnectivity(Connectivity):
    """
    Use of this class explicitly indicates that there are no constraints on the
    connectivity, including Dale's principle being ignored.
    """
    pass


class DefaultConnectivity(Connectivity):
    """
    Connectivity class taking Dale's principle into account. This is the
    default.
    """
    pass


class ConstrainedConnectivity(Connectivity):
    """
    This class can be used to specify convergence and divergence parameters for
    a connection. Furthermore, an optional probability matrix or callable can be
    supplied that computes connection probabilities between individual neurons.
    """

    convergence = IntParam(
        name="convergence",
        optional=True,
        readonly=True,
    )

    divergence = IntParam(
        name="divergence",
        optional=True,
        readonly=True,
    )

    probabilities = ConnectivityProbabilitiesParam(
        name="probabilities",
        default=None,
        optional=True,
        readonly=True,
    )

    @property
    def _argreprs(self):
        return [
            "convergence={}".format(self.convergence),
            "divergence={}".format(self.divergence),
            "probabilities={}".format(self.probabilities),
        ]

    def __init__(self, convergence=None, divergence=None, probabilities=None):
        super().__init__()

        self.convergence = convergence
        self.divergence = divergence
        self.probabilities = probabilities


class SpatiallyConstrainedConnectivity(ConstrainedConnectivity):
    """
    Same as "ConstrainedConnectivity", but with a default callback for the
    "probabilities" callback that computes connection probabilities based on the
    location of the neurons.
    """

    sigma = NumberParam(
        name="sigma",
        low=0.0,
        default=0.25,
        low_open=True,
        readonly=True,
    )

    projection = NdarrayParam(
        name="projection",
        default=np.zeros((0, )),
        optional=True,
        shape=('*', '*'),
        readonly=True,
    )

    @property
    def _argreprs(self):
        return super()._argreprs + [
            "sigma={}".format(self.sigma),
            "projection={}".format(self.projection),
        ]

    def get_probabilities(self, n_pre, n_post, pre_obj, post_obj, data):
        # Fetch the neuron locations
        xs_pre, xs_post = None, None

        # If the "locations" attribute is set
        if (pre_obj in data) and hasattr(data[pre_obj], 'locations'):
            xs_pre = data[pre_obj].locations
        if (post_obj in data) and hasattr(data[post_obj], 'locations'):
            xs_post = data[post_obj].locations

        # We cannot compute connectivity constraints if the locations are not
        # defined -- just use uniform connection probabilities (by returning
        # "None")
        if (xs_pre is None) or (xs_post is None):
            return None

        # Make sure the number of pre-neurons and the number of post-neurons
        # are correct
        if xs_pre.ndim != 2:
            raise ValueError(
                "Pre-population neuron locations must be a 2D array, "
                "but got {}D array".format(xs_pre.ndim))
        if xs_post.ndim != 2:
            raise ValueError(
                "Post-population neuron locations must be a 2D array, "
                "but got {}D array".format(xs_pre.ndim))
        if n_pre != xs_pre.shape[0]:
            raise ValueError(
                "Expected pre-population neuron location shape ({}, d_pre), "
                "but got ({}, d_pre)".format(n_pre, xs_pre.shape[0]))
        if n_post != xs_post.shape[0]:
            raise ValueError(
                "Expected post-population neuron location shape ({}, d_post), "
                "but got ({}, d_post)".format(n_post, xs_post.shape[0]))

        # Fetch the dimensionality of the neuron locations
        d_pre, d_post = xs_pre.shape[1], xs_post.shape[1]

        # Project the locations onto the minimum dimensionality
        d_min, d_max = min(d_pre, d_post), max(d_pre, d_post)
        P = np.eye(d_min,
                   d_max) if self.projection is None else self.projection

        # Make sure the projection vector has the correct size
        if (P.shape[0] != d_min and (d_min != d_max)) or (P.shape[1] != d_max):
            raise ValueError("Expected a projection matrix of size ({}, {}), "
                             "but got projection vector of shape {}".format(
                                 d_min, d_max, P.shape))

        # Apply the projection
        if xs_pre.shape[1] == d_max:
            xs_pre = xs_pre @ P.T
        if xs_post.shape[1] == d_max:
            xs_post = xs_post @ P.T

        # Compute the squared distance
        dists = np.sum(np.square(xs_pre[:, None] - xs_post[None, :]), axis=-1)

        # Apply exponential falloff
        return np.exp(-dists / np.square(self.sigma))

    def __init__(self,
                 convergence=None,
                 divergence=None,
                 probabilities=None,
                 sigma=0.25,
                 projection=None):

        # Call the inherited constructor
        super().__init__(convergence, divergence, probabilities)

        # Copy the sigma parameters
        self.sigma = sigma

        # Copy the projection parameter
        self.projection = projection

        # Copy the probabilities
        if probabilities is None:
            def get_probabilities_wrapper(*args, **kwargs):
                return self.get_probabilities(*args, **kwargs)

            self.probabilities = get_probabilities_wrapper
        else:
            self.probabilities = probabilities


class ConnectivityParam(Parameter):
    """
    The ConnectivityParam class describes the data that can be passed to the
    "connectivity" property of the Connection class. The parameter may either
    be an instance of one of the classes derived from "Connectivity", or a
    dictionary mapping from (pre, post) tuples onto an instance of the
    "Connectivity" class.
    """
    def __init__(self, name):
        super().__init__(name,
                         default=DefaultConnectivity(),
                         optional=True,
                         readonly=False)

    def coerce(self, instance, obj):
        # Make sure obj is either an instance of "Connectivity" or a dictionary
        # of the right format
        if isinstance(obj, Connectivity):
            return obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if (not isinstance(key, tuple)) or (len(key) != 2):
                    raise ValueError(
                        "Expected 2-tuple (pre, post) as connectivity dictionary "
                        "keys, but got {}".format(key))
                if not isinstance(value, Connectivity):
                    raise ValueError(
                        "Expected instance of class Connectivity as dictionary "
                        "keys")
            return obj
        else:
            raise ValueError(
                "\"connectivity\" must either be an instance of "
                "the \"Connectivity\" class or a dictionary mapping from "
                "(pre, post) tuples onto instances of the Connectivity class.")


# Nengo 2.8 compatibility
if hasattr(nengo.connection, 'ConnectionTransformParam'):
    ConnectionTransformParam = nengo.connection.ConnectionTransformParam
else:
    ConnectionTransformParam = nengo.connection.TransformParam


class Connection(nengo.config.SupportDefaultsMixin):

    label = StringParam('label', default=None, optional=True)
    seed = IntParam('seed', default=None, optional=True)

    pre = PreParam('pre')
    post = nengo.connection.PrePostParam('post', nonzero_size_in=True)

    synapse_exc = nengo.connection.SynapseParam(
        'synapse_exc', default=nengo.synapses.Lowpass(tau=0.005))
    synapse_inh = nengo.connection.SynapseParam(
        'synapse_inh', default=nengo.synapses.Lowpass(tau=0.005))

    function_info = ConnectionFunctionParam('function',
                                            default=None,
                                            optional=True)

    transform = ConnectionTransformParam('transform', default=1.0)

    solver = nengo.solvers.SolverParam('solver', default=QPSolver())
    eval_points = nengo.dists.DistOrArrayParam('eval_points',
                                               default=None,
                                               optional=True,
                                               sample_shape=('*', 'size_in'))
    scale_eval_points = BoolParam('scale_eval_points', default=False)
    n_eval_points = IntParam('n_eval_points', default=None, optional=True)
    decode_bias = BoolParam('decode_bias', default=True)

    connectivity = ConnectivityParam('connectivity')

    _param_init_order = [
        'pre', 'post', 'synapse_exc', 'synapse_inh', 'function_info'
    ]

    def __init__(self,
                 pre,
                 post,
                 synapse_exc=Default,
                 synapse_inh=Default,
                 function=Default,
                 transform=Default,
                 solver=Default,
                 eval_points=Default,
                 scale_eval_points=Default,
                 n_eval_points=Default,
                 decode_bias=Default,
                 connectivity=Default,
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
        self.connectivity = connectivity

        # For each pre object add two actual nengo connections: an excitatory
        # path and an inhibitory path
        self.connections = []
        arr_ens, arr_ns, _ = self.pre.flatten()
        for i, (ens, ns) in enumerate(zip(arr_ens, arr_ns)):

            def mkcon(synapse_type, synapse):
                return ConnectionPart(pre=ens,
                                      post=self.post,
                                      transform=np.zeros(
                                          (self.post.size_in, ens.size_out)),
                                      seed=self.seed,
                                      synapse=synapse,
                                      solver=SolverWrapper(
                                          self.solver, i, self, ns,
                                          synapse_type),
                                      synapse_type=synapse_type)

            self.connections.append(
                (mkcon(Excitatory, synapse_exc), mkcon(Inhibitory,
                                                       synapse_inh)))

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
        return not (self.solver.weights or
                    (isinstance(self.pre_obj, Neurons)
                     and isinstance(self.post_obj, Neurons)))

    @property
    def _label(self):
        if self.label is not None:
            return self.label

        return "from %s to %s%s" % (self.pre, self.post, " computing '%s'" %
                                    function_name(self.function)
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
