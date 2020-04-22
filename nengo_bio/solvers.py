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

from nengo.params import BoolParam, NumberParam
from nengo.solvers import Solver

_solve_qp = None

def _get_solve_qp_instance():
    # Only run the below code once
    global _solve_qp
    if not _solve_qp is None:
        return _solve_qp

    # Determine which library to use
    try:
        import bioneuronqp
        try:
            bioneuronqp.solve(
                np.ones((10, 5)),
                np.ones((10, 3)),
                np.array([0.0, 1.0, -1.0, 1.0, 0.0, 0.0]),
                n_threads=1)
            _solve_qp = bioneuronqp.solve
            return _solve_qp
        except OSError:
            warnings.warn(
                "bioneuronqp is installed, but did not find libbioneuronqp.so; "
                "make sure the library is located in your search path",
                category=UserWarning)
    except ImportError:
        warnings.warn(
            "Install the bioneuronqp library to make solving for weights "
            "faster",
            category=UserWarning)
        import nengo_bio.internal.qp_solver

    # Use the internal solver instead
    import nengo_bio.internal.qp_solver
    _solve_qp = nengo_bio.internal.qp_solver.solve
    return _solve_qp


class ExtendedSolver(Solver):
    """
    Extended solver class used by nengo_bio. Passes neuron types and target
    currents to the solver routine. Always solves for a full weight matrix.
    Non-compositional.
    """

    compositional = False

    def __init__(self):
        super().__init__(weights=True)

    def __call__(self, A, J, synapse_types, rng=np.random):
        raise NotImplementedError("Solvers must implement '__call__'")


class SolverWrapper(ExtendedSolver):
    def __init__(self, solver, pre_idx, connection, neuron_indices,
                 synapse_type):
        super().__init__()
        self.solver = solver
        self.pre_idx = pre_idx
        self.connection = connection
        self.neuron_indices = neuron_indices
        self.synapse_type = synapse_type

    def __call__(self, A, J, connection_matrix, i_th, tuning, rng=np.random):
        return self.solver(A, J, connection_matrix, i_th, tuning, rng)


class QPSolver(ExtendedSolver):

    reg = NumberParam('reg', low=0)
    relax = BoolParam('relax')

    def __init__(self, reg=1e-3, relax=False, extra_args=None):
        super().__init__()
        self.reg = reg
        self.relax = relax
        self.extra_args = {} if extra_args is None else extra_args

    def __call__(self, A, J, connection_matrix, i_th, tuning, rng=np.random):
        # Neuron model parameters. For now we only support current-based LIF
        if tuning is None:
            ws = np.array((0.0, 1.0, -1.0, 1.0, 0.0, 0.0))
        else:
            ws = tuning

        # Determine the final regularisatio nparameter
        reg = (self.reg * np.max(A))**2

        # If subthreshold relaxation is switched off, set the spike threshold
        # to "None"
        i_th = None if not self.relax else i_th

        # Use the faster NNLS solver instead of CVXOPT if we do not need
        # current relaxation
        use_lstsq = i_th is None

        return _get_solve_qp_instance()(
            A, J, ws, connection_matrix, iTh=i_th, reg=reg,
            use_lstsq=use_lstsq, **self.extra_args)

