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

from nengo.params import BoolParam, NumberParam
from nengo.solvers import Solver
from nengo_bio.internal import qp_solver

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

    def __call__(self, A, J, connectivity, rng=np.random):
        return self.solver(A, J, connectivity, rng)


class QPSolver(ExtendedSolver):

    reg = NumberParam('reg', low=0)
    relax = BoolParam('relax')

    def __init__(self, reg=1e-3, relax=False):
        super().__init__()
        self.reg = reg
        self.relax = relax

    def __call__(self, A, J, connectivity, rng=np.random):
        # Neuron model parameters. For now we only support current-based LIF
        ws = np.array((0.0, 1.0, -1.0, 1.0, 0.0, 0.0))

        # Determine the final regularisatio nparameter
        reg = (self.reg * np.max(A))**2 * A.shape[1]

        # If subthreshold relaxation is switched off, set the spike threshold
        # to "None"
        iTh = None if not self.relax else 1.0

        # Use the faster NNLS solver instead of CVXOPT if we do not need
        # current relaxation
        use_lstsq = iTh is None

        return qp_solver.solve(A, J, ws, connectivity,
                               iTh=iTh, reg=reg, use_lstsq=use_lstsq)

