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

class QPSolver(ExtendedSolver):
    def __call__(self, A, J, synapse_types, rng=np.random):
        ws = np.array((0.0, 1.0, -1.0, 1.0, 0.0, 0.0))
        reg = (0.01 * np.max(A))**2 * A.shape[1]
        return qp_solver.solve(A, J, ws,
            synapse_types, iTh=1.0, reg=reg, use_lstsq=True)

