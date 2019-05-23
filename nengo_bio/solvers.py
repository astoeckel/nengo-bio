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

import time
import nengo.solvers
import nengo.params
import nengo.builder

class WeightType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

Excitatory = WeightType("Excitatory")
Inhibitory = WeightType("Inhibitory")

class QPSolver(nengo.solvers.Solver):
    """
    The QPSolver class is a stub that is attached to the excitatory and
    inhibitory connections bridging two ensembles.
    """

    compositional = False

    def __init__(self, pre, pre_idx, post, connection, weight_type):
        super().__init__(weights=True)
        self.pre = pre
        self.pre_idx = pre_idx
        self.post = post
        self.connection = connection
        self.weight_type = weight_type
        self.model = None

    def __call__(self, A, Y, rng=np.random):
        assert False, "This method should never be called directly"

