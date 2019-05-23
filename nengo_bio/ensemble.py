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

from .utils import *

import nengo.ensemble
import nengo.params
import nengo.exceptions

class Ensemble(nengo.ensemble.Ensemble):
    """
    Wrapper class for creating a Dale's Principle enabled neuron population.
    All parameters are passed to the nengo Population object, except for the new
    p_exc and p_inh parameters. These parameters indicate the relative number of
    excitatory and inhibitory neurons.
    """

    p_exc = nengo.params.NumberParam('p_exc', optional=True, low=0.0, high=1.0)
    p_inh = nengo.params.NumberParam('p_inh', optional=True, low=0.0, high=1.0)

    def __init__(self, *args, **kw_args):
        """
        Forwards 
        p_exc: 
        """
        steal_param(self, 'p_exc', kw_args)
        steal_param(self, 'p_inh', kw_args)
        super(Ensemble, self).__init__(*args, **kw_args)

