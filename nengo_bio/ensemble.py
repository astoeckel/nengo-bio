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
import collections

from nengo_bio.common import steal_param, Excitatory, Inhibitory

import nengo.ensemble
import nengo.builder

class Ensemble(nengo.ensemble.Ensemble):
    """
    Wrapper class for creating a Dale's Principle enabled neuron population.
    All parameters are passed to the nengo Population object, except for the new
    p_exc and p_inh parameters. These parameters indicate the relative number of
    excitatory and inhibitory neurons.
    """

    @staticmethod
    def _coerce_p_exc_inh(p_exc, p_inh, ens=None):
        has_p_exc, has_p_inh = not p_exc is None, not p_inh is None
        if has_p_exc or has_p_inh:
            # Both p_exc and p_inh are given must sum to one
            if has_p_exc and has_p_inh:
                if abs(p_exc + p_inh - 1.0) > 1e-3:
                    raise ValueError(
                        "p_exc={} and p_inh={} do not add up to one for {}"
                        .format(p_exc, p_inh, ens)
                    )

            # At least p_exc is given, check range
            if has_p_exc:
                if p_exc < 0.0 or p_exc > 1.0:
                    raise ValueError(
                        "p_exc={} must be between 0.0 and 1.0 for {}"
                        .format(p_exc, ens))
                p_inh = 1.0 - p_exc
            # At least p_inh is given, check range
            elif has_p_inh:
                if p_inh < 0.0 or p_inh > 1.0:
                    raise ValueError(
                        "p_inh={} must be between 0.0 and 1.0 for {}"
                        .format(p_inh, ens))
                p_exc = 1.0 - p_inh
        return p_exc, p_inh

    def __init__(self, *args, **kw_args):
        """
        Forwards all parameters except for p_exc and p_inh to the parent
        Ensemble class.
        """

        self._p_exc, _ = Ensemble._coerce_p_exc_inh(
            steal_param('p_exc', kw_args), steal_param('p_inh', kw_args), self)

        super(Ensemble, self).__init__(*args, **kw_args)

    @property
    def p_exc(self):
        return self._p_exc

    @property
    def p_inh(self):
        if self._p_exc is None:
            return None
        else:
            return 1.0 - self._p_exc

    @p_exc.setter
    def p_exc(self, value):
        self._p_exc, _ = Ensemble._coerce_p_exc_inh(value, None, self)

    @p_inh.setter
    def p_inh(self, value):
        self._p_exc, _ = Ensemble._coerce_p_exc_inh(None, value, self)

