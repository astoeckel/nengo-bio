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

import nengo

def set_defaults():
    """
    Sets the default parameters for ensembles to more biologically realisitc
    defaults. In particular, makes sure that the maximum firing rate is between
    50 and 100 Hz and restricts the intercepts to values between -0.9 and 0.9.
    This prevents target currents from being too large.
    """
    nengo.Ensemble.max_rates.default = nengo.dists.Uniform(50, 100)
    nengo.Ensemble.intercepts.default = nengo.dists.Uniform(-0.9, 0.9)
