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

def lif_rate(J, tau_ref=2e-3, tau_rc=20e-3):
    """
    Calculates the firing rate of a LIF neuron with current based synapses.
    Input to the model is the current J. All parameters may be numpy arrays.

    J: input current
    tau_ref: refractory period in seconds.
    tau_rc: membrane time constant in seconds.
    """

    mask = 1 * (J > 1)
    t_spike = -np.log1p(-mask * 1.0 / (mask * J + (1 - mask))) * tau_rc
    return mask / (tau_ref + t_spike)


def lif_rate_inv(r, tau_ref=2e-3, tau_rc=20e-3):
    """
    Calculates the firing rate of a LIF neuron with current based synapses.
    Input to the model is the current J. All parameters may be numpy arrays.

    r: input rate
    tau_ref: refractory period in seconds.
    tau_rc: membrane time constant in seconds.
    """
    mask = 1.0 * (r > 1e-6)
    return -mask / (np.exp((r * tau_ref - 1) / ((1.0 - mask) + r * tau_rc)) - 1)

