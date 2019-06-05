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

from nengo_bio.neurons import TwoCompLIF

def test_two_comp_lif_lif_rate():
    lif = TwoCompLIF()

    # Sample the current space
    i_th = lif.threshold_current()
    js = np.linspace(-10 * i_th, 10 * i_th, 1000)

    # Compute the LIF rates. Make sure the threshold current computation was
    # correct-
    rates = lif._lif_rate(js)
    assert np.all(rates[js < i_th] == 0)
    assert np.all(rates[js > i_th] > 0)

    # Apply the inverse. Make sure the resulting function is linear.
    js_reconstructed = lif._lif_rate_inv(rates)
    assert np.all(js_reconstructed[js < i_th] == 0)
    assert np.allclose(js[js > i_th], js_reconstructed[js > i_th])


def test_two_comp_lif_gain_bias():
    lif = TwoCompLIF()

    # Sample some maximum rates and intercepts
    max_rates = np.linspace(10, 100, 100)
    intercepts = np.linspace(-0.99, 0.99, 100)

    # Compute the corresponding gain and bias
    gain, bias = lif.gain_bias(max_rates, intercepts)

    # Make sure the gain and bias are correct
    assert np.allclose(max_rates, lif._lif_rate(gain + bias))
    assert np.allclose(lif.threshold_current(), gain * intercepts + bias)

    # Make sure the inverse function to gain, bias works
    max_rates_reconstructed, intercepts_reconstructed = \
        lif.max_rates_intercepts(gain, bias)
    assert np.allclose(max_rates, max_rates_reconstructed)
    assert np.allclose(intercepts, intercepts_reconstructed)


