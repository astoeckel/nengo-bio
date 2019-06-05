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

from nengo_bio.internal.multi_compartment_lif_parameters import \
    DendriticParameters


def test_dendritic_parameters_two_comp_lif():
    params = DendriticParameters.make_two_comp_lif(
        C_som=1e-9,
        C_den=1e-9,
        g_leak_som=50e-9,
        g_leak_den=50e-9,
        g_couple=50e-9,
        E_rev_leak=-65e-3,
        E_rev_exc=20e-3,
        E_rev_inh=-75e-3,
        input_mul=1e-9)

    assert params.n_comp == 2
    assert params.n_inputs == 2
    assert np.allclose(params.A, [[ 0.,  0.], [-1., -1.]])
    assert np.allclose(params.a_const, [-100., -100.])
    assert np.allclose(params.B, [[ 0.,  0.], [0.02, -0.075]])
    assert np.allclose(params.b_const, [-3.25, -3.25])
    assert np.allclose(params.C, [[0., 50.], [50., 0.]])

