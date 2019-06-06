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

import json
import numpy as np

class SomaticParameters:
    """
    The SomaticParameters class describes the parameters of the active LIF
    compartment such as the refractory period, spike time and threshold
    potential.
    """

    def __init__(self,
                 tau_ref=2e-3,
                 tau_spike=1e-3,
                 v_th=-50e-3,
                 v_reset=-65e-3,
                 v_spike=20e-3):
        self.tau_ref = tau_ref
        self.tau_spike = tau_spike
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_spike = v_spike


class DendriticParameters:
    """
    The DendriticParameters class describes the connectivity of the dendritic
    tree. In particular, the dynamics of the tree are given as

    dv/dt = [C + diag(a_const + A @ x)] @ v + [b_const + B @ x]
    """

    def __init__(self, A, a_const, B, b_const, C):
        # Copy the matrices/vectors
        self.A = np.asarray(A, order='C', dtype=np.float64)
        self.a_const = np.asarray(a_const, order='C', dtype=np.float64)
        self.B = np.asarray(B, order='C', dtype=np.float64)
        self.b_const = np.asarray(b_const, order='C', dtype=np.float64)
        self.C = np.asarray(C, order='C', dtype=np.float64)

        # Make sure they have the right sizes
        self.a_const = self.a_const.flatten()
        self.b_const = self.b_const.flatten()
        assert self.A.ndim == self.B.ndim == self.C.ndim == 2
        assert self.C.shape[0] == self.C.shape[1]
        assert self.a_const.size == self.A.shape[0] == self.C.shape[0]
        assert self.b_const.size == self.B.shape[0] == self.C.shape[0]
        assert self.A.shape[1] == self.B.shape[1]

    @property
    def n_comp(self):
        """Returns the number of compartments in the neuron model."""
        return self.C.shape[0]

    @property
    def n_inputs(self):
        """Returns the number of inputs feeding into the neuron model."""
        return self.A.shape[1]

    @staticmethod
    def make_lif(C_som, g_leak_som, E_rev_leak, input_mul=1.):
        """
        This function creates a DendriticParameters instance describing the
        parameters of a single-compartment, current-based LIF neuron.
        """

        A = np.array(((0., 0.), )) / C_som
        a_const = np.array((-g_leak_som, )) / C_som

        B = np.array(((input_mul, -input_mul), )) / C_som
        b_const = np.array((E_rev_leak * g_leak_som, )) / C_som

        C = np.array(((0., ), )) / C_som

        return DendriticParameters(A, a_const, B, b_const, C)

    @staticmethod
    def make_two_comp_lif(C_som=1e-9,
                          C_den=1e-9,
                          g_leak_som=50e-9,
                          g_leak_den=50e-9,
                          g_couple=50e-9,
                          E_rev_leak=-65e-3,
                          E_rev_exc=20e-3,
                          E_rev_inh=-75e-3,
                          input_mul=1.):
        """
        This function creates a DendriticParameters instance describing the
        parameters of a two-compartment LIF neuron.
        """

        Cs = np.array((C_som, C_den))

        A = np.array(((0., 0.), (-input_mul, -input_mul))) / Cs[:, None]
        a_const = np.array(
            (-(g_leak_som + g_couple), -(g_leak_den + g_couple))) / Cs

        B = np.array(
            ((0., 0.),
             (input_mul * E_rev_exc, input_mul * E_rev_inh))) / Cs[:, None]
        b_const = np.array(
            (E_rev_leak * g_leak_som, E_rev_leak * g_leak_den)) / Cs

        C = np.array(((0., g_couple), (g_couple, 0.))) / Cs[:, None]

        return DendriticParameters(A, a_const, B, b_const, C)

