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

    def _reduce_system(self, A, b, i, v):
        """
        Reduces given the system by clamping a certain compartment i to the
        given potential v. Returns three matrices corresponding to the reduced
        system A_red, b_red and the vector required to compute the current
        flowing into the clamped compartment.
        """

        # Abort if this is a one-compartment neuron
        n = A.shape[0]
        if n <= 1:
            return 0, 0, 0

        # Fetch the A sub-matrix, add the current caused by the clamped somatic
        # compartment
        sel = list(range(0, i)) + list(range(i + 1, n))
        A_red = A[np.ix_(sel, sel)]
        b_red = b[sel] + v * A[sel, i]
        return A_red, b_red

    def vEq_extreme(self, params_som=None):
        """Returns the absolute minimum and maximum potentials reachable in each
           compartment by setting a conductance input to infinity. Note that
           current inputs will"""

        # Initialize v_min and v_max to the resting potential
        A, b = self.C + np.diag(self.a_const), self.b_const
        v_eq = -np.linalg.solve(A, b)
        v_min, v_max = v_eq, v_eq

        # Iterate over all compartments and all inputs and see what happens when
        # we set one of the inputs to infinity.
        for i in range(self.n_comp):
            for j in range(self.n_inputs):
                for sign in [1, -1]:
                    # Skip inputs that are not directly connected to the given
                    # compartment
                    if self.A[i, j] == 0.0 and self.B[i, j] == 0.0:
                        continue

                    # Compute the reversal potential for each compartment this input
                    # is connected to. Set current based inputs to a reversal
                    # potential of infinity
                    if self.A[i, j] == 0:
                        E_rev = sign * np.inf
                    else:
                        E_rev = self.B[i, j] / self.A[i, j]

                    # Assume the compartment had the potential we just computed. Do
                    # this by reducing system to a smaller one without this
                    # compartment.
                    if self.n_comp > 1:
                        A_red, b_red = self._reduce_system(A, b, i, E_rev)
                        v_eq = -np.linalg.solve(A_red, b_red)
                    else:
                        v_eq = E_rev

                    # Update the minimum/maximum potential
                    v_min = np.minimum(v_min, v_eq)
                    v_max = np.maximum(v_max, v_eq)

        # If present, use some knowledge from params_som to restrict the
        # minimum/maximum voltages further
        if not params_som is None:
            if np.isinf(v_min[0]):
                v_min[0] = params_som.v_reset
            if np.isinf(v_max[0]):
                v_max[0] = params_som.v_spike

        return v_min, v_max

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
    def make_lif_cond(C_som=1e-9,
                      g_leak_som=50e-9,
                      E_rev_leak=-65e-3,
                      E_rev_exc=20e-3,
                      E_rev_inh=-75e-3,
                      input_mul=1.):
        """
        This function creates a DendriticParameters instance describing the
        parameters of a single-compartment, conductance-based LIF neuron.
        """

        A = np.array(((-input_mul, -input_mul), )) / C_som
        a_const = np.array((-g_leak_som, )) / C_som

        B = np.array(
            ((input_mul * E_rev_exc, input_mul * E_rev_inh), )) / C_som
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

