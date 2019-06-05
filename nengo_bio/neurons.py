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

from nengo.exceptions import ValidationError
from nengo.neurons import NeuronType
from nengo.params import IntParam, NumberParam

from nengo_bio.common import Excitatory, Inhibitory
from nengo_bio.internal import lif_utils

class MultiInputNeuronType(NeuronType):
    inputs = ()

    def step_math(self, dt, J, output):
        """
        The step_math function is not used by MultiInputNeuron instances. This
        is mainly to prevent mixing of MultiInputNeuron instances with standard
        Nengo ensembles.
        """
        raise RuntimeError(
            "Neurons of type can only be used in conjunction with"
            "nengo_bio.Connection")

    def tune(self, dt, model, ens, rng=np.random):
        """
        Called as part of the build process to give the neuron type the chance
        to determine parameters that are required for the weight solver or the
        compilation process. The returned object will be stored in the built
        ensemble and also passed to the compile() function below.
        """
        return None

    def compile(self, dt, n_neurons, tuning):
        """
        Generates an efficient simulator for this neuron type.

        Returns two objects: a callable simulator object and the initial state.
        The simulator object receives the input (read only), state (read/write),
        and an output for each neuron (write only).

        Parameters
        ----------
        dt : float
            The timestep that is going to be used in the simulation. Since this
            parameter is known ahead of time it can be treated as a constant
            expression, potentially increasing performance.
        n_neurons : int
            The number of neurons for which the function should be compiled.
        tuning : object
            The result of the "tune" function above.
        """
        raise NotImplementedError(
            "MultiInputNeuronType must implement the \"compile\" function.")



class TwoCompLIF(MultiInputNeuronType):
    """
    A two-compartment LIF neuron with conductance based synapses.

    A TwoCompLIF neuron consists of a somatic as well as a dendritic
    compartment.
    """

    inputs = (Excitatory, Inhibitory)
    probeable = ('spikes', 'voltage', 'voltage_den', 'refractory_time')

    C_som = NumberParam('c_som', low=0, low_open=True)
    C_den = NumberParam('c_den', low=0, low_open=True)

    g_leak_som = NumberParam('g_leak_som', low=0, low_open=True)
    g_leak_den = NumberParam('g_leak_den', low=0, low_open=True)
    g_couple = NumberParam('g_couple', low=0, low_open=True)

    tau_spike = NumberParam('tau_spike', low=0)
    tau_ref = NumberParam('tau_ref', low=0)

    v_spike = NumberParam('v_spike')
    v_reset = NumberParam('v_reset')
    v_th = NumberParam('v_th')

    E_rev_leak = NumberParam('E_leak')
    E_rev_exc = NumberParam('E_exc')
    E_rev_inh = NumberParam('E_inh')

    subsample = IntParam('subsample', low=1)

    def __init__(self,
                 C_som=1e-9,
                 C_den=1e-9,
                 g_leak_som=50e-9,
                 g_leak_den=50e-9,
                 g_couple=50e-9,
                 tau_spike=1e-3,
                 tau_ref=2e-3,
                 v_spike=20e-3,
                 v_reset=-65e-3,
                 v_th=-50e-3,
                 E_rev_leak=-65e-3,
                 E_rev_exc=20e-3,
                 E_rev_inh=-75e-3,
                 subsample=10):
        self.C_som = C_som
        self.C_den = C_den
        self.g_leak_som = g_leak_som
        self.g_leak_den = g_leak_den
        self.g_couple = g_couple
        self.tau_spike = tau_spike
        self.tau_ref = tau_ref
        self.v_spike = v_spike
        self.v_reset = v_reset
        self.v_th = v_th
        self.E_rev_leak = E_rev_leak
        self.E_rev_exc = E_rev_exc
        self.E_rev_inh = E_rev_inh
        self.subsample = subsample

    def threshold_current(self):
        """
        Returns the input current at which the neuron is supposed to start
        spiking.
        """
        return (self.v_th - self.E_rev_leak) * self.g_leak_som

    def _lif_parameters(self):
        """
        Returns the LIF parameters of the somatic compartments. These parameters
        are used in the gain/bias computations.
        """
        tau_ref = self.tau_spike + self.tau_ref
        tau_rc = self.C_som / self.g_leak_som
        i_th = self.threshold_current()
        return tau_ref, tau_rc, i_th

    def _lif_rate(self, J):
        """
        Returns the LIF rate for a given input current.
        """
        tau_ref, tau_rc, i_th = self._lif_parameters()
        return lif_utils.lif_rate(J / i_th, tau_ref, tau_rc)

    def _lif_rate_inv(self, a):
        """
        Returns the input current resulting in the given rate.
        """
        tau_ref, tau_rc, i_th = self._lif_parameters()
        return lif_utils.lif_rate_inv(a, tau_ref, tau_rc) * i_th

    def gain_bias(self, max_rates, intercepts):
        # Make sure the input is a 1D array
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        # Make sure the maximum rates are not surpassing the maximally
        # attainable rate
        tau_ref, _, i_th = self._lif_parameters()
        inv_tau_ref = 1. / tau_ref if tau_ref > 0 else np.inf
        if np.any(max_rates > inv_tau_ref):
            raise ValidationError("Max rates must be below the inverse "
                                  "of the sum of the refractory and spike "
                                  "period ({0.3f})".format(inv_tau_ref),
                                  attr='max_rates', obj=self)

        # Solve the following linear system for gain, bias
        #   i_th  = gain * intercepts + bias
        #   i_max = gain              + bias
        i_max = self._lif_rate_inv(max_rates)
        gain = (i_max - i_th) / (1 - intercepts)
        bias = i_max - gain

        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        # The max rate is defined as the rate for the input current gain + bias
        max_rates = self._lif_rate(gain + bias)

        # Solve i_th = gain * intercept + bias for the intercept; warn about
        # invalid values
        intercepts = (self.threshold_current() - bias) / gain
        if not np.all(np.isfinite(intercepts)):
            warnings.warn("Non-finite values detected in `intercepts`; this "
                          "probably means that `gain` was too small.")

        return max_rates, intercepts

    def rates(self, x, gain, bias):
        return self._lif_rate(gain * x + bias)
