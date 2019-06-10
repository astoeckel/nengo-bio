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
from nengo.dists import Uniform
from nengo.cache import Fingerprint

from nengo_bio.common import Excitatory, Inhibitory
from nengo_bio.internal import (lif_utils, multi_compartment_lif_parameters)


class MultiChannelNeuronType(NeuronType):
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

    @property
    def n_inputs(self):
        return len(self.inputs)

    def tune(self, dt, model, ens):
        return None

    def compile(self, dt, n_neurons, tuning):
        """
        Generates an efficient simulator for this neuron type.

        Returns a callable "step_math" function. The "step_math" function
        receives the current input and returns the output (whether a neuron
        spiked or not), as well as the current state.

        Parameters
        ----------
        dt : float
            The timestep that is going to be used in the simulation. Since this
            parameter is known ahead of time it can be treated as a constant
            expression, potentially increasing performance.
        n_neurons : int
            The number of neurons for which the function should be compiled.
        """
        raise NotImplementedError(
            "MultiInputNeuronType must implement the \"compile\" function.")


class LIF(MultiChannelNeuronType):
    """
    A standard single-compartment LIF neuron with separate exciatory and
    inhibitory inputs.
    """

    inputs = (Excitatory, Inhibitory)

    C_som = NumberParam('C', low=0, low_open=True)
    g_leak_som = NumberParam('g_leak', low=0, low_open=True),

    tau_spike = NumberParam('tau_spike', low=0)
    tau_ref = NumberParam('tau_ref', low=0)

    v_spike = NumberParam('v_spike')
    v_reset = NumberParam('v_reset')
    v_th = NumberParam('v_reset')

    E_rev_leak = NumberParam('E_leak')

    subsample = IntParam('subsample', low=1)

    def __init__(self,
                 C_som=1e-9,
                 g_leak_som=50e-9,
                 E_rev_leak=-65e-3,
                 tau_ref=2e-3,
                 tau_spike=1e-3,
                 v_th=-50e-3,
                 v_reset=-65e-3,
                 v_spike=20e-3,
                 subsample=10):

        super(LIF, self).__init__()

        self.C_som = C_som
        self.g_leak_som = g_leak_som
        self.E_rev_leak = E_rev_leak
        self.tau_ref = tau_ref
        self.tau_spike = tau_spike
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_spike = v_spike
        self.subsample = subsample

    @property
    def threshold_current(self):
        return (self.v_th - self.v_reset) * self.g_leak_som

    def _lif_parameters(self):
        """
        Returns the LIF parameters of the somatic compartments. These parameters
        are used in the gain/bias computations.
        """
        tau_ref = self.tau_spike + self.tau_ref
        tau_rc = self.C_som / self.g_leak_som
        i_th = self.threshold_current
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
        inv_tau_ref = 1. / tau_ref if tau_ref > 0. else np.inf
        if np.any(max_rates > inv_tau_ref):
            raise ValidationError(
                "Max rates must be below the inverse "
                "of the sum of the refractory and spike "
                "period ({:0.3f})".format(inv_tau_ref),
                attr='max_rates',
                obj=self)

        # Solve the following linear system for gain, bias
        #   i_th  = gain * intercepts + bias
        #   i_max = gain              + bias
        i_max = self._lif_rate_inv(max_rates)
        gain = (i_max - i_th) / (1. - intercepts)
        bias = i_max - gain

        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        # The max rate is defined as the rate for the input current gain + bias
        max_rates = self._lif_rate(gain + bias)

        # Solve i_th = gain * intercept + bias for the intercept; warn about
        # invalid values
        i_th = self.threshold_current
        intercepts = (i_th - bias) / gain
        if not np.all(np.isfinite(intercepts)):
            warnings.warn("Non-finite values detected in `intercepts`; this "
                          "probably means that `gain` was too small.")

        return max_rates, intercepts

    def rates(self, x, gain, bias):
        return self._lif_rate(gain * x + bias)

    def _params_som(self):
        return multi_compartment_lif_parameters.SomaticParameters(
            tau_ref=self.tau_ref,
            tau_spike=self.tau_spike,
            v_th=self.v_th,
            v_reset=self.v_reset,
            v_spike=self.v_spike,
        )

    def _params_den(self):
        return multi_compartment_lif_parameters.DendriticParameters.\
            make_lif(
            C_som=self.C_som,
            g_leak_som=self.g_leak_som,
            E_rev_leak=self.E_rev_leak)

    def _compile(self, dt, force_python_sim=False):
        import nengo_bio.internal.multi_compartment_lif_cpp as mcl_cpp
        import nengo_bio.internal.multi_compartment_lif_python as mcl_python

        # Either instantiate the C++ simulator or the reference simulator
        params_som = self._params_som()
        params_den = self._params_den()
        if force_python_sim or not mcl_cpp.supports_cpp():
            sim_class = mcl_python.compile_simulator_python(
                params_som, params_den, dt=dt, ss=self.subsample)
        else:
            sim_class = mcl_cpp.compile_simulator_cpp(
                params_som, params_den, dt=dt, ss=self.subsample)

        # Return the simulator class
        return sim_class

    def _estimate_model_weights(self):
        return np.array((0., 1., -1., 1., 0, 0))

    def _filter_input(self, in_exc, in_inh):
        """
        Function used to quickly discard samples that will definetively lead to
        a zero output rate.
        """
        b0, b1, b2, a0, a1, a2 = self._estimate_model_weights()
        return (b0 + b1 * in_exc + b2 * in_inh) / (
            a0 + a1 * in_exc + a2 * in_inh) * 1.2 > self.threshold_current

    def _estimate_input_range(self, max_rate):
        """
        This function returns the 2D area over the excitatory and inhibitory
        input that should be sampled by the "tune" function.
        """

        # Fetch the model parameters. Estimate the absolute maximum and minimum
        # current.
        b0, b1, b2, a0, a1, a2 = self._estimate_model_weights()
        Jmin = b2 / a2 if a2 != 0.0 else -np.inf
        Jmax = b1 / a1 if a1 != 0.0 else  np.inf

        # Convert the given ramp to a current. Clamp the rate to the maximum/
        # minimum currents.
        J = self._lif_rate_inv(max_rate)
        J = np.clip(J, Jmin * 0.95, Jmax * 0.95)

        # Compute the gE that will reach the computed J for gI = 0
        gE = -(b0 - J * a0) / (b1 - J * a1)

        # Compute the gI that will result in J = i_th for the above gE
        gI = (self.threshold_current - (b0 + b1 * gE)) / b2

        # Return gE and gI
        return gE, gI

    def tune(self, dt, model, ens):
        from nengo_bio.internal.multi_compartment_lif_sim import GaussianSource
        from nengo_bio.internal.model_weights import tune_two_comp_model_weights

        # Fetch the maximum rates for which we need to determine the neuron
        # parameters
        max_rates = model.params[ens].max_rates
        if isinstance(ens.max_rates, Uniform):
            min_max_rate = ens.max_rates.low
            max_max_rate = ens.max_rates.high
        else:
            min_max_rate = np.min(max_rates)
            max_max_rate = np.max(max_rates)

        # Get the parameter hash
        params_hash = str(Fingerprint([self, dt]))

        # Function running a single neuron simulation
        sim_class = self._compile(dt)

        def run_single_sim(idx, out, in_exc, in_inh):
            xs = np.asarray((in_exc, in_inh), order='C', dtype=np.float64)
            # TODO: Get these parameters from the connection
            sim_class().run_single_with_gaussian_sources(
                out, [
                    GaussianSource(
                        seed=4902 + idx,
                        mu=in_exc,
                        sigma=in_exc * 0.2,
                        tau=5e-3,
                        offs=0.0),
                    GaussianSource(
                        seed=5821 + 7 * idx,
                        mu=in_inh,
                        sigma=in_inh * 0.2,
                        tau=5e-3,
                        offs=0.0),
                ])

        return tune_two_comp_model_weights(
            dt=dt,
            max_rates=max_rates,
            min_max_rate=min_max_rate,
            max_max_rate=max_max_rate,
            run_single_sim=run_single_sim,
            estimate_input_range=self._estimate_input_range,
            filter_input=self._filter_input,
            lif_rate_inv=self._lif_rate_inv,
            params_hash=params_hash)

    def compile(self,
                dt,
                n_neurons,
                tuning=None,
                force_python_sim=False,
                get_class=False):
        sim_class = self._compile(dt, force_python_sim)
        if get_class:
            return sim_class
        return sim_class(n_neurons).run_step_from_memory


class LIFCond(LIF):
    """
    Single compartment LIF neuron with conductance-based synapses.
    """

    inputs = (Excitatory, Inhibitory)

    E_rev_exc = NumberParam('E_exc')
    E_rev_inh = NumberParam('E_inh')

    def __init__(self,
                 C_som=1e-9,
                 g_leak_som=50e-9,
                 E_rev_leak=-65e-3,
                 E_rev_exc=20e-3,
                 E_rev_inh=-75e-3,
                 tau_ref=2e-3,
                 tau_spike=1e-3,
                 v_th=-50e-3,
                 v_reset=-65e-3,
                 v_spike=20e-3,
                 subsample=10):

        super(LIFCond, self).__init__(
            C_som=C_som,
            g_leak_som=g_leak_som,
            E_rev_leak=E_rev_leak,
            tau_ref=tau_ref,
            tau_spike=tau_spike,
            v_th=v_th,
            v_reset=v_reset,
            v_spike=v_spike,
            subsample=subsample)

        self.E_rev_exc = E_rev_exc
        self.E_rev_inh = E_rev_inh

    def _estimate_model_weights(self):
        v_som = 0.5 * (self.v_th + self.v_reset)
        ws = np.array((
            0.,
            (self.E_rev_exc - v_som),
            (self.E_rev_inh - v_som),
            1.,
            0.,
            0.,
        ))

        # Normalise ws[1] = 1
        ws = ws / ws[1]

        return ws

    def _params_den(self):
        return multi_compartment_lif_parameters.DendriticParameters.\
            make_lif_cond(
            C_som=self.C_som,
            g_leak_som=self.g_leak_som,
            E_rev_leak=self.E_rev_leak,
            E_rev_exc=self.E_rev_exc,
            E_rev_inh=self.E_rev_inh
        )


class TwoCompLIF(LIFCond):
    """
    A two-compartment LIF neuron with conductance based synapses.

    A TwoCompLIF neuron consists of a somatic as well as a dendritic
    compartment.
    """

    inputs = (Excitatory, Inhibitory)

    C_den = NumberParam('c_den', low=0, low_open=True)

    g_leak_den = NumberParam('g_leak_den', low=0, low_open=True)
    g_couple = NumberParam('g_couple', low=0, low_open=True)

    def __init__(self,
                 C_som=1e-9,
                 C_den=1e-9,
                 g_leak_som=50e-9,
                 g_leak_den=50e-9,
                 g_couple=50e-9,
                 E_rev_leak=-65e-3,
                 E_rev_exc=20e-3,
                 E_rev_inh=-75e-3,
                 tau_ref=2e-3,
                 tau_spike=1e-3,
                 v_th=-50e-3,
                 v_reset=-65e-3,
                 v_spike=20e-3,
                 subsample=10):

        super(TwoCompLIF, self).__init__(
            C_som=C_som,
            g_leak_som=g_leak_som,
            E_rev_leak=E_rev_leak,
            E_rev_exc=E_rev_exc,
            E_rev_inh=E_rev_inh,
            tau_ref=tau_ref,
            tau_spike=tau_spike,
            v_th=v_th,
            v_reset=v_reset,
            v_spike=v_spike,
            subsample=subsample)

        self.C_den = C_den
        self.g_leak_den = g_leak_den
        self.g_couple = g_couple

    def _estimate_model_weights(self):
        v_som = 0.5 * (self.v_th + self.v_reset)
        ws = np.array((
            self.g_couple * self.g_leak_den * (self.E_rev_leak - v_som),
            self.g_couple * (self.E_rev_exc - v_som),
            self.g_couple * (self.E_rev_inh - v_som),
            self.g_couple + self.g_leak_den,
            1.,
            1.,
        ))

        # Normalise ws[1] = 1
        ws = ws / ws[1]

        return ws

    def _params_den(self):
        return multi_compartment_lif_parameters.DendriticParameters.\
            make_two_comp_lif(
            C_som=self.C_som,
            C_den=self.C_den,
            g_leak_som=self.g_leak_som,
            g_leak_den=self.g_leak_den,
            g_couple=self.g_couple,
            E_rev_leak=self.E_rev_leak,
            E_rev_exc=self.E_rev_exc,
            E_rev_inh=self.E_rev_inh
        )


# Alias for TwoCompLIF emphasizing the use of conductance based synapses
TwoCompLIFCond = TwoCompLIF

# Whitelist the neuron types
Fingerprint.whitelist(LIF)
Fingerprint.whitelist(LIFCond)
Fingerprint.whitelist(TwoCompLIF)

