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

from nengo_bio.neurons import MultiChannelNeuronType

import nengo.builder
from nengo.builder.operator import Operator

class SimMultiChannelNeurons(Operator):
    def __init__(self, neurons, step_math, xs, output, tag=None):
        super().__init__(tag=tag)
        self.neurons = neurons
        self.step_math = step_math

        self.sets = [output]
        self.incs = []
        self.reads = [x for x in xs]
        self.updates = []

    @property
    def xs(self):
        return self.reads

    @property
    def output(self):
        return self.sets[0]

    @property
    def states(self):
        return self.sets[1:]

    def _descstr(self):
        return '%s, %s, %s' % (self.neurons, self.xs, self.output)

    def make_step(self, signals, dt, rng):
        xs = [signals[x] for x in self.xs]
        output = signals[self.output]
        def step_simneurons():
            self.step_math(output, *xs)
        return step_simneurons


@nengo.builder.Builder.register(MultiChannelNeuronType)
def build_neurons(model, neuron_type, neurons):
    # Compile the step_math function for this neuron type
    tuning = model.params[neurons.ensemble].tuning
    step_math = neuron_type.compile(model.dt, neurons.size_in, tuning)

    # Add the MultiChannelNeuronType simulator operator
    n_inputs = neuron_type.n_inputs
    xs = [model.sig[neurons]['in_{}'.format(i)] for i in range(n_inputs)]
    model.add_op(SimMultiChannelNeurons(neurons=neuron_type,
                                        step_math=step_math,
                                        xs=xs,
                                        output=model.sig[neurons]['out']))
