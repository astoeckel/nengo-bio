#!/usr/bin/env python3

import nengo
import nengo_bio as bio
import numpy as np

with nengo.Network() as model:
    inp_a = nengo.Node(0)

    # Excitatory source population
    ens_source = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=1.0)

    # Inhibitory inter-neuron population
    ens_inhint = bio.Ensemble(n_neurons=102, dimensions=1, p_inh=1.0)
                              #intercepts=nengo.dists.Uniform(-1.5, -0.25))

    # Target population
    ens_target = bio.Ensemble(n_neurons=103, dimensions=1)

    nengo.Connection(inp_a, ens_source)
    bio.Connection(ens_source, ens_inhint)
    bio.Connection((ens_source, ens_inhint), ens_target,
                   function=lambda x: np.mean(x)**2,
                   eval_points=[1, 1] * np.linspace(-1, 1, 1000)[:, None])

#    # TODO: The above should be simpler, probably allow something such as
#    bio.Connection(Join(ens_source, ens_inhint), ens_target,
#                   function=lambda x: x**2)
#    # Where the default operation is "Stack"

#    # Note: the solver of the last connection has access to the state of both
#    # the pre-populations at the same time. This is equivalent to a connection
#    # from a 2D ensemble with orthogonal encoders.
#    ens_combined = nengo.Ensemble(n_neurons=203, dimensions=2,
#                    encoders=nengo.dists.Choice([[0, 1], [1, 0], [0, -1], [-1, 0]]))
#    nengo.Connection(inp_a, ens_combined[0])
#    nengo.Connection(ens_combined[0], ens_combined[1])
#    nengo.Connection(ens_combined, ens_target, function=lambda x: np.mean(x))

with nengo.Simulator(model, progress_bar=None) as sim:
    sim.run(1.0)
