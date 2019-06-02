#!/usr/bin/env python3

import nengo
import nengo_bio as bio

with nengo.Network() as model:
    inp_a = nengo.Node(lambda t: 1.0 if (t > 0.1 and t < 1.1) else 0.0)
    ens_a = bio.Ensemble(n_neurons=100, dimensions=1, p_exc=1.0)
    ens_b = bio.Ensemble(n_neurons=100, dimensions=1, p_inh=1.0)

    tau = 100e-3

    nengo.Connection(inp_a, ens_a)
    bio.Connection((ens_a, ens_b), ens_b,
                   function=lambda x: tau * x[0] + x[1],
                   synapse_exc=tau,
                   synapse_inh=tau)

with nengo.Simulator(model) as sim:
    sim.run(1.0)
