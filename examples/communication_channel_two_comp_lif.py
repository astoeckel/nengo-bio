#!/usr/bin/env python3

import nengo
import nengo_bio as bio
import numpy as np

with nengo.Network(seed=4589) as model:
    inp_a = nengo.Node(0)
    inp_b = nengo.Node(0)

    ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.5,
                         max_rates=nengo.dists.Uniform(50, 100),
                         intercepts=nengo.dists.Uniform(-0.95, 0.95))

    ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.5,
                         max_rates=nengo.dists.Uniform(50, 100),
                         intercepts=nengo.dists.Uniform(-0.95, 0.95))

    ens_c = bio.Ensemble(n_neurons=103, dimensions=2,
                         neuron_type=bio.neurons.TwoCompLIF(),
                         intercepts=nengo.dists.Uniform(-0.95, 0.95),
                         max_rates=nengo.dists.Uniform(50, 100))

    nengo.Connection(inp_a, ens_a)
    nengo.Connection(inp_b, ens_b)

    bio.Connection((ens_a, ens_b), ens_c)

with nengo.Simulator(model, progress_bar=None, dt=1e-4) as sim:
    sim.run(1.0)
