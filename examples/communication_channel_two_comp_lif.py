#!/usr/bin/env python3

import nengo
import nengo_bio as bio
import numpy as np

with nengo.Network() as model:
    inp_a = nengo.Node(0)
    inp_b = nengo.Node(0)

    ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8,
                         eval_points=nengo.dists.Uniform(0, 1))
    ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_inh=0.8,
                         eval_points=nengo.dists.Uniform(0, 1))
    ens_c = bio.Ensemble(n_neurons=103, dimensions=1,
                         neuron_type=bio.neurons.TwoCompLIF(
                             g_couple=100e-9
                         ),
                         max_rates=nengo.dists.Uniform(75, 100),
                         encoders=nengo.dists.Choice([[1]]),
                         intercepts=nengo.dists.Uniform(-0.9, 0.9))

    nengo.Connection(inp_a, ens_a)
    nengo.Connection(inp_b, ens_b)

    bio.Connection((ens_a, ens_b), ens_c,
                   function=lambda x: x[0] * x[1],
                   solver=bio.solvers.QPSolver(relax=True, reg=1e-5))

with nengo.Simulator(model) as sim:
    sim.run(1.0)
