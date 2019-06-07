#!/usr/bin/env python3

import nengo
import nengo_bio as bio
import numpy as np

with nengo.Network(seed=4819) as model:
    inp_a = nengo.Node(0)
    inp_b = nengo.Node(0)
    mul_a_b = nengo.Node(lambda t, x: x[0] * x[1], size_in=2)
    nengo.Connection(inp_a, mul_a_b[0])
    nengo.Connection(inp_b, mul_a_b[1])

    ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8,
                         eval_points=nengo.dists.Uniform(0, 1))
    ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.8,
                         eval_points=nengo.dists.Uniform(0, 1))
    ens_c = bio.Ensemble(n_neurons=103, dimensions=1,
                         neuron_type=bio.neurons.TwoCompLIF(),
                         max_rates=nengo.dists.Uniform(50, 100),
                         intercepts=nengo.dists.Uniform(-0.95, 0.95))
    ens_d = bio.Ensemble(n_neurons=200, dimensions=2, p_exc=0.8,
                         max_rates=nengo.dists.Uniform(50, 100),
                         eval_points=np.array((
                             np.random.uniform(0, 1, 750),
                             np.random.uniform(0, 1, 750),
                         )).T)
    ens_e = bio.Ensemble(n_neurons=100, dimensions=1, p_exc=0.8,
                         max_rates=nengo.dists.Uniform(50, 100))

    nengo.Connection(inp_a, ens_a)
    nengo.Connection(inp_b, ens_b)

    bio.Connection((ens_a, ens_b), ens_c,
                    function=lambda x: x[0] * x[1],
                    solver=bio.solvers.QPSolver(relax=True, reg=1e-5))
    bio.Connection((ens_a, ens_b), ens_d)
    bio.Connection(ens_d, ens_e, function=lambda x: x[0] * x[1])

with nengo.Simulator(model) as sim:
    sim.run(1.0)
