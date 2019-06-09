#!/usr/bin/env python3

import nengo
import nengo_bio as bio
import numpy as np

with nengo.Network(seed=1556) as model:
    inp_a = nengo.Node(0)
    inp_b = nengo.Node(0)
    mul_ab = nengo.Node(lambda t, x: x[0] * x[1], size_in=2)
    nengo.Connection(inp_a, mul_ab[0])
    nengo.Connection(inp_b, mul_ab[1])

    ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8,
                         max_rates=nengo.dists.Uniform(50, 100),
                         eval_points=nengo.dists.Uniform(0, 1))
    ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.8,
                         max_rates=nengo.dists.Uniform(50, 100),
                         eval_points=nengo.dists.Uniform(0, 1))
    ens_c = bio.Ensemble(n_neurons=103, dimensions=1,
                         #neuron_type=bio.neurons.LIF(),
                         max_rates=nengo.dists.Uniform(50, 100),
                         intercepts=nengo.dists.Uniform(-0.95, 0.95))
    ens_d = bio.Ensemble(n_neurons=103, dimensions=1,
                         neuron_type=bio.neurons.TwoCompLIF(),
                         max_rates=nengo.dists.Uniform(50, 100),
                         intercepts=nengo.dists.Uniform(-0.95, 0.95))

    nengo.Connection(inp_a, ens_a)
    nengo.Connection(inp_b, ens_b)

    bio.Connection((ens_a, ens_b), ens_c, 
                   function=lambda x: x[0] * x[1],
                   solver=bio.solvers.QPSolver(relax=True))
    bio.Connection((ens_a, ens_b), ens_d, 
                   function=lambda x: x[0] * x[1],
                   solver=bio.solvers.QPSolver(relax=True, reg=1e-6))

    tar_cd = nengo.Node(size_in=3, size_out=3)
    nengo.Connection(ens_c, tar_cd[0])
    nengo.Connection(ens_d, tar_cd[1])
    nengo.Connection(mul_ab, tar_cd[2])

with nengo.Simulator(model, progress_bar=None) as sim:
    sim.run(1.0)
