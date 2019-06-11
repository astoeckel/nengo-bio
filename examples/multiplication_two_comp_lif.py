#!/usr/bin/env python3

import nengo
import nengo_bio as bio
import numpy as np

bio.set_defaults()

xs = np.linspace(0, 1, 16)
ys = np.linspace(0, 1, 16)
xss, yss = np.meshgrid(xs, ys)

f = np.prod

with nengo.Network(seed=4589) as model:
    inp_a = nengo.Node(0)
    inp_b = nengo.Node(0)

    # Compute the refernce input
    ref = nengo.Node(lambda t, x: f(x), size_in=2)
    nengo.Connection(inp_a, ref[0])
    nengo.Connection(inp_b, ref[1])

    ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.7,
                         eval_points=xss.flatten().reshape(-1, 1))

    ens_b = bio.Ensemble(n_neurons=102, dimensions=1, p_exc=0.7,
                         eval_points=yss.flatten().reshape(-1, 1))

    ens_c = bio.Ensemble(n_neurons=103, dimensions=1,
                         neuron_type=bio.neurons.LIF())

    ens_d = bio.Ensemble(n_neurons=103, dimensions=1,
                         neuron_type=bio.neurons.TwoCompLIF())

    nengo.Connection(inp_a, ens_a)
    nengo.Connection(inp_b, ens_b)

    bio.Connection((ens_a, ens_b), ens_c,
                   function=f,
                   solver=bio.solvers.QPSolver(relax=False, reg=1e-3),
                   synapse_exc=100e-3,
                   synapse_inh=100e-3)
    bio.Connection((ens_a, ens_b), ens_d,
                   function=f,
                   solver=bio.solvers.QPSolver(relax=False, reg=1e-5),
                   synapse_exc=100e-3,
                   synapse_inh=100e-3)

    tar = nengo.Node(size_in=3)
    nengo.Connection(ens_c, tar[0])
    nengo.Connection(ens_d, tar[1])
    nengo.Connection(ref, tar[2])

with nengo.Simulator(model, progress_bar=None) as sim:
    sim.run(1.0)
