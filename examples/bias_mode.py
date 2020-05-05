import nengo
import nengo_bio as bio
import numpy as np

f1 = lambda t: np.sin(t)
with nengo.Network(seed=5892) as model:
    inp_a = nengo.Node(f1)

    ens_a = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=1.0)
    ens_b = bio.Ensemble(n_neurons=102, dimensions=1)
    ens_c = bio.Ensemble(n_neurons=102, dimensions=1)
    ens_d = bio.Ensemble(n_neurons=102, dimensions=1)
    ens_e = bio.Ensemble(n_neurons=102, dimensions=1)

    nengo.Connection(inp_a, ens_a)

    bio.Connection(ens_a, ens_b, bias_mode=bio.Decode)
    bio.Connection(ens_a, ens_c, bias_mode=bio.JBias)
    bio.Connection(ens_a, ens_d, bias_mode=bio.ExcJBias)
    bio.Connection(ens_a, ens_e, bias_mode=bio.InhJBias)
