# NengoBio
## Extensions facilitating the description of biological constraints in Nengo networks

**Warning:** This project (including its name) is work-in progress and things are still subject to change.

## What is NengoBio?

*NengoBio* is an add-on library for the [Nengo](https://nengo.ai/) spiking neural network simulator and description language. Nengo is used to construct detailed models of neurobiological systems. Nengo, as well as the underlying NEF, have some restrictions that can be seen as limitations on their biological plausibility. *NengoBio* lifts some of these restrictions by implementing the following:

* Support distinction between excitatory and inhibitory neurons (Dale's Principle) (*Fully implemented*)
* Elimination of the bias current (*Fully implemented*)
* Support for dendritic computation (*Partially implemented*)
* Support for conductance-based synapses as well as neurons with arbitrary passive dendritic trees (*Planned*)

## Installing NengoBio

**Dependencies:** *NengoBio* requires Python 3 and depends on `numpy>=1.16.3`, `scipy>=1.2.0`, `cvxopt>=1.2.2`, `nengo>=3.0.0.dev0`.

Clone this repository by running
```bash
git clone https://github.com/astoeckel/nengo_bio
```
You can then install the package by running the following inside the `nengo_bio` repository
```bash
pip3 install -e .
```
This will automatically install all dependencies. Note that *NengoBio* currently requires the most recent development version of *Nengo*, which has to be installed separately.

## Using NengoBio

Using *NengoBio* is quite simple if you know how to use the original Nengo. Just add the following to your set of imports
```
import nengo_bio as bio
```
and replace `nengo.Ensemble` with `bio.Ensemble` and `nengo.Connection` with `bio.Connection` where applicable.

### The `bio.Ensemble` class

The `bio.Ensemble` class acts like a normal Nengo ensemble but has two additional parameters: `p_exc` and `p_inh`. These parameters describe the relative number of excitatory/inhibitory neurons in the population. Note that `p_exc` and `p_inh` have to sum to one.

**Note:** Neurons will be assigned a synapse type at build time. If any of `p_exc` or `p_inh` is set, each neuron will either be excitatory or inhibitory. Without `p_exc` and `p_inh`, the ensemble will behave just like a normal Nengo ensemble.

**Example:** An ensemble exclusively consisting of excitatory neurons
```py
ens_exc = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=1.0)
```

**Example:** An ensemble exclusively consisting of inhibitory neurons
```py
ens_inh = bio.Ensemble(n_neurons=101, dimensions=1, p_inh=1.0)
```

**Example:** An ensemble consisting of 80% excitatory and 20% inhibitory neurons (both lines are equivalent):
```py
ens_mix = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8)
ens_mix = bio.Ensemble(n_neurons=101, dimensions=1, p_inh=0.2)
```

### The `bio.Connection` class

A `bio.Connection` connection connects *n*-pre ensembles to a single target ensemble. It will automatically account for the synapse type assigned to each neuron.

**Notable Parameters:**

* `pre`: This can be either a single pre-population or a tuple of pre-populations. The dimensions of the values represented by the pre-populations will be stacked.

* `decode_bias` (default `True`): if `True` the post-neuron bias current will be decoded from the pre-population instead of being assumed constant. Set this to `False` for any but the first `bio.connection` targeting the same post population.

* `solver` (default `QPSolver()`): an `ExtendedSolver` instance from `nengo_bio.solvers`. `ExtendedSolvers` can solve for currents and take neuron parameters into account.

**Example:** Simple communication channel between `ens_a` and `ens_b` taking neuron/synapse types into account and decoding the bias current:
```py
bio.Connection(ens_a, ens_b)
```

**Example:** 2D communication channel where `ens_a`, `ens_b` represent a one-dimensional value and `ens_c` represents a two-dimensional value.
```py
bio.Connection((ens_a, ens_b), ens_c)
```

**Example:** Linear "Dendritic Computation"
```py
bio.Connection((ens_a, ens_b), ens_c, function=lambda x: np.mean(x))
```

## Citing this work

The techniques used in this library are described in more detail in this arXiv preprint: https://arxiv.org/abs/1904.11713

```
@misc{stoeckel2019passive,
    author = {Andreas Stöckel and Chris Eliasmith},
    title = {Passive nonlinear dendritic interactions as a general computational resource in functional spiking neural networks},
    year = {2019},
    eprint = {arXiv:1904.11713},
}
```
