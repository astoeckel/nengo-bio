![NengoBio Logo](doc/logo.png)

# NengoBio ‒ Biologically (more) plausible Nengo models

> **⚠ Warning:** This project is work-in progress. Everything described here, including the name of the project and the API, is subject to change.

*NengoBio* is an add-on library for the [Nengo](https://nengo.ai/) spiking neural network simulator. Nengo is used by scientists to construct detailed models of neurobiological systems. However, Nengo and, to some degree, the underlying [Neural Engineering Framework](http://compneuro.uwaterloo.ca/research/nef.html), have restrictions that limit the biological plausibility of the created networks. *NengoBio* lifts some of these restrictions by implementing the following:

* **Dale's Principle** (:ballot_box_with_check: *Fully implemented*)<br>
  While it is possible to work around this limitation, Nengo usually does not explicitly mark neurons as excitatory or inhibitory. This means that a single can connect to post-neurons both excitatorily and inhibitorily, depending on the sign of the weights computed by of the weight solver. *NengoBio* marks neurons as either excitatory or inhibitory and accounts for this while solving for connection weights.
* **Bias current elimination** (:ballot_box_with_check: *Fully implemented*)<br>
  The Neural Engineering Framework assumes that each neuron is connected to a constant bias current source. This bias current is used to diversify the avilable neuron tuning curves, yet is a little unrealistic from a biological perspective. *NengoBio* eliminates the bias current by solving for synaptic weights in current space, effectively decoding the bias current from the pre-population state.
* **Support for dendritic computation** (*Partially implemented*)<br>
  Dendritic nonlinearities play a key role in information processing in central nervous systems and can be systematically exploited to perfrom nonlinear, multivariate computations. *NengoBio* adds support for dendritic computation to Nengo by allowing an arbitrary number of neuron ensembles as pre-objects in a connection.
* **Support for conductance-based synapses as well as neurons with arbitrary passive dendritic trees** (*Planned*)
  Dendritic computation relies on nonlinear effects in the dendritic tree and the specific tree topology. *NengoBio* adds support for arbitrary passive multicompartment neuron models to Nengo.

## Installing NengoBio

**Dependencies:** *NengoBio* requires Python 3 and depends on `numpy>=1.16.3`, `scipy>=1.2.0`, `cvxopt>=1.2.2`, `nengo>=3.0.0.dev0`.

Clone this repository by running
```sh
git clone https://github.com/astoeckel/nengo_bio
```
You can then install the package by running the following inside the `nengo_bio` repository
```sh
pip3 install -e .
```
This will automatically install all dependencies. Note that *NengoBio* currently requires the most recent development version of *Nengo*, which has to be installed separately.

## Using NengoBio

### [📝 See the example notebook](https://github.com/astoeckel/nengo_bio/blob/master/examples/nengo_bio_examples.ipynb)

Assuming that you know how to use Nengo, using *NengoBio* should be quite simple. Just add the following to your list of imports
```py
import nengo_bio as bio
```
and replace `nengo.Ensemble` with `bio.Ensemble` and `nengo.Connection` with `bio.Connection` where applicable.

### The `bio.Ensemble` class

The `bio.Ensemble` class acts like a normal Nengo ensemble but has two additional parameters: `p_exc` and `p_inh`. These parameters describe the relative number of excitatory/inhibitory neurons in the population. Note that `p_exc` and `p_inh` have to sum to one. These parameters are only relevant if an ensemble is a pre-object.

**Note:** Neurons will be assigned a synapse type at build time. If any of `p_exc` or `p_inh` is set, each neuron will either be excitatory or inhibitory. Without `p_exc` and `p_inh`, the ensemble will behave just like a normal Nengo ensemble.

**Warning:** `bio.Ensemble` can be used in conjunction with the normal `nengo.Connection` class. The excitatory/inhibitory nature of the neurons in a `bio.Ensemble` will only be taken into account when using `bio.Connection` (see below).

### Examples

**Examples 1:** An ensemble exclusively consisting of excitatory neurons
```py
ens_exc = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=1.0)
```
**Examples 2:** An ensemble exclusively consisting of inhibitory neurons
```py
ens_inh = bio.Ensemble(n_neurons=101, dimensions=1, p_inh=1.0)
```
**Examples 3:** An ensemble consisting of 80% excitatory and 20% inhibitory neurons (both lines are equivalent):
```py
ens_mix = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8)
ens_mix = bio.Ensemble(n_neurons=101, dimensions=1, p_inh=0.2)
```

### The `bio.Connection` class

A `bio.Connection` connection connects *n*-pre ensembles to a single target ensemble. It will automatically account for the synapse type assigned to each neuron.

### Notable Parameters

* `pre`: This can be either a single pre-population or a tuple of pre-populations. The dimensions of the values represented by the pre-populations will be stacked.

* `decode_bias` (default `True`): if `True` the post-neuron bias current will be decoded from the pre-population instead of being assumed constant. Set this to `False` for any but the first `bio.connection` targeting the same post population.

* `solver` (default `QPSolver()`): an `ExtendedSolver` instance from `nengo_bio.solvers`. `ExtendedSolvers` can solve for currents and take neuron parameters into account.

### Examples

**Example 1:** Simple communication channel between `ens_a` and `ens_b` taking neuron/synapse types into account and decoding the bias current:
```py
bio.Connection(ens_a, ens_b)
```

**Example 2:** 2D communication channel where `ens_a`, `ens_b` represent a one-dimensional value and `ens_c` represents a two-dimensional value.
```py
bio.Connection((ens_a, ens_b), ens_c)
```

**Example 3:** Linear "Dendritic Computation"
```py
bio.Connection((ens_a, ens_b), ens_c, function=lambda x: np.mean(x))
```

## Citing

The techniques used in this library are described in more detail in this arXiv preprint: https://arxiv.org/abs/1904.11713. We would appreciate it if you could cite this paper in case you use this library in a published model.

```bib
@misc{stoeckel2019passive,
    author = {Andreas Stöckel and Chris Eliasmith},
    title = {Passive nonlinear dendritic interactions as a general computational resource in functional spiking neural networks},
    year = {2019},
    eprint = {arXiv:1904.11713},
}
```

## License

```
nengo_bio -- Extensions to Nengo for more biological plausibility
Copyright (C) 2019  Andreas Stöckel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
