
#   nengo_bio -- Extensions to Nengo for more biological plausibility
#   Copyright (C) 2019  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

SAMPLE_CACHE = {}

def sample_rates(dt,
                 max_rates,
                 min_max_rate,
                 max_max_rate,
                 run_single_sim,
                 estimate_input_range,
                 filter_input,
                 params_hash,
                 n_samples=1000,
                 rate_discretization=20,
                 T=10.0):
    def halton(i, b):
        f = 1
        r = 0
        while i > 0:
            f = f / b
            r = r + f * (i % b)
            i = i // b
        return r

    def discretize_rate(a):
        return int(np.ceil(a / rate_discretization) * rate_discretization)

    n_inputs = 2  # TODO

    # Create the sample cache map
    if not params_hash in SAMPLE_CACHE:
        SAMPLE_CACHE[params_hash] = {}
    cache = SAMPLE_CACHE[params_hash]

    # Discretize the minimum and maximum rates
    min_max_rate = discretize_rate(min_max_rate)
    max_max_rate = discretize_rate(max_max_rate)
    rate_range = range(max_max_rate, min_max_rate - 1, -rate_discretization)

    # Find the inputs corresponding to the threshold currents
    in_E0, _ = estimate_input_range(1)
    in_E0 = in_E0 * 0.8 # add some safety margin

    # Generate samples for each possible discretization step. Note that
    # this is completely deterministic as long as max_max_rate is the same.
    rate_map = {}
    for i, rate in enumerate(rate_range):
        # Generate a new entry in the rate_map
        rate_map[rate] = {"neurons": set(), "samples": set(), "result": None}

        # Estimate the search space
        in_E, in_I = estimate_input_range(rate)

        # Select as many points as possible from the previous set
        samples = rate_map[rate]["samples"]
        if i > 0:
            for sample in rate_map[rate + rate_discretization]["samples"]:
                if sample[0] < in_E and sample[1] < in_I:
                    samples.add(sample)

        # Fill the set of sample points up to N_SAMPLES samples
        k = int(rate / rate_discretization * n_samples)
        i = len(samples)
        while len(samples) < n_samples:
            sample = (
                in_E0 + halton(k + i, 2) * (in_E - in_E0),
                        halton(k + i, 3) *  in_I)
            i += 1
            if filter_input(*sample):
                samples.add(sample)

    # Iterate over the actual maximum rates and associate each neuron with a
    # slot in the rate map.
    for i, rate in enumerate(max_rates):
        rate = discretize_rate(rate)
        rate_map[rate]["neurons"].add(i)

    # Remove entries that do not have any neurons associated with them
    for key in [key for key in rate_map if len(rate_map[key]["neurons"]) == 0]:
        del rate_map[key]

    # Fetch all samples for which we need to run a simulation
    samples = set()
    for entry in rate_map.values():
        samples = samples.union(entry["samples"])

    # Compute the average rate for each sample by running a single-neuron
    # simulation
    # TODO: Use threading
    samples = list(samples)
    rates = np.empty(len(samples))
    ts = np.arange(0, T, dt)
    for i, sample in enumerate(samples):
        if sample in cache:
            rates[i] = SAMPLE_CACHE[params_hash][sample]
            continue
        out = np.zeros_like(ts)
        run_single_sim(i, out, *sample)
        times = ts[out > 0]
        if (len(times) > 1):
            rates[i] = 1. / np.mean((times[1:] - times[:-1]))
        else:
            rates[i] = 0.0
        cache[sample] = rates[i]

    # Write the measured rates back to the rate_map
    for rate, entry in rate_map.items():
        entry_result = np.empty((len(entry["samples"]), n_inputs + 1))
        j = 0
        for i, sample in enumerate(samples):
            if sample in entry["samples"]:
                entry_result[j, :n_inputs] = sample
                entry_result[j, n_inputs:] = rates[i]
                j += 1
        entry["result"] = entry_result

    return rate_map

def fit_model_weights_two_comp(Js, samples, valid):
    from .qp_solver import solve_linearly_constrained_quadratic_loss

    # Assemble the quadratic loss function
    scale_cur = 1e9
    scale_cond = 1e6
    n_samples = samples.shape[1]
    n_valid = int(np.sum(valid))

    n_cstr = n_valid
    n_vars = 5
    C, d = np.zeros((n_cstr, n_vars)), np.zeros(n_cstr)
    C[:n_valid, 0] = 1
    C[:n_valid, 1] = samples[1, valid] * scale_cond
    C[:n_valid, 2] = -Js[valid] * scale_cur
    C[:n_valid, 3] = -Js[valid] * samples[0, valid] * scale_cur * scale_cond
    C[:n_valid, 4] = -Js[valid] * samples[1, valid] * scale_cur * scale_cond
    d[:n_valid] = -samples[0, valid] * scale_cond

    # Make sure a0, a1, a2 are greater than zero
    G, h = np.zeros((3, n_vars)), np.zeros(3)
    G[0, 2] = -1
    G[1, 3] = -1
    G[2, 4] = -1

    # Solve for model weights
    ws = solve_linearly_constrained_quadratic_loss(C=C, d=d, G=G, h=h, tol=1e-12)[:, 0]

    # Rescale the output
    ws = np.array((
        ws[0] / scale_cond,
        1.0,
        ws[1],
        ws[2] * scale_cur / scale_cond,
        ws[3] * scale_cur,
        ws[4] * scale_cur,
    ))

    return ws


def tune_two_comp_model_weights(dt, max_rates, min_max_rate, max_max_rate,
                                run_single_sim, estimate_input_range,
                                filter_input, lif_rate_inv, params_hash):
    # Generate sample points and measure the
    rate_map = sample_rates(dt, max_rates, min_max_rate, max_max_rate,
                            run_single_sim, estimate_input_range, filter_input, params_hash)

    # Compute the per-neuron model weights
    n_neurons = len(max_rates)
    ws = np.zeros((n_neurons, 6), dtype=np.float64)

    for rate, entry in rate_map.items():
        # Solve for model weights
        result = entry["result"]
        samples, rates = result[:, 0:2], result[:, 2]
        Js = lif_rate_inv(rates)
        valid = rates > 10 # Ignore points with a small rate
        w = fit_model_weights_two_comp(Js, samples.T, valid)

        # Store per-neuron weights in the corresponding slot
        for i in entry["neurons"]:
            ws[i] = w

#        def H(w, gE, gI):
#            return (w[0] + w[1] * gE + w[2] * gI) / (w[3] + w[4] * gE + w[5] * gI)

#        import matplotlib.pyplot as plt
#        import matplotlib
#        matplotlib.use('Agg')

#        gE, gI = samples.T
#        gEs = np.linspace(np.min(gE), np.max(gE))
#        gIs = np.linspace(np.min(gI), np.max(gI))
#        gEss, gIss = np.meshgrid(gEs, gIs)
#        Js_model = H(w, gEss, gIss)

#        fig, ax = plt.subplots()
#        sc = ax.scatter(gE * 1e9, gI * 1e9, c=rates)
#        ax.contour(gEs * 1e9, gIs * 1e9, Js_model, levels=np.linspace(0, np.max(Js), 10), colors=['white'])
#        ax.contour(gEs * 1e9, gIs * 1e9, Js_model, levels=np.linspace(0, np.max(Js), 10), colors=['k'], linestyles='--')
#        ax.set_xlim(0, np.max(gE) * 1e9)
#        ax.set_ylim(0, np.max(gI) * 1e9)
#        ax.set_xlabel("Excitatory input")
#        ax.set_ylabel("Inhibitory input")
#        plt.colorbar(sc)

#        fig.savefig('{}_{}.png'.format(params_hash, rate), dpi=300)

    return ws

