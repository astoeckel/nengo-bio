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

from nengo_bio.internal.sequences import hilbert_curve

import nengo.params
import nengo.dists

class NeuralSheetDist(nengo.dists.Distribution):
    """
    Distribution arranging neurons either along a line or 
    """

    dimensions = nengo.params.IntParam("dimensions", readonly=True)
    x0 = nengo.params.NumberParam("x0", readonly=True)
    y0 = nengo.params.NumberParam("y0", readonly=True)
    x1 = nengo.params.NumberParam("x1", readonly=True)
    y1 = nengo.params.NumberParam("y1", readonly=True)

    def __init__(self, dimensions=2, x0=-1.0, y0=-1.0, x1=1.0, y1=1.0):
        super().__init__()
        self.dimensions = dimensions
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def _jitter_and_normalise(self, xs, x0, x1, rng):
        N = xs.shape[0]
        return (x1 - x0) * (xs + rng.uniform(0.0, 1.0, xs.shape)) / N + x0

    def _sample_1d(self, n, rng):
        return self._jitter_and_normalise(
            np.arange(n, dtype=np.float), self.x0, self.x1, rng).reshape(-1, 1)

    def _sample_2d(self, n, rng):
        # Comptue the order of the hilbert curve that is being used to sample
        # the neural locations
        order = int(np.ceil(np.log2(np.ceil(np.sqrt(max(1, n))))))
        N = 2**order

        # Compute the hilbert curve coordinates
        xs, ys = np.zeros((2, N, N))
        for i in range(0, N):
            for j in range(0, N):
                k = i * N + j
                xs[i, j], ys[i, j] = hilbert_curve(k, order)

        # Add some random jitter (move each point by at most one cell)
        xs = self._jitter_and_normalise(xs, self.x0, self.x1, rng)
        ys = self._jitter_and_normalise(ys, self.y0, self.y1, rng)

        # Pick n samples
        idcs = np.linspace(0, N * N - 1, n, dtype=int)
        return np.array((xs.flatten()[idcs], ys.flatten()[idcs])).T


    def _sample_shape(self, n, d=None):
        if (d == 1) or (d == 2):
            return (n, d) # 1D or 2D
        return (n, self.dimensions) # Default to 2D


    def sample(self, n, d=None, rng=np.random):
        # Make sure the given dimensionality parameter is correct
        d = self.dimensions if d is None else d

        # Either sample from a one- or two-dimensional distribution
        if d == 1:
            return self._sample_1d(n, rng)
        elif d == 2:
            return self._sample_2d(n, rng)
        else:
            raise ValueError("NeuralSheetDist must be one- or two-dimensional")

