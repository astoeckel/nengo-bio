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

# Issue a warning if OMP_NUM_THREADS is not set
def _warn_omp_num_threads():
    import os
    import warnings
    var = "OMP_NUM_THREADS"
    if ((not var in os.environ) or (os.environ[var] != "1")):
        warnings.warn(
            "The environment variable OMP_NUM_THREADS is not set to \"1\". "
            "This will result in reduced performance when solving for "
            "neuron weights.",
            RuntimeWarning)
_warn_omp_num_threads()

# Set the package name
name = "nengo_bio"

# Expose commonly used objects
from nengo_bio.connection import Connection
from nengo_bio.ensemble import Ensemble
from nengo_bio.common import Excitatory, Inhibitory

# Implicitly register the builder components
import nengo_bio.builder as builder
