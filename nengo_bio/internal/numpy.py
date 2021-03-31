#   nengo_bio -- Extensions to Nengo for more biological plausibility
#   Copyright (C) 2021  Andreas Stöckel
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

import numpy  as np

def is_array_like(obj):
    """
    Checks whether an object can be converted into a numpy array.
    This is kind of a brute-force approach because I can't copy
    code from nengo...
    """
    try:
       np.asarray(obj, dtype=float)
       return True
    except TypeError:
       return False
