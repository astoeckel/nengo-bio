#   nengo_bio -- Extensions to Nengo for more biological plausibility
#   Copyright (C) 2020  Andreas Stöckel
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

def halton(i, b):
    f = 1
    r = 0
    while i > 0:
        f = f / b
        r = r + f * (i % b)
        i = i // b
    return r

def hilbert_curve(idx, order):
    """
    Hilbert Curve Python Implementation by John Burkardt, Licensed under the
    LGPL.

    https://people.sc.fsu.edu/~jburkardt/py_src/hilbert_curve/hilbert_curve.html
    """
    def rot(n, x, y, rx, ry):
        if (ry == 0):
            if (rx == 1):
                x = n - 1 - x
                y = n - 1 - y
            t = x
            x = y
            y = t

        return x, y

    d, m = idx, order

    n = 2**m
    x = 0
    y = 0
    t = d
    s = 1
    while (s < n):
        rx = ((t // 2) % 2)
        if (rx == 0):
            ry = (t % 2)
        else:
            ry = ((t ^ rx) % 2)
        x, y = rot(s, x, y, rx, ry)
        x = x + s * rx
        y = y + s * ry
        t = (t // 4)

        s = s * 2

    return x, y

