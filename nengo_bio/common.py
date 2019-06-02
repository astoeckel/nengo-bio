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

def steal_param(param_name, kw_args, default=None):
    if param_name in kw_args:
        value = kw_args[param_name]
        del kw_args[param_name]
        return value
    else:
        return default

class SynapseType:
    """
    The SynapseType class can be used to mark neurons as either excitatory or
    inhibitory.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

Excitatory = SynapseType("excitatory")
Inhibitory = SynapseType("inhibitory")

