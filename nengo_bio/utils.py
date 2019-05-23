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

def steal_param(target, param_name, kw_args, default=None):
    """
    Helper function used to extract additional keyword arguments passed to
    complex parent class constructors.

    target: object to which an attribute with the name 'param_name' will
        be added.
    param_name: name of the parameter that will be removed from the kw_args
        and the attribute that will be added to the target object.
    kw_args: dictionary from which the key 'param_name' will be removed if it
        exists.
    """

    if param_name in kw_args:
        setattr(target, param_name, kw_args[param_name])
        del kw_args[param_name]
    else:
        setattr(target, param_name, default)

