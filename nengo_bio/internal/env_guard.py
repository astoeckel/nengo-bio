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

import os

class EnvGuard:
    """
    Class used to temporarily set some environment variables to a certain value.
    In particular, this is used to set the variable OMP_NUM_THREADS to "1" for
    each of the subprocesses using cvxopt.
    """

    def __init__(self, env):
        self.env = env
        self.old_env_stack = []

    def __enter__(self):
        # Create a backup of the environment variables and write the desired
        # value.
        old_env = {}
        for key, value in self.env.items():
            if key in os.environ:
                old_env[key] = os.environ[key]
            else:
                old_env[key] = None
            os.environ[key] = str(value)

        # Push the old_env object onto the stack of old_env objects.
        self.old_env_stack.append(old_env)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get the environment variable backup from the stack.
        old_env = self.old_env_stack.pop()

        # Either delete environment variables that were not originally present
        # or reset them to their original value.
        for key, value in old_env.items():
            if not value is None:
                os.environ[key] = value
            else:
                del os.environ[key]

        return False
