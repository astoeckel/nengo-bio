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

from nengo_bio.internal.env_guard import EnvGuard

def env_process(i):
    import os
    return dict(os.environ)

def do_test_env_guard(env):
    import multiprocessing
    with EnvGuard(env) as guard:
        p = multiprocessing.Pool(1)
        return p.map(env_process, (0,))[0]

def test_env_guard():
    import os

    # Make sure the test key is not in the environment
    env = do_test_env_guard({})
    assert not "_NENGO_BIO_TEST_" in env
    assert not "_NENGO_BIO_TEST_" in os.environ

    # Try to write the test key to the environment
    env = do_test_env_guard({"_NENGO_BIO_TEST_": "foo"})
    assert "_NENGO_BIO_TEST_" in env
    assert env["_NENGO_BIO_TEST_"] == "foo"
    assert not "_NENGO_BIO_TEST_" in os.environ

    # Try to write the test key to the environment
    os.environ["_NENGO_BIO_TEST_2_"] = "bar"
    env = do_test_env_guard({"_NENGO_BIO_TEST_": "foo", "_NENGO_BIO_TEST_2_": "foo2"})
    assert "_NENGO_BIO_TEST_" in env
    assert env["_NENGO_BIO_TEST_"] == "foo"
    assert not "_NENGO_BIO_TEST_" in os.environ

    assert "_NENGO_BIO_TEST_2_" in env
    assert env["_NENGO_BIO_TEST_2_"] == "foo2"
    assert os.environ["_NENGO_BIO_TEST_2_"] == "bar"

