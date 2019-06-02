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

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nengo_bio',
    packages=[
        'nengo_bio'
    ],
    version='0.1',
    author='Andreas Stöckel',
    author_email='astoecke@uwaterloo.ca',
    description='Dendritic Computation Primitives for Nengo',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/astoeckel/nengo-bio',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=[
        "cvxopt>=1.2.2",
        "nengo>=2.8",
        "numpy>=1.16.3",
        "scipy>=1.2.0",
    ],
)

