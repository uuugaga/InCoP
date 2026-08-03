# -*- coding: utf-8 -*-

from os.path import dirname, realpath

from setuptools import find_packages, setup

from opencood.version import __version__


def _read_requirements_file():
    """Return the entries in requirements.txt."""
    req_file_path = "%s/requirements.txt" % dirname(realpath(__file__))
    with open(req_file_path) as requirements_file:
        return [line.strip() for line in requirements_file]


setup(
    name="InCoP",
    version=__version__,
    packages=find_packages(),
    license="Academic Software License",
    author="InCoP Authors",
    description=(
        "Supplementary code for indoor collaborative 3D object detection "
        "with complementary ground-robot views"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=[],
)
