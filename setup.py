# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

from setuptools import find_packages, setup


def get_install_requirements():
    requirements = []
    requirements_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "requirements.txt"
    )
    with open(requirements_file) as f_req:
        for line in f_req:
            line = line.strip()
            if not line.startswith("#") and not line.startswith("-f") and len(line) > 0:
                requirements.append(line)

    return requirements


setup(
    name="diht",
    py_modules=["diht"],
    version="1.0",
    description="",
    author="Meta AI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=get_install_requirements(),
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
