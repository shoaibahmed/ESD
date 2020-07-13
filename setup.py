# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup
from setuptools import find_packages

cwd = os.path.dirname(os.path.abspath(__file__))
version = '0.1.0-alpha'

def build_deps():
    version_path = os.path.join(cwd, 'esd', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))

build_deps()

with open(os.path.join(cwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [x.strip() for x in open("requirements.txt", "r").readlines()]

setup(name='esd',
      version=version,
      description='Empirical Shattering Dimension Library',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/shoaibahmed/esd',
      author='Shoaib Ahmed Siddiqui',
      author_email='shoaib_ahmed.siddiqui@dfki.de',
      license='MIT',
      keywords=['shattering_dimension'],
      install_requires=install_requires,
      packages=find_packages(),
      classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
)
