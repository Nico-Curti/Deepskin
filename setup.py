#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

def dump_version_file (
    author : str,
    email : str,
    version : str
  ):
  '''
  Dump the __version__.py file as python script

  Parameters
  ----------
    author: str
      List of author/maintainer names comma separated

    email: str
      List of author/maintainer emails comma separated

    version: str
      String of the version code as major.minor.revision
  '''

  script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__  = '{author}'
__email__ = '{email}'

__version__ = '{version}'
'''

  filename = os.path.abspath(os.path.dirname(__file__))
  filename = f'{filename}/deepskin/__version__.py'

  with open(filename, 'w') as fp:
    fp.write(script)


PACKAGE_NAME = 'deepskin'
AUTHOR = 'Nico Curti'
EMAIL = 'nico.curti2@unibo.it'
REQUIRES_PYTHON = '>=3.5'
PACKAGE_VERSION = '0.0.1'
DESCRIPTION = 'Wound analysis using smartphone images'
URL = 'https://github.com/Nico-Curti/Deepskin'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = EMAIL
DOWNLOAD_URL = URL

# dump the version file
dump_version_file(
  author=MAINTAINER,
  email=MAINTAINER_EMAIL,
  version=PACKAGE_VERSION
)

setup(
  name=PACKAGE_NAME,
  version=PACKAGE_VERSION,
  description=DESCRIPTION,
  author=AUTHOR,
  author_email=EMAIL,
  python_requires=REQUIRES_PYTHON,
  install_requires=[],
  url=URL,
  download_url=DOWNLOAD_URL,
  setup_requires=[],
  packages=[
    PACKAGE_NAME,
  ],
  package_data={
    PACKAGE_NAME: [],
  },
  include_package_data=True,
  platforms='any',
  classifiers=[
    'Natural Language :: English',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: Implementation :: CPython',
    'Programming Language :: Python :: Implementation :: PyPy'
  ],
  entry_points={'console_scripts': [
    'deepskin = deepskin.__main__:main',
    ],
  },
)
