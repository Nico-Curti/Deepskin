#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import argparse

from .__version__ import __version__

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

def parse_args ():

  description = ('deepskin library - '
    'Wound analysis using smartphone images'
  )

  # global sofware information
  parser = argparse.ArgumentParser(
    prog='deepskin',
    argument_default=None,
    add_help=True,
    prefix_chars='-',
    allow_abbrev=True,
    exit_on_error=True,
    description=description,
    epilog=f'Deepskin Python package v{__version__}'
  )

  # deepskin --version
  parser.add_argument(
    '--version', '-v',
    dest='version',
    required=False,
    action='store_true',
    default=False,
    help='Get the current version installed',
  )

  args = parser.parse_args()

  return args


def main ():

  # get the cmd parameters
  args = parse_args()

  # results if version is required
  if args.version:
    # print it to stdout
    print(f'Deepskin package v{__version__}',
      end='\n', file=sys.stdout, flush=True
    )
    # exit success
    exit(0)

  # exit success
  exit(0)

if __name__ == '__main__':

  main ()
