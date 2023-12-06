#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse

# package version
from .__version__ import __version__
# segmentation model for wound identification
from .segmentation import wound_segmentation
# constant values
from .constants import GREEN_COLOR_CODE
from .constants import RED_COLOR_CODE
from .constants import RESET_COLOR_CODE
# pwat evaluation function
from .pwat import evaluate_PWAT_score

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

  # input image filename
  parser.add_argument(
    '--input', '-i',
    dest='filepath',
    required=False,
    action='store',
    default=None,
    type=str,
    help=(
      'Input filename or path on which load the image. '
      'Ref https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html '
      'for the list of supported formats. '
    )
  )

  # deepskin --verbose
  parser.add_argument(
    '--verbose', '-w',
    dest='verbose',
    required=False,
    action='store_true',
    default=True,
    help='Enable/Disable the code logging',
  )

  # deepskin --mask
  parser.add_argument(
    '--mask', '-m',
    dest='mask',
    required=False,
    action='store_true',
    default=False,
    help=(
      'Evaluate the semantic segmentation mask using the Deepskin model; '
      'the resulting mask will be saved to a png file in the same location '
      'of the input file'
    )
  )

  # deepskin --pwat
  parser.add_argument(
    '--pwat', '-p',
    dest='pwat',
    required=False,
    action='store_true',
    default=False,
    help='Compute the PWAT score of the given wound-image',
  )

  args = parser.parse_args()

  return args

def main ():

  # get the cmd parameters
  args = parse_args()

  if args.verbose:
    print(fr'''{GREEN_COLOR_CODE}
     _                     _    _
    | |                   | |  (_)
  __| | ___  ___ _ __  ___| | ___ _ __
 / _` |/ _ \/ _ \ '_ \/ __| |/ / | '_ \
| (_| |  __/  __/ |_) \__ \   <| | | | |
 \__,_|\___|\___| .__/|___/_|\_\_|_| |_|
                | |
                |_|
      {RESET_COLOR_CODE}''',
      end='\n',
      flush=True,
    )

  # results if version is required
  if args.version:
    # print it to stdout
    print(f'Deepskin package v{__version__}',
      end='\n', flush=True
    )
    # exit success
    exit(0)

  if args.filepath is None:
    print(
      f'{RED_COLOR_CODE}deepskin Error: the following arguments are required: --input/-i, --weight/-w{RESET_COLOR_CODE}',
      end='\n',
      flush=True,
    )
    exit(1)
  elif not os.path.exists(args.filepath):
    print((
      f'{RED_COLOR_CODE}deepskin Error:{RESET_COLOR_CODE} input image file not found\n'
      f'Given: {args.filepath}'
      ),
      end='\n',
      flush=True,
    )
    exit(1)

  if args.verbose:
    print(f'Load the input image...',
      end='',
      flush=True,
    )

  # load the image using opencv
  bgr = cv2.imread(args.filepath, cv2.IMREAD_COLOR)
  # convert the image from BGR to RGB fmt
  rgb = bgr[..., ::-1]

  if args.verbose:
    print(f' {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE}',
      end='\n',
      flush=True
    )

  if args.mask or args.pwat:
    # get the semantic segmentation mask
    mask = wound_segmentation(
      img=rgb,
      tol=0.5,
      verbose=args.verbose,
    )
    # dump the resulting mask to file

    # get the output directory
    outdir = os.path.dirname(args.filepath)
    # get the filename
    name = os.path.basename(args.filepath)
    # remove extension
    name, _ = os.path.splitext(name)
    # build the output filename
    outfile = f'{outdir}/{name}_deepskin_mask.png'
    # dump the mask in BGR fmt
    cv2.imwrite(outfile, mask[..., ::-1])

  if args.pwat:
    # compute the wound PWAT
    pwat = evaluate_PWAT_score(
      img=rgb,
      mask=mask,
      verbose=args.verbose,
    )

    print(f'{GREEN_COLOR_CODE}PWAT prediction: {pwat:.3f}{RESET_COLOR_CODE}',
      end='\n',
      flush=True,
    )


if __name__ == '__main__':

  main ()
