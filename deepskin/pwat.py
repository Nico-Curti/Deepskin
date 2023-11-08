#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np # just for typehints
from time import time as now

# image pre/post processing algorithms
from .imgproc import imfill
from .imgproc import get_perilesion_mask
# image feature functions
from .features import evaluate_features
# pwat evaluation coefficients
from .constants import Deepskin_CENTER
from .constants import Deepskin_SCALE
from .constants import Deepskin_PWAT_PARAMS
from .constants import Deepskin_PWAT_BIAS
from .constants import CRLF
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'evaluate_PWAT_score'
]


def evaluate_PWAT_score (img : np.ndarray,
                         mask : np.ndarray,
                         ksize : tuple = (20, 20),
                         verbose : bool = False
                         ) -> float :
  '''
  Evaluate the PWAT score as combination of features
  extracted from the wound and peri-wound areas.

  Parameters
  ----------
    img : np.ndarray
      Input image to analyze in RGB fmt

    mask : np.ndarray
      Semantic mask of the image

    ksize : tuple
      Kernel dimension for the mask processing

    verbose : bool (default := False)
      Enable/Disable the logging of the steps

  Returns
  -------
    pwat : float
      Photographic Wound Assessment Tool score for the image
  '''
  tic = now()
  step = 'Perform the PWAT estimation... '

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[1/4] extract the peri-wound mask{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )

  # un-pack the semantic mask into its components
  wound_mask, body_mask, bg_mask = cv2.split(mask)

  # get the peri-wound mask
  periwound_mask = get_perilesion_mask(
    mask=wound_mask,
    ksize=ksize,
  )
  # correct the peri-wound mask according to the body
  periwound_mask = cv2.bitwise_and(
    periwound_mask,
    periwound_mask,
    mask=imfill(body_mask | wound_mask)
  )

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[2/4] evaluate the wound features{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  # evaluate the wound features
  wound_features = evaluate_features(
    img=img,
    mask=wound_mask,
    prefix='w_'
  )
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[3/4] evaluate the peri-wound features{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  # evaluate the peri-wound features
  periwound_features = evaluate_features(
    img=img,
    mask=periwound_mask,
    prefix='p_'
  )

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[4/4] compute the PWAT regression score{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )

  # standardize the wound features
  wound_features = {k : (v - Deepskin_CENTER.get(k, 0.0)) / Deepskin_SCALE.get(k, 1.0)
    for k, v in wound_features.items()
  }

  # standardize the peri-wound features
  periwound_features = {k : (v - Deepskin_CENTER.get(k, 0.0)) / Deepskin_SCALE.get(k, 1.0)
    for k, v in periwound_features.items()
  }

  # evaluate the PWAT contribution of the wound
  pwat_wound = sum([v * Deepskin_PWAT_PARAMS.get(k, 0.0)
    for k, v in wound_features.items()
  ])

  # evaluate the PWAT contribution of the peri-wound
  pwat_periwound = sum([v * Deepskin_PWAT_PARAMS.get(k, 0.0)
    for k, v in periwound_features.items()
  ])

  # get the final PWAT score
  pwat = pwat_wound + pwat_periwound + Deepskin_PWAT_BIAS

  toc = now()

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE}  ({toc - tic:.3f} sec)                              ',
      end='\n',
      flush=True,
    )

  return pwat
