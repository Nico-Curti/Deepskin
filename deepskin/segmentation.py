#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# segmentation model
from .model import deepskin_model
from .model import MODEL_CHECKPOINT
# download model weights
from .checkpoints import download_model_weights

import cv2
import numpy as np
from time import time as now

# constant values
from .constants import CRLF
from .constants import IMG_SIZE
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'wound_segmentation'
]

def wound_segmentation (img : np.ndarray,
                        tol : float = 0.5,
                        verbose : bool = False
                       ) -> np.ndarray :
  '''
  Perform the semantic image segmentation using the
  Deepskin semantic model.

  Parameters
  ----------
    img : np.ndarray
      Input image to analyze in RGB format

    tol : float (default := 0.5)
      Threshold to apply on the resulting mask for the
      output binarization

    verbose : bool (default := False)
      Enable/Disable the logging of the steps

  Returns
  -------
    pred : np.ndarray
      Output image mask obtained by the model, in which
      the semantic meaning is organized as: background
      (channel 0), body (channel 1), wound (channel 2)
  '''

  tic = now()
  step = 'Perform the semantic image segmentation... '

  if verbose:
    print(f'{step}',
      end='\r',
      flush=True,
    )

  # build the model for the semantic segmentation
  model = deepskin_model(
    verbose=False
  )

  # load the appropriated weights
  local = os.path.dirname(os.path.abspath(__file__))
  # build the weights filepath
  weightspath = os.path.join(
    local,
    '..',
    'checkpoints',
    'efficientnetb3_deepskin_semantic.h5'
  )
  # if the weights file does not exists
  if not os.path.exists(weightspath):
    download_model_weights(
      Id=MODEL_CHECKPOINT,
      model_name = 'efficientnetb3_deepskin_semantic'
    )

  # load the weights
  model.load_weights(weightspath)

  # get the model input shape
  _, h, w, c = model.input.shape

  # pre-process the input

  # resize the image into the shape required by the model
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[1/3] pre-process the input image{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  resized = cv2.resize(
    img,
    dsize=(h, w),
    interpolation=cv2.INTER_CUBIC
  )
  # convert the image into floating-point values
  resized = np.float32(resized)
  # normalize the image into [0, 1] range
  resized *= 1. / 255
  # extend the dimensionality of the input array
  # to the [batch, h, w, c] format
  resized = resized.reshape(1, *resized.shape)

  # apply the segmentation model

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[2/3] apply the segmentation model{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  # apply the model to get the prediction
  pred = model.predict(resized, verbose=0)
  # remove useless dimensions from the image
  pred = np.squeeze(pred)
  # filter the mask output to binary format
  pred = np.where(pred > tol, 255, 0)
  # convert the mask into uint8 fmt
  pred = np.uint8(pred)

  # post-process the mask

  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[3/3] post-process the segmentation mask{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  # resize the output mask to the same
  # shape of the original image, with an
  # appropriated interpolation algorithm
  pred = cv2.resize(
    pred,
    dsize=(img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST_EXACT
  )

  toc = now()
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE} ({toc - tic:.3f} sec)                  ',
      end='\n',
      flush=True,
    )

  return pred
