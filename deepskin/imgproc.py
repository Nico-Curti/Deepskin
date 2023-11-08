#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'imfill',
  'get_perilesion_mask'
]

def imfill (img : np.ndarray) -> np.ndarray:
  '''
  Fill the holes of a single image.

  Parameters
  ----------
    img : np.ndarray
      Image to fill in GRAYSCALE

  Returns
  -------
    filled : array-like
      filled image
  '''
  # Copy the thresholded image.
  img = np.pad(
    img.astype('uint8'),
    pad_width=((2, 2), (2, 2)),
    mode='constant',
    constant_values=(0., 0.)
  )
  im_floodfill = img.copy()
  # Floodfill from point (0, 0)
  cv2.floodFill(
    im_floodfill,
    mask=None,
    seedPoint=(0, 0),
    newVal=255
  )
  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)
  # Combine the two images to get the foreground.
  im_floodfill = img | im_floodfill_inv
  return im_floodfill[2:-2, 2:-2]

def get_perilesion_mask (mask : np.ndarray,
                         ksize : tuple = (20, 20)
                        ) -> np.ndarray :
  '''
  Extract the peri-lesion mask from the wound mask

  Parameters
  ----------
    mask : np.ndarray
      Input wound mask in GRAYSCALE

    ksize : tuple
      Kernel dimension for the mask processing

  Returns
  -------
    periwound_mask : np.ndarray
      Peri-lesion mask of the wound
  '''
  # define a circular kernel with a big radius
  kernel = cv2.getStructuringElement(
    shape=cv2.MORPH_ELLIPSE,
    ksize=ksize,
    anchor=(-1, -1)
  )
  # perform an erosion on the mask
  erosion = cv2.erode(
    src=mask,
    kernel=kernel,
    anchor=(-1, -1),
    iterations=1,
    borderType=cv2.BORDER_CONSTANT
  )
  # perform a dilation on the mask
  dilation = cv2.dilate(
    src=mask,
    kernel=kernel,
    anchor=(-1, -1),
    iterations=1,
    borderType=cv2.BORDER_CONSTANT
  )
  # the peri-wound section is given by the subtraction
  # of the dilated mask and the eroded one
  periwound_mask = cv2.subtract(
    src1=dilation,
    src2=erosion
  )
  return periwound_mask
