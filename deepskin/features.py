#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mahotas as mh

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'evaluate_features'
]


def get_haralick (masked : np.ndarray) -> np.ndarray :
  '''
  Extract the Haralick features from the masked image

  Parameters
  ----------
  masked : np.ndarray
    Masked image in RGB, i.e cv2.bitwise_and(img, img, mask=mask)

  Returns
  -------
  h_feature : np.ndarray
    Haralick array of features
  '''
  # if there are not enough points in the mask
  if cv2.countNonZero(masked[..., 0]) < 10:
    # return an zero-filled array of features
    return np.zeros(
      shape=(13, ),
      dtype=np.float32
    )
  # compute the Haralick features
  h_feature = mh.features.haralick(
    masked,
    ignore_zeros=True,
    return_mean=True,
    distance=1
  )

  return h_feature

def get_rgb_channel_stats (img : np.ndarray,
                           mask : np.ndarray
                          ) -> tuple :
  '''
  Get Average score of RGB channels

  Parameters
  ----------
  img : np.ndarray
    Input image in RGB format

  mask : np.ndarray
    Input mask in GRAYSCALE format

  Returns
  -------
  (avg, std) : tuple
    Average & Std values related to RGB channels
  '''
  # convert the image into floating-point values
  rgb = np.float32(img)
  # normalize the values in [0, 1]
  rgb *= 1. / 255
  # compute avg and std of each channel
  avg, std = cv2.meanStdDev(
    src=rgb,
    mask=mask,
  )
  return avg, std

def get_hsv_channel_stats (img : np.ndarray,
                           mask : np.ndarray
                          ) -> tuple :
  '''
  Get Average score of HSV channels

  Parameters
  ----------
  img : np.ndarray
    Input image in RGB format

  mask : np.ndarray
    Input mask in GRAYSCALE format

  Returns
  -------
  (avg, std) : tuple
    Average & Std values related to HSV channels
  '''
  # conver the image from RGB to HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  # convert the image into floating-point values
  hsv = np.float32(hsv)
  # normalize the values in [0, 1]
  hsv *= 1./255
  # compute avg and std of each channel
  avg, std = cv2.meanStdDev(
    src=hsv,
    mask=mask
  )
  return avg, std

def get_lab_channel_stats (img : np.ndarray,
                           mask : np.ndarray
                          ) -> tuple :
  '''
  Get Average score of Lab channels

  Parameters
  ----------
  img : np.ndarray
    Input image in RGB format

  mask : np.ndarray
    Input mask in GRAYSCALE format

  Returns
  -------
  (avg, std) : tuple
    Average + Std values related to Lab channels
  '''
  # conver the image from RGB to LAB
  lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
  # convert the image into floating-point values
  lab = np.float32(lab)
  # normalize the values in [0, 1]
  lab *= 1./255
  # compute avg and std of each channel
  avg, std = cv2.meanStdDev(
    src=lab,
    mask=mask
  )
  return avg, std

def get_park_redness (img : np.ndarray,
                      mask : np.ndarray
                     ) -> float :
  '''
  Get the Park et al. Redness score feature (Does it work??)

  Parameters
  ----------
  img : np.ndarray
    Input image in RGB format

  mask : np.ndarray
    Input mask in GRAYSCALE format

  Returns
  -------
  score : float
    Park et al. redness score
  '''
  # apply the mask on the image
  image = cv2.bitwise_and(
    src1=img,
    src2=img,
    mask=mask,
  )
  # count the non zero values in the mask
  N = np.sum(mask != 0)
  # if the mask is empty return a fixed value
  if N == 0:
    return -0.5

  # convert the image into floating-point values
  image = np.float32(image)
  # split the image into channels
  r, g, b = cv2.split(image)
  # compute the Park redness score
  f2 = (2*r - g - b) / (2 * (r + g + b + 1e-5)) # range[-.5, 1]
  f2 = f2 * np.where(mask != 0, 1, np.nan)
  f2 = np.nansum(f2) / N
  return f2

def get_amparo_redness (img : np.ndarray,
                        mask : np.ndarray
                       ) -> float :
  '''
  Get the Amparo et al. Redness score feature (Does it work??)

  Parameters
  ----------
  img : np.ndarray
    Input image in RGB format

  mask : np.ndarray
    Input mask in GRAYSCALE format

  Returns
  -------
  score : float
    Amparo et al. redness score
  '''
  # apply the mask on the image
  image = cv2.bitwise_and(
    src1=img,
    src2=img,
    mask=mask,
  )
  # count the non zero values in the mask
  N = np.sum(mask != 0)
  # if the mask is empty return a fixed value

  if N == 0:
    return 0.0

  # convert the image to HSV color space
  hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  # split the image in channels
  h, s, v = cv2.split(hsv)
  # convert the h channel to float
  h = np.float32(h)
  # convert the s channel to float
  s = np.float32(s)
  # compute the Amparo et al. redness score
  f1 = (h * s) / (255 * 255) # range[0, 1]
  f1 = f1 * np.where(mask != 0, 1, np.nan)
  f1 = np.nansum(f1) / N
  return f1

def evaluate_features (img : np.ndarray,
                       mask : np.ndarray,
                       prefix : str
                      ) -> dict :
  '''
  Evaluate the deepskin feature according to the
  ROI identified by the provided mask.

  Parameters
  ----------
    img : np.ndarray
      Input original image in RGB format

    mask : np.ndarray
      Input mask for the ROI identification in binary format

    prefix : str
      Prefix name to prepend on the feature names

  Returns
  -------
    features : dict
      Output dictionary of features
  '''
  # apply the mask on the image to turn-off
  # all the background pixels
  # NOTE: this is mandatory only for the Haralick features
  masked = cv2.bitwise_and(
    img,
    img,
    mask=mask
  )
  # get the Haralick features
  haralick = get_haralick(masked).ravel()
  # convert them to dict
  haralick = {f'{prefix}haralick{i:d}' : v
    for i, v in enumerate(haralick)
  }

  # get the RGB color space features
  avg_rgb, std_rgb = get_rgb_channel_stats(
    img=img,
    mask=mask
  )
  # convert them to dict
  avg_rgb = {f'{prefix}avg{c}' : v
    for c, v in zip(['R', 'G', 'B'], avg_rgb.ravel())
  }
  std_rgb = {f'{prefix}std{c}' : v
    for c, v in zip(['R', 'G', 'B'], std_rgb.ravel())
  }

  # get the HSV color space features
  avg_hsv, std_hsv = get_hsv_channel_stats(
    img=img,
    mask=mask
  )
  # convert them to dict
  avg_hsv = {f'{prefix}avg{c}' : v
    for c, v in zip(['H', 'S', 'V'], avg_hsv.ravel())
  }
  std_hsv = {f'{prefix}std{c}' : v
    for c, v in zip(['H', 'S', 'V'], std_hsv.ravel())
  }

  # get the Lab color space features
  avg_lab, std_lab = get_lab_channel_stats(
    img=img,
    mask=mask
  )
  # convert them to dict
  avg_lab = {f'{prefix}avg{c}' : v
    for c, v in zip(['L', 'a', 'b'], avg_lab.ravel())
  }
  std_lab = {f'{prefix}std{c}' : v
    for c, v in zip(['L', 'a', 'b'], std_lab.ravel())
  }

  # get the Park redness feature
  park = get_park_redness(
    img=img,
    mask=mask
  )
  # convert it to dict
  park = {f'{prefix}park': park}

  # get the Amparo redness feature
  amparo = get_amparo_redness(
    img=img,
    mask=mask
  )
  amparo = {f'{prefix}amparo': amparo}

  # build the feature set
  features = {
    **haralick,
    **avg_rgb, **std_rgb,
    **avg_hsv, **std_hsv,
    **avg_lab, **std_lab,
    **park, **amparo,
  }

  return features
