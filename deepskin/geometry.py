#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from .imgproc import _get_zcoord
from .imgproc import _triangle_area
from .imgproc import extract_triangulation
from .imgproc import rectify_boundary_mask

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'shape_area',
  'shape_edge',
]


def shape_area (mask : np.ndarray,
                depth : np.ndarray,
                max_area_triangles : float = 100,
               ) -> float :
  '''
  Get the shape area of the input mask using
  a Triangulation 2.5D approach, correcting
  the mesh according to the given depth-map.

  Parameters
  ----------
    mask : np.ndarray
      Input binary mask of the desired shape

    depth : np.ndarray
      Depth map or deformation field to use
      for the mesh coordinates correction

    max_area_triangles : float (default := 100)
      Maximum area of each triangle.
      The set of triangles will be iteratively refined
      splitting the shapes with area greater than
      this threshold and considering the triangle
      circumcenter as additional point for the
      triangulation

  Returns
  -------
    area : float
      Area of the shape corrected by depth map expressed
      in the same unit of measurements of the given
      deformation field.

  Notes
  -----
  A low value of `max_area_triangles` parameters leads to a
  finer subdivision of the shape and to a more detailed mesh
  representation.
  This parameter must be tuned balancing the computational
  time and the precision of the desired result.

  References
  ----------
  *  Dall'Olio L, Griffa D, Zengarini C, et al.
    Automated 3D Wound Surface Area Estimation: Novel Computer Vision 
    Approaches for Clinical Application. 
    Available at SSRN: https://ssrn.com/abstract=6100282 or http://dx.doi.org/10.2139/ssrn.6100282 
  '''

  # extract the shape contours
  shape_cnt, _ = cv2.findContours(
    image=mask,
    mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_NONE
  )
  # filter only the largest one
  shape_cnt = max(shape_cnt, key=len)
  # remove useless dimensions
  shape_cnt = np.squeeze(shape_cnt)
  # convert it to a set for a faster computation
  shape_cnt = set(map(tuple, shape_cnt))

  # apply an erosion to the mask to get the
  # contour points of an inner shape
  kernel = cv2.getStructuringElement(
    shape=cv2.MORPH_ELLIPSE,
    ksize=(3, 3)
  )
  eroded = cv2.erode(
    mask,
    kernel=kernel,
    iterations=1
  )
  # get the internal contours
  internal_cnt, _ = cv2.findContours(
    image=eroded,
    mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_NONE
  )
  # filter only the largest one
  internal_cnt = max(internal_cnt, key=len)
  # remove useless dimensions
  internal_cnt = np.squeeze(internal_cnt)
  # convert it to a set for a faster computation
  internal_cnt = set(map(tuple, internal_cnt))

  # update the list of points combining the two sets
  internal_cnt.update(shape_cnt)
  # convert the list to a numpy array
  points = list(internal_cnt)

  # get the internal triangles
  internal_triangles = extract_triangulation(
    points=points,
    shape_cnt=shape_cnt,
    max_area=max_area_triangles
  )

  # add the z-axis to the triangle vertices
  extended_internal_triangles = [
    np.c_[t, _get_zcoord(t[:, ::-1], depth)]
    for t in internal_triangles
  ]

  return sum([
    _triangle_area(t)
    for t in extended_internal_triangles
  ], 0.0)

def shape_edge (mask : np.ndarray,
                depth : np.ndarray,
                iterations : int = 1,
               ) -> tuple :
  '''
  Get the shape edge of the input mask using
  the information extracted from the depth-map
  and considering the transition area between inner
  and outer part of the mask.

  Parameters
  ----------
    mask : np.ndarray
      Input binary mask of the desired shape

    depth : np.ndarray
      Depth map or deformation field to use
      for the mesh coordinates correction

  Returns
  -------
    mean : np.ndarray
      Average trend of the mask edge.
    
    std : np.ndarray
      Standard deviation of the shape edge

  Notes
  -----
  The correct evaluation of this edge requires a 
  consistent depth-map estimation and some preliminary
  adjustment could improve the quality of the 
  reconstruction.
  Furthermore, the input mask should include only one
  connected component to guarantee the correct estimation
  of the mask rectification.

  References
  ----------
  * Zengarini C, Giacometti T, Merli Y, et al. 
    Toward Objective Wound Edge Classification in Clinical Practice. 
    Exp Dermatol. 2026;35(6):e70287. doi:10.1111/exd.70287 
  '''
  # rectify the depth map
  rectify_depth, (_, _) = rectify_boundary_mask(
    np.where(depth != 0, depth, np.nan), 
    mask, 
    iterations=iterations
  )
  # permute the axis for a correct estimation
  rectify_depth = np.swapaxes(
    rectify_depth, 0, 1
  )
  # get the signal data
  mean = np.nanmean(rectify_depth, axis=1)
  std = np.nanstd(rectify_depth, axis=1)

  return mean, std
