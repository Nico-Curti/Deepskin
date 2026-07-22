#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import scipy
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'imfill',
  'get_perilesion_mask',
  'extract_triangulation',
  '_triangle_area',
  '_filter_valid_triangles',
  '_get_zcoord',
]


def imfill (img : np.ndarray) -> np.ndarray :
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

def _triangle_area (vertices : np.ndarray) -> float :
  '''
  Compute the area of the triangle
  starting from the list of vertices using
  the Heron's formula

  Parameters
  ----------
    vertices : np.ndarray
      Array of vertices as 3x3 matrix

  Returns
  -------
    area : float
      Floating point area of the triangle
  '''
  # compute the sides of the triangle
  # as pairwaised distance of its vertices
  sides = pdist(vertices)
  # evaluate the semi-perimenter
  S = np.sum(sides) * .5
  # apply the Heron's equation
  return np.sqrt(S * np.prod(S - sides))

def _filter_valid_triangles (delaunay : scipy.spatial._qhull.Delaunay,
                             shape_cnt : list
                            ) -> list :
  '''
  Filter the Delaunay Triangles considering
  only the triangles *inside* the contour shape.

  Parameters
  ----------
    delaunay : scipy.spatial._qhull.Delaunay
      Delaunay Triangulation object

    shape_cnt : list
      List of tuples (x, y) belonging to the contour
      of the 2D shape.
      These points will be used as constraint for
      the filtering of spurious triangles due to
      putative shape concavities

  Returns
  -------
    triangles : list
      List of triangles considered inside the
      contour shape

  Notes
  -----
  A triangle is considered inside the shape if
  at least one of its vertices *does not* belong
  to the contour list
  '''
  # extract the Delaunay triangles
  triangles = delaunay.points[delaunay.simplices]
  # filter the triangles outside the contour
  internal_triangles = [
    t for t in triangles
    if not all(ti in shape_cnt for ti in map(tuple, t))
  ]
  return internal_triangles

def extract_triangulation (points : np.ndarray,
                           shape_cnt : list,
                           max_area : float = 100
                          ) -> list :
  '''
  Evaluate the Constraint Delaunay Triangulation of
  a 2D series of points.
  The triangulation will be refined using the `max_area`
  threshold and considering only the triangles
  inside the `shape_cnt` list of vertices.

  Parameters
  ----------
    points : list
      List of tuples (x, y) to use for the Delaunay
      Triangulation

    shape_cnt : list
      List of tuples (x, y) belonging to the contour
      of the 2D shape.
      These points will be used as constraint for
      the filtering of spurious triangles due to
      putative shape concavities

    max_area : float (default := 100)
      Maximum area of each triangle.
      The set of triangles will be iteratively refined
      splitting the shapes with area greater than
      this threshold and considering the triangle
      circumcenter as additional point for the
      triangulation

  Returns
  -------
  triangles : list
      List of valid triangles found represented as
      matrix of vertices (N, 3, 2)

  Notes
  -----
  A low value of `max_area` parameters leads to a finer
  subdivision of the shape and to a more detailed mesh
  representation.
  This parameter must be tuned balancing the computational
  time and the precision of the desired result.
  '''
  # extract the Delaunay triangulation
  delaunay = Delaunay(
    points=points,
    furthest_site=False,
    incremental=True,
    qhull_options=None
  )

  # extract the internal triangles applying the
  # shape constraint
  internal_triangles = _filter_valid_triangles(
    delaunay=delaunay,
    shape_cnt=shape_cnt
  )

  # incremental add new points in the
  # triangulation if the area is over the threshold
  # NOTE: as new points we will use the circumcenter
  # coordinate of the triangle shape
  refined = True
  while refined:
    refined = False
    # if the triangle area is greater
    # than the given threshold, compute
    # the triangle circumcenter and add it
    # to the list of new points
    new_points = [
      np.mean(t, axis=0)
      for t in internal_triangles
      if _triangle_area(t) > max_area
    ]

    # if there are new points to insert
    if len(new_points):
      refined = True
      # update the triangulation
      delaunay.add_points(new_points)
      # re-compute the triangles AND
      # extract the internal triangles
      internal_triangles = _filter_valid_triangles(
        delaunay=delaunay,
        shape_cnt=shape_cnt
      )
  # close the triangulation
  delaunay.close()

  return internal_triangles

def _get_zcoord (xy : list, depth : np.ndarray) -> np.ndarray :
  '''
  Extract the z-coordinate from a depth
  matrix using floating point coordinates.

  Parameters
  ----------
    xy : list
      List of 2D coordinates in floating point

    depth : np.ndarray
      Matrix of values to sample

  Returns
  -------
    z : list
      Z-coordinates associated to that points
  '''
  # get the coordinate approximations
  fl = np.int32(np.floor(xy))
  cl = np.int32(np.ceil(xy))
  # get the value as average of coord ROI
  z = [
    np.mean(
      depth[
        fl_x : max(fl_x + 1, cl_x),
        fl_y : max(fl_y + 1, cl_y)
      ]
    )
    for (fl_x, fl_y), (cl_x, cl_y) in zip(fl, cl)
  ]
  return z
