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
# post-processing functions
from .postprocess import (
  softmax_to_label_map,
  label_map_to_onehot,
  label_map_to_rgb,
  label_map_to_mask,
  SegmentationReport,
  convert_mask_format,
)

import cv2
import numpy as np
from time import time as now
from typing import Optional, Dict, Any

# constant values
from .constants import CRLF
from .constants import IMG_SIZE
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'wound_segmentation',
  'wound_segmentation_advanced',
]

def _get_model_weights_path() -> str:
  '''获取模型权重文件路径'''
  local = os.path.dirname(os.path.abspath(__file__))
  weightspath = os.path.join(
    local,
    '..',
    'checkpoints',
    'efficientnetb3_deepskin_semantic.h5'
  )
  return weightspath

def _ensure_weights_available(weightspath: str, verbose: bool = False):
  '''确保模型权重文件可用'''
  if not os.path.exists(weightspath):
    download_model_weights(
      Id=MODEL_CHECKPOINT,
      model_name='efficientnetb3_deepskin_semantic'
    )

def _preprocess_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
  '''预处理输入图像'''
  # resize the image into the shape required by the model
  resized = cv2.resize(
    img,
    dsize=(target_h, target_w),
    interpolation=cv2.INTER_CUBIC
  )
  # convert the image into floating-point values
  resized = np.float32(resized)
  # normalize the image into [0, 1] range
  resized *= 1. / 255
  # extend the dimensionality of the input array
  # to the [batch, h, w, c] format
  resized = resized.reshape(1, *resized.shape)
  return resized

def _postprocess_mask(
  pred: np.ndarray,
  original_shape: tuple,
  tol: float = 0.5,
  output_format: str = 'mask'
) -> np.ndarray:
  '''
  后处理分割结果

  Parameters
  ----------
  pred : np.ndarray
    模型原始输出 (softmax概率)
  original_shape : tuple
    原始图像shape (H, W)
  tol : float
    阈值（用于二值化，在onehot/mask格式中使用）
  output_format : str
    输出格式: 'mask', 'labels', 'onehot', 'rgb'

  Returns
  -------
  result : np.ndarray
    处理后结果
  '''
  # remove useless dimensions from the image
  pred = np.squeeze(pred)

  # 转换为单标签语义分割图（每个像素只属于一个类别）
  label_map = softmax_to_label_map(pred)

  # 根据输出格式转换
  if output_format == 'labels':
    result = label_map
  elif output_format == 'onehot':
    result = label_map_to_onehot(label_map)
    # 应用阈值二值化
    result = np.where(result > tol * 255, 255, 0).astype(np.uint8)
  elif output_format == 'rgb':
    result = label_map_to_rgb(label_map)
  elif output_format == 'mask':
    # 与原API兼容的格式：3通道二值mask
    result = label_map_to_mask(label_map)
    # 应用阈值二值化
    result = np.where(result > tol * 255, 255, 0).astype(np.uint8)
  else:
    raise ValueError(f"Unknown output_format: {output_format}")

  # resize the output mask to the same shape of the original image
  if output_format in ('labels', 'rgb'):
    interpolation = cv2.INTER_NEAREST_EXACT
  else:
    interpolation = cv2.INTER_NEAREST_EXACT

  if result.ndim == 2:
    result = cv2.resize(
      result,
      dsize=(original_shape[1], original_shape[0]),
      interpolation=interpolation
    )
  else:
    # 多通道需要逐通道resize
    channels = []
    for c in range(result.shape[-1]):
      ch = cv2.resize(
        result[..., c],
        dsize=(original_shape[1], original_shape[0]),
        interpolation=interpolation
      )
      channels.append(ch)
    result = np.stack(channels, axis=-1)

  return result


def wound_segmentation_advanced(
  img: np.ndarray,
  tol: float = 0.5,
  output_format: str = 'mask',
  verbose: bool = False
) -> Dict[str, Any]:
  '''
  高级伤口分割函数，支持多种输出格式和分析报告

  Parameters
  ----------
  img : np.ndarray
    Input image to analyze in RGB format
  tol : float (default := 0.5)
    Threshold to apply on the resulting mask for the output binarization
  output_format : str (default := 'mask')
    Output format: 'mask' (兼容原API), 'labels', 'onehot', 'rgb'
  verbose : bool (default := False)
    Enable/Disable the logging of the steps

  Returns
  -------
  result : dict
    包含以下键的字典:
    - 'mask': 分割结果（格式由output_format决定）
    - 'label_map': 单标签图 (H, W)
    - 'softmax_output': 原始softmax输出
    - 'report': SegmentationReport对象（可用于导出JSON报告）
  '''
  tic = now()
  step = 'Perform the semantic image segmentation... '

  if verbose:
    print(f'{step}',
      end='\r',
      flush=True,
    )

  # build the model for the semantic segmentation
  model = deepskin_model(verbose=False)

  # load the appropriated weights
  weightspath = _get_model_weights_path()
  _ensure_weights_available(weightspath, verbose=verbose)
  model.load_weights(weightspath)

  # get the model input shape
  _, h, w, c = model.input.shape
  original_shape = img.shape[:2]

  # pre-process the input
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[1/4] pre-process the input image{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  resized = _preprocess_image(img, h, w)

  # apply the segmentation model
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[2/4] apply the segmentation model{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  # apply the model to get the prediction (softmax probabilities)
  softmax_output = model.predict(resized, verbose=0)
  # remove batch dimension
  softmax_output = np.squeeze(softmax_output)

  # 生成单标签图（用于报告）
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[3/4] generate semantic label map{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )
  label_map = softmax_to_label_map(softmax_output)

  # post-process the mask
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[4/4] post-process the segmentation mask{RESET_COLOR_CODE}',
      end='',
      flush=True,
    )

  # resize label_map到原始尺寸用于报告
  label_map_resized = cv2.resize(
    label_map.astype(np.uint8),
    dsize=(original_shape[1], original_shape[0]),
    interpolation=cv2.INTER_NEAREST_EXACT
  )

  # 生成报告（使用resize后的label_map）
  # 同时resize softmax_output用于计算置信度
  softmax_resized = []
  for c in range(softmax_output.shape[-1]):
    ch = cv2.resize(
      softmax_output[..., c],
      dsize=(original_shape[1], original_shape[0]),
      interpolation=cv2.INTER_LINEAR
    )
    softmax_resized.append(ch)
  softmax_resized = np.stack(softmax_resized, axis=-1)

  report = SegmentationReport(label_map_resized, softmax_resized)

  # 生成指定格式的mask输出
  mask = _postprocess_mask(
    pred=softmax_output,
    original_shape=original_shape,
    tol=tol,
    output_format=output_format
  )

  toc = now()
  if verbose:
    print(f'{CRLF}{step} {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE} ({toc - tic:.3f} sec)                  ',
      end='\n',
      flush=True,
    )

  return {
    'mask': mask,
    'label_map': label_map_resized,
    'softmax_output': softmax_resized,
    'report': report,
  }


def wound_segmentation(
  img: np.ndarray,
  tol: float = 0.5,
  verbose: bool = False
) -> np.ndarray:
  '''
  Perform the semantic image segmentation using the
  Deepskin semantic model.

  此函数保持与原API完全兼容，内部使用新的单标签语义分割逻辑。
  每个像素只能属于一个类别（background, body, 或 wound）。

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
    注意：由于使用单标签语义分割，每个像素在所有通道中
    只有一个通道的值为255，其他通道为0。
  '''
  result = wound_segmentation_advanced(
    img=img,
    tol=tol,
    output_format='mask',  # 与原API兼容的格式
    verbose=verbose,
  )
  return result['mask']
