#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
from typing import Dict, Any, Tuple, Optional

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'softmax_to_label_map',
  'label_map_to_onehot',
  'label_map_to_rgb',
  'SegmentationReport',
]


# 类别定义
CLASS_NAMES = ['background', 'body', 'wound']
CLASS_COLORS = {
  'background': (0, 0, 255),    # 蓝色 - 对应原mask的channel 0
  'body': (0, 255, 0),          # 绿色 - 对应原mask的channel 1
  'wound': (255, 0, 0),         # 红色 - 对应原mask的channel 2
}
CLASS_INDICES = {
  'background': 0,
  'body': 1,
  'wound': 2,
}


def softmax_to_label_map(softmax_output: np.ndarray) -> np.ndarray:
  '''
  将模型softmax输出转换为单标签语义分割图
  每个像素只能属于一个类别（取概率最大的类别）

  Parameters
  ----------
  softmax_output : np.ndarray
    模型softmax输出，shape为 (H, W, C) 或 (H, W, 3)
    通道顺序: 0=background, 1=body, 2=wound

  Returns
  -------
  label_map : np.ndarray
    单标签图，shape为 (H, W)，dtype为uint8
    像素值: 0=background, 1=body, 2=wound
  '''
  if softmax_output.ndim != 3:
    raise ValueError(f"Expected 3D array (H, W, C), got {softmax_output.ndim}D")

  # 取每个像素概率最大的类别索引
  label_map = np.argmax(softmax_output, axis=-1)
  return label_map.astype(np.uint8)


def label_map_to_onehot(label_map: np.ndarray, num_classes: int = 3) -> np.ndarray:
  '''
  将单标签图转换为one-hot编码

  Parameters
  ----------
  label_map : np.ndarray
    单标签图，shape为 (H, W)
  num_classes : int (default := 3)
    类别数量

  Returns
  -------
  onehot : np.ndarray
    one-hot编码，shape为 (H, W, num_classes)，dtype为uint8
    通道顺序: 0=background, 1=body, 2=wound
  '''
  if label_map.ndim != 2:
    raise ValueError(f"Expected 2D array (H, W), got {label_map.ndim}D")

  h, w = label_map.shape
  onehot = np.zeros((h, w, num_classes), dtype=np.uint8)

  for c in range(num_classes):
    onehot[..., c] = (label_map == c).astype(np.uint8) * 255

  return onehot


def label_map_to_rgb(label_map: np.ndarray) -> np.ndarray:
  '''
  将单标签图转换为RGB彩色图（用于可视化）

  Parameters
  ----------
  label_map : np.ndarray
    单标签图，shape为 (H, W)

  Returns
  -------
  rgb_mask : np.ndarray
    RGB彩色图，shape为 (H, W, 3)，dtype为uint8
    background=蓝色, body=绿色, wound=红色
  '''
  if label_map.ndim != 2:
    raise ValueError(f"Expected 2D array (H, W), got {label_map.ndim}D")

  h, w = label_map.shape
  rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

  colors = [CLASS_COLORS['background'], CLASS_COLORS['body'], CLASS_COLORS['wound']]
  for c, color in enumerate(colors):
    mask = (label_map == c)
    rgb_mask[mask] = color

  return rgb_mask


def label_map_to_mask(label_map: np.ndarray) -> np.ndarray:
  '''
  将单标签图转换为与原API兼容的mask格式（3通道，每通道一个类别的二值mask）

  Parameters
  ----------
  label_map : np.ndarray
    单标签图，shape为 (H, W)

  Returns
  -------
  mask : np.ndarray
    与原API兼容的mask，shape为 (H, W, 3)，dtype为uint8
    通道顺序: 0=background, 1=body, 2=wound
  '''
  return label_map_to_onehot(label_map, num_classes=3)


class SegmentationReport:
  '''
  分割结果分析报告类

  生成包含以下信息的JSON报告:
  - 每个类别的像素数
  - 每个类别的面积占比
  - wound区域bbox
  - 是否检测到wound
  - wound平均置信度（如果有softmax概率）
  '''

  def __init__(self, label_map: np.ndarray, softmax_output: Optional[np.ndarray] = None):
    '''
    Parameters
    ----------
    label_map : np.ndarray
      单标签图，shape为 (H, W)
    softmax_output : np.ndarray, optional
      模型softmax输出，shape为 (H, W, C)，用于计算置信度
    '''
    self.label_map = label_map
    self.softmax_output = softmax_output
    self.h, self.w = label_map.shape
    self.total_pixels = self.h * self.w

  def _count_pixels(self) -> Dict[str, int]:
    '''统计每个类别的像素数'''
    pixel_counts = {}
    for name, idx in CLASS_INDICES.items():
      pixel_counts[name] = int(np.sum(self.label_map == idx))
    return pixel_counts

  def _calculate_ratios(self, pixel_counts: Dict[str, int]) -> Dict[str, float]:
    '''计算每个类别的面积占比'''
    ratios = {}
    for name, count in pixel_counts.items():
      ratios[name] = round(count / self.total_pixels, 6) if self.total_pixels > 0 else 0.0
    return ratios

  def _get_wound_bbox(self) -> Optional[Dict[str, int]]:
    '''获取wound区域的边界框'''
    wound_mask = (self.label_map == CLASS_INDICES['wound']).astype(np.uint8)

    if np.sum(wound_mask) == 0:
      return None

    # 找到非零像素的坐标
    y_indices, x_indices = np.where(wound_mask > 0)

    bbox = {
      'x_min': int(np.min(x_indices)),
      'y_min': int(np.min(y_indices)),
      'x_max': int(np.max(x_indices)),
      'y_max': int(np.max(y_indices)),
      'width': int(np.max(x_indices) - np.min(x_indices) + 1),
      'height': int(np.max(y_indices) - np.min(y_indices) + 1),
    }
    return bbox

  def _get_wound_confidence(self) -> Optional[float]:
    '''计算wound区域的平均置信度'''
    if self.softmax_output is None:
      return None

    wound_mask = (self.label_map == CLASS_INDICES['wound'])

    if np.sum(wound_mask) == 0:
      return None

    # 获取wound类别的softmax概率
    wound_probs = self.softmax_output[..., CLASS_INDICES['wound']]
    wound_confidence = np.mean(wound_probs[wound_mask])

    return round(float(wound_confidence), 6)

  def generate(self) -> Dict[str, Any]:
    '''
    生成完整的分析报告

    Returns
    -------
    report : dict
      JSON格式的分析报告
    '''
    pixel_counts = self._count_pixels()
    area_ratios = self._calculate_ratios(pixel_counts)
    wound_bbox = self._get_wound_bbox()
    wound_confidence = self._get_wound_confidence()

    report = {
      'image_shape': [self.h, self.w],
      'total_pixels': self.total_pixels,
      'pixel_counts': pixel_counts,
      'area_ratios': area_ratios,
      'wound_detected': pixel_counts['wound'] > 0,
    }

    if wound_bbox is not None:
      report['wound_bbox'] = wound_bbox

    if wound_confidence is not None:
      report['wound_confidence'] = wound_confidence

    return report

  def to_json(self, indent: int = 2) -> str:
    '''将报告转换为JSON字符串'''
    return json.dumps(self.generate(), indent=indent, ensure_ascii=False)

  def save(self, filepath: str, indent: int = 2):
    '''保存报告到JSON文件'''
    with open(filepath, 'w', encoding='utf-8') as f:
      json.dump(self.generate(), f, indent=indent, ensure_ascii=False)


def convert_mask_format(
  label_map: np.ndarray,
  format: str,
  softmax_output: Optional[np.ndarray] = None
) -> np.ndarray:
  '''
  将单标签图转换为指定格式的mask

  Parameters
  ----------
  label_map : np.ndarray
    单标签图，shape为 (H, W)
  format : str
    输出格式: 'labels', 'onehot', 'rgb', 'mask'
    - 'labels': 返回单标签图 (H, W)
    - 'onehot': 返回one-hot编码 (H, W, 3)
    - 'rgb': 返回RGB彩色图 (H, W, 3)
    - 'mask': 返回与原API兼容的3通道mask (H, W, 3)
  softmax_output : np.ndarray, optional
    模型softmax输出，某些格式可能需要

  Returns
  -------
  mask : np.ndarray
    指定格式的mask
  '''
  format = format.lower()

  if format == 'labels':
    return label_map
  elif format in ('onehot', 'mask'):
    return label_map_to_onehot(label_map)
  elif format == 'rgb':
    return label_map_to_rgb(label_map)
  else:
    raise ValueError(f"Unknown format: {format}. Supported formats: 'labels', 'onehot', 'rgb', 'mask'")
