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
from typing import Tuple, Dict, Optional, Union

# constant values
from .constants import CRLF
from .constants import IMG_SIZE
from .constants import RESET_COLOR_CODE
from .constants import GREEN_COLOR_CODE

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'wound_segmentation',
  'seg_to_labels',
  'seg_to_onehot',
  'seg_to_rgb_mask',
  'generate_analysis_report',
]

# 类别定义
CLASS_NAMES = ['background', 'body', 'wound']
CLASS_COLORS = np.array([
    [0, 0, 255],      # background - blue
    [0, 255, 0],      # body - green
    [255, 0, 0]       # wound - red
], dtype=np.uint8)

def _preprocess_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    '''预处理输入图像'''
    resized = cv2.resize(
        img,
        dsize=target_size,
        interpolation=cv2.INTER_CUBIC
    )
    resized = np.float32(resized)
    resized *= 1. / 255
    resized = resized.reshape(1, *resized.shape)
    return resized

def _postprocess_output(pred: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    '''后处理模型输出'''
    pred = np.squeeze(pred)
    pred = cv2.resize(
        pred,
        dsize=original_shape,
        interpolation=cv2.INTER_NEAREST_EXACT
    )
    return pred

def seg_to_labels(seg_output: np.ndarray) -> np.ndarray:
    '''
    将模型softmax输出转换为二维整数标签图
    
    Parameters
    ----------
    seg_output : np.ndarray
        模型输出的softmax结果，shape为 (H, W, 3)
    
    Returns
    -------
    labels : np.ndarray
        二维整数标签图，0=background, 1=body, 2=wound，dtype=np.uint8
    '''
    return np.argmax(seg_output, axis=-1).astype(np.uint8)

def seg_to_onehot(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    '''
    将标签图转换为one-hot编码格式
    
    Parameters
    ----------
    labels : np.ndarray
        二维整数标签图，shape为 (H, W)
    
    num_classes : int
        类别数量，默认为3
    
    Returns
    -------
    onehot : np.ndarray
        one-hot编码的mask，shape为 (H, W, num_classes)，dtype=np.uint8
    '''
    h, w = labels.shape
    onehot = np.zeros((h, w, num_classes), dtype=np.uint8)
    for i in range(num_classes):
        onehot[..., i] = (labels == i).astype(np.uint8) * 255
    return onehot

def seg_to_rgb_mask(labels: np.ndarray) -> np.ndarray:
    '''
    将标签图转换为RGB可视化mask
    
    Parameters
    ----------
    labels : np.ndarray
        二维整数标签图，shape为 (H, W)
    
    Returns
    -------
    rgb_mask : np.ndarray
        RGB格式的mask，shape为 (H, W, 3)，dtype=np.uint8
    '''
    return CLASS_COLORS[labels]

def generate_analysis_report(
    seg_output: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict:
    '''
    生成分割结果的分析报告
    
    Parameters
    ----------
    seg_output : np.ndarray
        模型输出的softmax结果，shape为 (H, W, 3)
    
    labels : np.ndarray, optional
        预计算的标签图，如未提供将从seg_output计算
    
    Returns
    -------
    report : Dict
        包含分析结果的字典
    '''
    if labels is None:
        labels = seg_to_labels(seg_output)
    
    h, w = labels.shape
    total_pixels = h * w
    
    report = {
        'image_size': {'height': h, 'width': w},
        'total_pixels': int(total_pixels),
        'classes': {},
        'has_wound': bool(np.any(labels == 2)),
    }
    
    # 计算每个类别的统计信息
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_mask = (labels == class_id)
        pixel_count = int(np.sum(class_mask))
        
        class_stats = {
            'pixel_count': pixel_count,
            'area_percentage': float(pixel_count / total_pixels * 100),
        }
        
        # 如果有softmax概率，计算平均置信度
        if seg_output is not None and pixel_count > 0:
            class_probs = seg_output[..., class_id][class_mask]
            class_stats['mean_confidence'] = float(np.mean(class_probs))
        
        report['classes'][class_name] = class_stats
    
    # 计算wound的bbox
    if report['has_wound']:
        wound_pixels = np.where(labels == 2)
        y_min, y_max = int(np.min(wound_pixels[0])), int(np.max(wound_pixels[0]))
        x_min, x_max = int(np.min(wound_pixels[1])), int(np.max(wound_pixels[1]))
        report['wound_bbox'] = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min + 1,
            'height': y_max - y_min + 1,
        }
    else:
        report['wound_bbox'] = None
    
    return report

def wound_segmentation(
    img: np.ndarray,
    tol: float = 0.5,
    verbose: bool = False,
    return_raw: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    '''
    Perform the semantic image segmentation using the Deepskin semantic model.
    
    Parameters
    ----------
    img : np.ndarray
        Input image to analyze in RGB format
    
    tol : float (default := 0.5)
        兼容性参数，不再使用，保留用于API兼容
    
    verbose : bool (default := False)
        Enable/Disable the logging of the steps
    
    return_raw : bool (default := False)
        如果为True，同时返回原始softmax输出和标签图
    
    Returns
    -------
    pred : np.ndarray
        输出图像mask，采用单标签语义分割格式（每个像素属于一个类别）
        默认返回与PWAT计算兼容的格式 (H, W, 3)，其中:
        - channel 0: background mask (0/255)
        - channel 1: body mask (0/255)
        - channel 2: wound mask (0/255)
    
    (raw_output, pred) : Tuple[np.ndarray, np.ndarray]
        当return_raw=True时，返回原始softmax输出和处理后的mask
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
    
    resized = _preprocess_image(img, (h, w))

    # apply the segmentation model

    if verbose:
        print(f'{CRLF}{step} {GREEN_COLOR_CODE}[2/3] apply the segmentation model{RESET_COLOR_CODE}',
            end='',
            flush=True,
        )
    # apply the model to get the prediction
    pred = model.predict(resized, verbose=0)
    # post-process to original image size
    pred = _postprocess_output(pred, (img.shape[1], img.shape[0]))

    # post-process the mask - 转换为单标签语义分割

    if verbose:
        print(f'{CRLF}{step} {GREEN_COLOR_CODE}[3/3] post-process the segmentation mask{RESET_COLOR_CODE}',
            end='',
            flush=True,
        )
    
    # 生成单标签图
    labels = seg_to_labels(pred)
    
    # 转换为与PWAT兼容的mask格式（三通道二值图）
    compatible_mask = seg_to_onehot(labels)

    toc = now()
    if verbose:
        print(f'{CRLF}{step} {GREEN_COLOR_CODE}[DONE]{RESET_COLOR_CODE} ({toc - tic:.3f} sec)                  ',
            end='\n',
            flush=True,
        )

    if return_raw:
        return pred, compatible_mask
    
    return compatible_mask
