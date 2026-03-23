#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import json

# package version
from .__version__ import __version__
# segmentation model for wound identification
from .segmentation import wound_segmentation
from .segmentation import seg_to_labels
from .segmentation import seg_to_onehot
from .segmentation import seg_to_rgb_mask
from .segmentation import generate_analysis_report
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

  # mask format
  parser.add_argument(
    '--mask-format',
    dest='mask_format',
    required=False,
    action='store',
    default='rgb',
    type=str,
    choices=['labels', 'rgb', 'onehot'],
    help=(
      'Output mask format: '
      'labels: 2D integer label map (0=background, 1=body, 2=wound), '
      'rgb: RGB color mask for visualization, '
      'onehot: 3-channel binary mask compatible with PWAT calculation'
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

  # report output
  parser.add_argument(
    '--report',
    dest='report',
    required=False,
    action='store',
    default=None,
    type=str,
    help=(
      'Path to save the analysis report in JSON format. '
      'The report includes pixel counts, area percentages, wound bbox, etc.'
    )
  )

  args = parser.parse_args()

  return args


def save_mask(mask, filepath, mask_format):
    '''Save mask according to specified format'''
    if mask_format == 'labels':
        # 保存为单通道PNG图像
        cv2.imwrite(filepath, mask)
    elif mask_format == 'rgb':
        # 保存为RGB彩色图像
        cv2.imwrite(filepath, mask[..., ::-1])  # BGR for OpenCV
    elif mask_format == 'onehot':
        # 保存为三通道二值图像（与现有格式兼容）
        cv2.imwrite(filepath, mask[..., ::-1])


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
      f'{RED_COLOR_CODE}deepskin Error: the following arguments are required: --input/-i{RESET_COLOR_CODE}',
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

  # 存储分割结果
  seg_raw = None
  seg_mask = None
  labels = None

  if args.mask or args.pwat or args.report:
    # 获取分割结果（同时获取原始softmax输出和兼容格式mask）
    seg_raw, seg_mask = wound_segmentation(
      img=rgb,
      tol=0.5,
      verbose=args.verbose,
      return_raw=True,
    )
    # 生成单标签图
    labels = seg_to_labels(seg_raw)

  # 处理mask输出
  if args.mask:
    # 根据指定格式转换mask
    if args.mask_format == 'labels':
        output_mask = labels
    elif args.mask_format == 'rgb':
        output_mask = seg_to_rgb_mask(labels)
    else:  # onehot format (default)
        output_mask = seg_mask

    # 获取输出目录
    outdir = os.path.dirname(args.filepath)
    # 获取文件名
    name = os.path.basename(args.filepath)
    # 移除扩展名
    name, _ = os.path.splitext(name)
    # 构建输出文件名
    outfile = f'{outdir}/{name}_deepskin_{args.mask_format}.png'
    # 保存mask
    save_mask(output_mask, outfile, args.mask_format)
    
    if args.verbose:
        print(f'{GREEN_COLOR_CODE}Mask saved to: {outfile}{RESET_COLOR_CODE}',
            end='\n', flush=True)

  if args.pwat:
    # 计算PWAT分数时使用兼容格式的mask
    pwat = evaluate_PWAT_score(
      img=rgb,
      mask=seg_mask,
      verbose=args.verbose,
    )

    print(f'{GREEN_COLOR_CODE}PWAT prediction: {pwat:.3f}{RESET_COLOR_CODE}',
      end='\n',
      flush=True,
    )

  if args.report:
    # 生成分析报告
    report = generate_analysis_report(seg_raw, labels)
    
    # 确保输出目录存在
    report_dir = os.path.dirname(args.report)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)
    
    # 保存报告到文件
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    if args.verbose:
        print(f'{GREEN_COLOR_CODE}Analysis report saved to: {args.report}{RESET_COLOR_CODE}',
            end='\n', flush=True)


if __name__ == '__main__':

  main ()
