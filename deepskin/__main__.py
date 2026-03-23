#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse

# package version
from .__version__ import __version__
# segmentation model for wound identification
from .segmentation import wound_segmentation, wound_segmentation_advanced
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

  # deepskin --mask-format
  parser.add_argument(
    '--mask-format',
    dest='mask_format',
    required=False,
    action='store',
    default='mask',
    type=str,
    choices=['labels', 'onehot', 'rgb', 'mask'],
    help=(
      'Output format for the segmentation mask. '
      'labels: 2D integer label map (0=background, 1=body, 2=wound); '
      'onehot: 3-channel one-hot encoded mask; '
      'rgb: RGB color map for visualization; '
      'mask: 3-channel binary mask (default, compatible with PWAT)'
    )
  )

  # deepskin --report
  parser.add_argument(
    '--report',
    dest='report_path',
    required=False,
    action='store',
    default=None,
    type=str,
    help=(
      'Export segmentation analysis report to JSON file. '
      'The report includes pixel counts, area ratios, wound bbox, '
      'and confidence scores.'
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

  args = parser.parse_args()

  return args

def save_mask(mask, filepath, mask_format):
  '''
  保存mask到文件，根据格式选择合适的保存方式
  '''
  if mask_format == 'labels':
    # 对于labels格式，保存为PNG（16位以支持更多类别）
    if mask.dtype != np.uint8 and mask.dtype != np.uint16:
      mask = mask.astype(np.uint16)
    cv2.imwrite(filepath, mask)
  elif mask_format == 'rgb':
    # RGB格式直接保存（BGR for OpenCV）
    cv2.imwrite(filepath, mask[..., ::-1])
  else:
    # onehot和mask格式保存为PNG
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

  # 使用高级分割函数获取更完整的结果
  if args.mask or args.pwat or args.report_path:
    # 使用新的高级分割函数
    result = wound_segmentation_advanced(
      img=rgb,
      tol=0.5,
      output_format=args.mask_format,
      verbose=args.verbose,
    )
    mask = result['mask']
    report = result['report']

    # dump the resulting mask to file
    if args.mask:
      # get the output directory
      outdir = os.path.dirname(args.filepath)
      # get the filename
      name = os.path.basename(args.filepath)
      # remove extension
      name, _ = os.path.splitext(name)

      # 根据格式确定文件扩展名
      if args.mask_format == 'labels':
        ext = '_labels.png'
      elif args.mask_format == 'onehot':
        ext = '_onehot.png'
      elif args.mask_format == 'rgb':
        ext = '_rgb.png'
      else:
        ext = '_mask.png'

      # build the output filename
      outfile = os.path.join(outdir, f'{name}_deepskin{ext}')

      # save mask according to format
      if args.mask_format == 'labels':
        cv2.imwrite(outfile, mask)
      else:
        cv2.imwrite(outfile, mask[..., ::-1])

      if args.verbose:
        print(f'Mask saved to: {outfile}')

    # 导出JSON报告
    if args.report_path:
      report.save(args.report_path)
      if args.verbose:
        print(f'Report saved to: {args.report_path}')

    # 打印报告摘要到控制台
    if args.verbose and args.report_path:
      report_data = report.generate()
      print(f'\nSegmentation Summary:')
      print(f'  - Image shape: {report_data["image_shape"]}')
      print(f'  - Wound detected: {report_data["wound_detected"]}')
      print(f'  - Pixel counts: {report_data["pixel_counts"]}')
      print(f'  - Area ratios: {report_data["area_ratios"]}')
      if 'wound_bbox' in report_data:
        print(f'  - Wound bbox: {report_data["wound_bbox"]}')
      if 'wound_confidence' in report_data:
        print(f'  - Wound confidence: {report_data["wound_confidence"]:.4f}')

  if args.pwat:
    # 对于PWAT，需要使用兼容的mask格式
    if args.mask_format != 'mask':
      # 重新获取标准mask格式用于PWAT计算
      standard_result = wound_segmentation_advanced(
        img=rgb,
        tol=0.5,
        output_format='mask',
        verbose=False,
      )
      mask_for_pwat = standard_result['mask']
    else:
      mask_for_pwat = mask

    # compute the wound PWAT
    pwat = evaluate_PWAT_score(
      img=rgb,
      mask=mask_for_pwat,
      verbose=args.verbose,
    )

    print(f'{GREEN_COLOR_CODE}PWAT prediction: {pwat:.3f}{RESET_COLOR_CODE}',
      end='\n',
      flush=True,
    )


if __name__ == '__main__':
  main ()
