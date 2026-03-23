#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .__version__ import __version__
# import the segmentation model
from .model import deepskin_model
# import useful constant values
from .model import MODEL_CHECKPOINT
# import model checkpoint getter
from .checkpoints import download_model_weights
# import the wound segmentation algorithm
from .segmentation import wound_segmentation, wound_segmentation_advanced
# import the features for the wound monitoring
from .features import evaluate_features
# import the PWAT evaluator for the wound scoring
from .pwat import evaluate_PWAT_score
# import post-processing functions
from .postprocess import (
    softmax_to_label_map,
    label_map_to_onehot,
    label_map_to_rgb,
    label_map_to_mask,
    SegmentationReport,
    convert_mask_format,
    CLASS_NAMES,
    CLASS_COLORS,
    CLASS_INDICES,
)

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']
