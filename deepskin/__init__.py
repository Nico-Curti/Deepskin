#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .__version__ import __version__
# import the segmentation model
from .model import deepskin_model
# import the features for the wound monitoring
from .features import evaluate_features
# import the PWAT evaluator for the wound scoring
from .pwat import evaluate_PWAT_score

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']
