.. graphomics package documentation master file

.. image:: ../../img/logo.png
   :width: 90

Deepskin
========

This is an open-source python package for the analysis of
wound images acquired via smartphone camera.

With this package we aim to propose a new reference for
Wound Image Analysis given by the automated segmentation
of wound area framework, combined with a series of image
processing algorithm for the quantification of clinical
outomes.

The package supports the feature extraction of 2D images
and can be used to calculate features according
to different classes.

.. note::

   **Not intended for clinical use.**

Overview
========

Official implementation of the deepskin algorithm published on International Journal of Molecular Science by Curti et al. [1_]

The `deepskin` package aims to propose a fully automated pipeline for the wound-image processing

The first step of the `deepskin` pipeline involves the automated identification of the wound ROI in the image.
This task is address using a deep learning U-Net model trained on a large set of images.
The model training was performed using an active semi-supervised learning strategy (ASSL), given by the following step of processing:

1. The images acquired using a smartphone were stored into the training dataset.
2. Starting with a small set of annotated images (not included into the scheme), we trained from scratch a neural network model for the wound segmentation.
3. All of the unlabeled images were used as validation set, and the generated masks were provided by the expert.
4. The expert analyzed the produced segmentation according to a predetermined evaluation criterion.
5. The masks which satisfied the criteria would be added as ground truth for the next round of training.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   features
   API/modules
   examples
   references
   contributing
   credits
   cite

.. _1: https://www.mdpi.com/1422-0067/24/1/706
