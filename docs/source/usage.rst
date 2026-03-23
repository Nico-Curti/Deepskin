.. _usage:

Usage
=====

You can use the `deepskin` library into your Python scripts or directly via command line.

Command Line Interface
----------------------

The `deepskin` package could be easily used via command line by simply calling the `deepskin` program.

The full list of available flags for the customization of the command line could be obtained by calling:

.. code-block:: bash

  $ deepskin --help

  usage: deepskin [-h] [--version] [--input FILEPATH] [--verbose] [--mask] [--mask-format {labels,onehot,rgb,mask}] [--report REPORT_PATH] [--pwat]

  deepskin library - Wound analysis using smartphone images

  optional arguments:
    -h, --help            show this help message and exit
    --version, -v         Get the current version installed
    --input FILEPATH, -i FILEPATH
                          Input filename or path on which load the image. Ref
                          https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html for the list of supported formats.
    --verbose, -w         Enable/Disable the code logging
    --mask, -m            Evaluate the semantic segmentation mask using the Deepskin model; the resulting mask will be
                          saved to a png file in the same location of the input file
    --mask-format {labels,onehot,rgb,mask}
                          Output format for the segmentation mask. labels: 2D integer label map (0=background, 1=body,
                          2=wound); onehot: 3-channel one-hot encoded mask; rgb: RGB color map for visualization; mask:
                          3-channel binary mask (default, compatible with PWAT)
    --report REPORT_PATH  Export segmentation analysis report to JSON file. The report includes pixel counts, area
                          ratios, wound bbox, and confidence scores.
    --pwat, -p            Compute the PWAT score of the given wound-image

  Deepskin Python package v0.0.1

Python script
-------------

A complete list of beginner-examples for the usage of the `deepskin` tools could be found in the example notebooks_,
in which are reported examples about the possible usage of the package and real-world examples.

For sake of completeness, a simple `deepskin` pipeline could be obtained by the following snippet:

.. code-block:: python

  import cv2
  from deepskin import wound_segmentation
  from deepskin import evaluate_PWAT_score

  # load the image in any OpenCV supported fmt
  bgr = cv2.imread('/path/to/picture.png')
  # convert the image from BGR to RGB fmt
  rgb = bgr[..., ::-1]
  # get the wound segmentation mask
  wound_mask = wound_segmentation(img=rgb)
  # compute the wound PWAT
  pwat = evaluate_PWAT_score(img=rgb, mask=wound_mask)
  # display the results
  print(f'PWAT score: {pwat:.3f}')

Advanced Usage with Multiple Output Formats
-------------------------------------------

The `wound_segmentation_advanced` function provides more control over the output format and generates detailed analysis reports:

.. code-block:: python

  import cv2
  from deepskin import wound_segmentation_advanced

  # load the image
  bgr = cv2.imread('/path/to/picture.png')
  rgb = bgr[..., ::-1]

  # perform segmentation with different output formats
  result = wound_segmentation_advanced(
      img=rgb,
      output_format='labels',  # 'labels', 'onehot', 'rgb', or 'mask'
      verbose=True
  )

  # access results
  label_map = result['label_map']      # 2D integer label map
  mask = result['mask']                 # formatted according to output_format
  report = result['report']             # SegmentationReport object

  # export analysis report to JSON
  report.save('segmentation_report.json')

  # or get report as dictionary
  data = report.generate()
  print(f"Wound detected: {data['wound_detected']}")
  print(f"Wound area ratio: {data['area_ratios']['wound']:.4f}")
  if 'wound_bbox' in data:
      print(f"Wound bbox: {data['wound_bbox']}")

Output Formats
--------------

The segmentation output supports multiple formats:

- **labels**: 2D integer label map where each pixel value represents the class (0=background, 1=body, 2=wound)
- **onehot**: 3-channel one-hot encoded mask, each channel corresponds to one class
- **rgb**: RGB color map for visualization (background=blue, body=green, wound=red)
- **mask**: 3-channel binary mask (default, compatible with PWAT calculation)

Segmentation Report
-------------------

The `SegmentationReport` class provides detailed analysis of the segmentation results:

.. code-block:: python

  from deepskin import SegmentationReport

  # create report from label map
  report = SegmentationReport(label_map, softmax_output)

  # get report as dictionary
  data = report.generate()

  # save to JSON file
  report.save('report.json')

The report includes:

- **image_shape**: Shape of the input image
- **total_pixels**: Total number of pixels
- **pixel_counts**: Pixel count for each class
- **area_ratios**: Area ratio for each class
- **wound_detected**: Boolean indicating if wound was detected
- **wound_bbox**: Bounding box of wound region (if detected)
- **wound_confidence**: Average confidence score for wound region (if softmax provided)

.. _here: https://github.com/Nico-Curti/graphomics/blob/main/examples
.. _notebooks: https://github.com/Nico-Curti/graphomics/blob/main/docs/source/notebooks
