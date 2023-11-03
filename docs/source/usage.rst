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

  usage: deepskin [-h] [--version] [--input FILEPATH] [--verbose]

  deepskin library - Wound analysis using smartphone images

  optional arguments:
    -h, --help            show this help message and exit
    --version, -v         Get the current version installed
    --input FILEPATH, -i FILEPATH
                          Input filename or path on which load the image. Ref
                          https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html for the list of supported formats.
    --verbose, -w         Enable/Disable the code logging

  Deepskin Python package v0.0.1

Python script
-------------

A complete list of beginner-examples for the usage of the `deepskin` tools could be found in the example notebooks_,
in which are reported examples about the possible usage of the package and real-world examples.

For sake of completeness, a simple `deepskin` pipeline could be obtained by the following snippet:

.. code-block:: python

  from deepskin import deepskin_model
  from deepskin import evaluate_PWAT_score

  # load the image in any OpenCV supported fmt
  bgr = cv2.imread('/path/to/picture.png')
  # convert the image from BGR to RGB fmt
  rgb = bgr[..., ::-1]
  # get the wound segmentation mask
  wound_mask = wound_segmentation(img=rgb)
  # compute the wound PWAT
  pwat = evaluate_PWAT_score(img=rgb, wound_mask=wound_mask)
  # display the results
  print(f'PWAT score: {pwat:.3f}')

.. _here: https://github.com/Nico-Curti/graphomics/blob/main/examples
.. _notebooks: https://github.com/Nico-Curti/graphomics/blob/main/docs/source/notebooks
