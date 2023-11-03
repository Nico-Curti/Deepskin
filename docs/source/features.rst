.. _features:

Deepskin features
=================

The `deepskin` package includes a set of medical imaging features
that can be split into several classes.

Color features
--------------

We extracted the average and standard deviation of *RGB* channels for each wound and peri-wound segmentation.
This set of measures aims to quantify the appearance of the wound area in terms of redness and color heterogeneity.

We converted each masked image into the corresponding *HSV* color space. For each channel, we extracted the average and standard deviation values.
The *HSV* color space is more informative than the *RGB* one since it takes care of different light exposition (saturation).
In this way, we monitored the various conditions in which the images were acquired.

Both these two sets of features aim to quantify the necrotic tissue components of the wounds.
The necrotic tissue, indeed, could be modeled as a darker component in the wound/peri-wound area, which alters the average color of the lesion.
The *Necrotic Tissue type* and the *Total Amount of Necrotic Tissue* involve 2/8 items in the PWAT estimation.

Redness features
----------------

The primary information on the healing stage of a wound can be obtained by monitoring its redness (erythema) compared to the surrounding area.
Several redness measurements are proposed in literature, belonging to different medical fields and applications.
We extracted two measures of redness.

The first measure was proposed by Park et al. 1_, and involves a combination of the *RGB* channels, i.e.,

.. math::

  Redness_\{RGB} = 1/n \sum_{i=1}^n \frac{(2R_i - G_i - B_i)}{(2 \times (R_i + G_i + B_i))}

Where *R*, *G*, and *B* are the red, green, and blue channels of the masked image, respectively, the *n* value represents the number of pixels in the considered mask.
This measure emphasizes the R intensity using a weighted combination of the three *RGB* channels.

The second measure was proposed by Amparo et al. 2_, and involves a combination of the *HSV* channels, i.e.,

.. math::

  Redness_\{HSV} = 1/n \sum_{i=1}^n H_i \times S_i

where *H* and *S* represent the hue and saturation intensities of the masked image, respectively.
This measure tends to be more robust against different image light expositions.

Both these features were extracted on the wound and peri-wound areas independently.
Redness estimations could help to quantify the *Peri-ulcer Skin Viability*, *Granulation Tissue Type*, and *Necrotic Tissue Type*, which represent 3/8 items involved in the PWAT estimation.

Morphological features
----------------------

We measured the morphological and textural characteristics of the wound and peri-wound areas by computing the 13 Haralick's features.
Haralick's features are becoming standard texture descriptors in multiple medical image analyses, especially in the Radiomic research field.
This set of features was evaluated on the grey-level co-occurrence matrix (GLCM) associated with the grayscale versions of the original images, starting from the areas identified by our segmentation models.
We computed the 13 standard Haralick's features, given by energy, inertia, entropy, inverse difference moment, cluster shade, and cluster prominence.
Using textural elements, we aimed to quantify information related to the *Granulation Tissue types* and *Amount of Granulation Tissue*, which are 2/8 items of the total PWAT score.

.. _1: https://iovs.arvojournals.org/article.aspx?articleid=2127854
.. _2: https://iovs.arvojournals.org/article.aspx?articleid=2128418
