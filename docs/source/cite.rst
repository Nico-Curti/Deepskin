.. _cite:

Cite
====

If `deepskin` has been significant in your research, and you would like to acknowledge the project in your academic publication, we suggest citing the following papers:

Research paper
--------------

Curti N, Merli Y, Zengarini C, Giampieri E, Merlotti A, Dall’Olio D, Marcelli E, Bianchi T, Castellani G. **Effectiveness of Semi-Supervised Active Learning in Automated Wound Image Segmentation.** *International Journal of Molecular Sciences*. 2023; 24(1):706. https://doi.org/10.3390/ijms24010706

Here's an example of a BibTeX entry:

.. code-block:: latex

  @article{ijms24010706,
    author = {Curti, Nico and Merli, Yuri and Zengarini, Corrado and Giampieri, Enrico and Merlotti, Alessandra and Dall'Olio, Daniele and Marcelli, Emanuela and Bianchi, Tommaso and Castellani, Gastone},
    title = {Effectiveness of Semi-Supervised Active Learning in Automated Wound Image Segmentation},
    journal = {International Journal of Molecular Sciences},
    volume = {24},
    year = {2023},
    number = {1},
    article-number = {706},
    url = {https://www.mdpi.com/1422-0067/24/1/706},
    pubmedid = {36614147},
    issn = {1422-0067},
    abstract = {Appropriate wound management shortens the healing times and reduces the management costs, benefiting the patient in physical terms and potentially reducing the healthcare system&rsquo;s economic burden. Among the instrumental measurement methods, the image analysis of a wound area is becoming one of the cornerstones of chronic ulcer management. Our study aim is to develop a solid AI method based on a convolutional neural network to segment the wounds efficiently to make the work of the physician more efficient, and subsequently, to lay the foundations for the further development of more in-depth analyses of ulcer characteristics. In this work, we introduce a fully automated model for identifying and segmenting wound areas which can completely automatize the clinical wound severity assessment starting from images acquired from smartphones. This method is based on an active semi-supervised learning training of a convolutional neural network model. In our work, we tested the robustness of our method against a wide range of natural images acquired in different light conditions and image expositions. We collected the images using an ad hoc developed app and saved them in a database which we then used for AI training. We then tested different CNN architectures to develop a balanced model, which we finally validated with a public dataset. We used a dataset of images acquired during clinical practice and built an annotated wound image dataset consisting of 1564 ulcer images from 474 patients. Only a small part of this large amount of data was manually annotated by experts (ground truth). A multi-step, active, semi-supervised training procedure was applied to improve the segmentation performances of the model. The developed training strategy mimics a continuous learning approach and provides a viable alternative for further medical applications. We tested the efficiency of our model against other public datasets, proving its robustness. The efficiency of the transfer learning showed that after less than 50 epochs, the model achieved a stable DSC that was greater than 0.95. The proposed active semi-supervised learning strategy could allow us to obtain an efficient segmentation method, thereby facilitating the work of the clinician by reducing their working times to achieve the measurements. Finally, the robustness of our pipeline confirms its possible usage in clinical practice as a reliable decision support system for clinicians.},
    doi = {10.3390/ijms24010706}
  }

Code reference
--------------

Nico Curti, **deepskin pipeline - Wound analysis using smartphone images**, *Github*, https://github.com/Nico-Curti/Deepskin

Here's an example of a BibTeX entry:

.. code-block:: latex

  @misc{deepskin,
    author = {Curti, Nico},
    title = {{deepskin pipeline}: Wound analysis using smartphone images},
    year = {2023},
    url = {https://github.com/Nico-Curti/Deepskin},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/Nico-Curti/Deepskin}}
  }

For any specific algorithm, also consider citing the code refence's paper.
