<p align="center">
  <img src="https://github.com/Nico-Curti/Deepskin/blob/main/img/logo.png" width="90" height="90">
  <br>
</p>

| **Authors**  | **Project** |  **Documentation** | **Build Status** | **License** |
|:------------:|:-----------:|:------------------:|:----------------:|:-----------:|
| [**N. Curti**](https://github.com/Nico-Curti) | **deepskin**<br/>[![International Journal of Molecular Science](https://www.mdpi.com/1422-0067/24/1/706)](https://www.mdpi.com/1422-0067/24/1/706) | **TODO** | [![Python](https://github.com/Nico-Curti/Deepskin/actions/workflows/python.yml/badge.svg)](https://github.com/Nico-Curti/Deepskin/actions/workflows/python.yml) | [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Nico-Curti/Deepskin/blob/main/LICENSE) |

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/Nico-Curti/deepskin.svg?style=plastic)](https://github.com/Nico-Curti/Deepskin/pulls)
[![GitHub issues](https://img.shields.io/github/issues/Nico-Curti/deepskin.svg?style=plastic)](https://github.com/Nico-Curti/Deepskin/issues)

[![GitHub stars](https://img.shields.io/github/stars/Nico-Curti/deepskin.svg?label=Stars&style=social)](https://github.com/Nico-Curti/Deepskin/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Nico-Curti/deepskin.svg?label=Watch&style=social)](https://github.com/Nico-Curti/Deepskin/watchers)

<a href="https://github.com/UniboDIFABiophysics">
  <div class="image">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
</a>

# Deepskin

## Wound analysis using smartphone images

Official implementation of the deepskin algorithm published on [International Journal of Molecular Science](https://www.mdpi.com/1422-0067/24/1/706) by Curti et al. [![International Journal of Molecular Science](https://img.shields.io/badge/IJMS-1422_0067/24/1/706-g.svg)](https://www.mdpi.com/1422-0067/24/1/706)

* [Overview](#overview)
* [Theory](#theory)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Efficiency](#efficiency)
* [Usage](#usage)
* [Testing](#testing)
* [Table of contents](#table-of-contents)
* [Contribution](#contribution)
* [References](#references)
* [FAQ](#faq)
* [Authors](#authors)
* [License](#license)
* [Acknowledgment](#acknowledgment)
* [Citation](#citation)

## Overview

TODO

## Theory

TODO

## Prerequisites

TODO

## Installation

TODO

## Efficiency

TODO

## Usage

You can use the `deepskin` package

```python

```

## Testing

TODO

## Table of contents

Description of the folders related to the project

TODO

## Contribution

Any contribution is more than welcome :heart:. Just fill an [issue](https://github.com/Nico-Curti/Deepskin/blob/main/.github/ISSUE_TEMPLATE/ISSUE_TEMPLATE.md) or a [pull request](https://github.com/Nico-Curti/Deepskin/blob/main/.github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md) and we will check ASAP!

See [here](https://github.com/Nico-Curti/Deepskin/blob/main/.github/CONTRIBUTING.md) for further informations about how to contribute with this project.

## References

<blockquote> </blockquote>

## FAQ

TODO

## Authors

* <img src="https://avatars0.githubusercontent.com/u/24650975?s=400&v=4" width="25px"> **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)
* <img src="https://avatars2.githubusercontent.com/u/1419337?s=400&v=4" width="25px;"/> **Enrico Giampieri** [git](https://github.com/EnricoGiampieri), [unibo](https://www.unibo.it/sitoweb/enrico.giampieri)

See also the list of [contributors](https://github.com/Nico-Curti/Deepskin/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Nico-Curti/Deepskin.svg?style=plastic)](https://github.com/Nico-Curti/Deepskin/graphs/contributors/) who participated in this project.

## License

The `deepskin` package is licensed under the MIT "Expat" License. [![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Nico-Curti/Deepskin/blob/main/LICENSE)

### Acknowledgment

Thanks goes to all contributors of this project.

### Citation

If you have found `deepskin` helpful in your research, please consider citing the original paper about the wound image segmentation

```BibTex
@article{ijms24010706,
  author = {Curti, Nico and Merli, Yuri and Zengarini, Corrado and Giampieri, Enrico and Merlotti, Alessandra and Dall’Olio, Daniele and Marcelli, Emanuela and Bianchi, Tommaso and Castellani, Gastone},
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
```
or just this repository

```BibTex
@misc{deepskin,
  author = {Curti, Nico},
  title = {{deepskin pipeline}: Wound analysis using smartphone images},
  year = {2023},
  url = {https://github.com/Nico-Curti/Deepskin},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Nico-Curti/Deepskin}}
}
```
