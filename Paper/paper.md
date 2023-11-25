---
title: 'SelfEEG: A flexible python library for electroencephalography-based self-supervised learning'
tags:
  - Python
  - Deep Learning (DL)
  - Self-Supervised Learning (SSL)
  - Contrastive Learning (CL)
  - Electroencephalography (EEG)
  - Biomedical signals
authors:
  - name: Federico Del Pup
    orcid: 0009-0004-0698-962X
    corresponding: true
    affiliation: "1, 2, 3"
  - name: Andrea Zanola
    orcid: 0000-0001-6973-8634
    equal-contrib: true
    affiliation: 3
  - name: Louis Fabrice Tshimanga
    orcid: 0009-0002-1240-4830
    equal-contrib: true
    affiliation: "2, 3"
  - name: Manfredo Atzori
    orcid: 0000-0001-5397-2063
    affiliation: "2, 3, 4"
affiliations:
 - name: Department of Information Engineering, University of Padova, Via Gradenigo 6/b, 35131 Padova, Italy
   index: 1
 - name: Department of Neuroscience, University of Padua, Via Belzoni 160, 35121 Padova, Italy
   index: 2
 - name: Padova Neuroscience Center, University of Padova, Via Orus 2/B, 35129 Padova, Italy
   index: 3
 - name: Information Systems Institute, University of Applied Sciences Western Switzerland (HES-SO Valais), 2800 Sierre, Switzerland
   index: 4
date: 01 December 2023
bibliography: bibliography.bib
---

# Summary
SelfEEG is an open-source Python library developed to assist researchers in conducting Self-Supervised Learning (SSL) experiments on electroencephalography (EEG) data. Its primary objective is to offer a user-friendly but highly customizable environment, enabling researchers to efficiently design and execute self-supervised learning tasks on EEG data. Within selfEEG, users can access various modules that cover all the stages of a typical SSL pipeline, ranging from the dataloaders creation to the model pretraining and fine-tuning.

The library also addresses key challenges identified from the analysis of previous works presenting SSL applications for biomedical signal analysis. These challenges include the need to split data at various granularity levels (e.g., trial-, session-, subject-, or dataset-based splits), effectively manage data of different formats during mini-batch construction, and provide a wide range of standard deep learning models, data augmentations and SSL baseline methods specific to EEG analysis.

Most of the functionalities offered by selfEEG can be executed on GPU devices, which expands its usability beyond the self-supervised learning area. Additionally, while the library mainly focuses on EEG data, it can be easily adapted for other biomedical signals, such as electromyography or electrocardiography, making it a versatile deep learning tool in the biomedical field.

The open-source code, examples, and instructions for installing and use selfEEG can be accessed through the [GitHub repository](https://github.com/MedMaxLab/selfEEG).

# Statement of need
Deep learning applications in the biomedical field frequently face challenges in assembling large, labeled datasets, which are needed to achieve effective training. Acquiring medical data like EEG is in fact expensive, time-consuming, and requires specific instrumentation, human volunteers, and expert neuroscientists to perform the laborious task of data annotation [@dl_promises].

Self-supervised learning (SSL) has recently emerged as a solution to this problem, as it allows extracting relevant features from vast unlabeled data. Several works have confirmed that this strategy can help to improve accuracy and mitigate overfitting in contexts with limited labeled data or multiple heterogeneous datasets to aggregate [@ericsson2022self].  Moreover, as reported in a recent survey [@DelPup2023], more than 20 works employing SSL for EEG analysis have already been published in the past few years, showing how active and proficient the research in this field is. However, despite the significant contributions, there are currently no frameworks or common standards for developing EEG-based self-supervised learning pipelines, unlike in other fields like computer vision. This lack of standardization hinders the comparison of different strategies and the progress of the field.

SelfEEG specifically aims at solving these limitations. Through its highly customizable environment, researchers can easily build an SSL pipeline, speeding up the experimental design and improving the reproducibility of their results. Moreover, the use of a common environment can enhance the comparison between different EEG-based SSL algorithms, facilitating the build of valuable benchmarks. 

# Library Overview
SelfEEG is built on top of Pytorch [@pytorch], chosen for its flexibility and high level of customization. It also requires few other standard packages to properly work. As a result, selfEEG can be easily integrated into existing environments without conflicts with other installations.

Regarding its structure, selfEEG comprises the following modules:

- **dataloading**: a collection of functions and classes to split a repository and construct an efficient Pytorch’s data loader.
- **augmentation**: a collection of data augmentation functions and classes designed to combine them, with full GPU support.
- **models**: a collection of deep learning models widely adopted for EEG-analysis.
- **losses**: a collection of self-supervised learning losses.
- **ssl**:  collection of SSL algorithms with a highly customizable fit method.
- **utils**: a collection of utility functions and classes, such as a Pytorch port of a pchip interpolator or an EEG range scaler with soft clipping.

# Related open source projects
Several deep learning frameworks have been developed for the analysis of neuroscientific data like EEG, which have been reviewed and listed in [@app13095472]. Few examples are EEG-DL [@eegdl] and [torchEEG](https://github.com/torcheeg/torcheeg), which characterized for their completeness and spread among the neuroscience community. However, a library focused on the development of self-supervised learning pipelines on EEG data is still not available to the best of our knowledge. A framework worth mentioning is [LightlySSL](https://github.com/lightly-ai/lightly), which, like selfEEG, is a self-supervised learning python framework. However, lightlySSL is designed for computer-vision applications, which, although similar in the pretraining strategy, require different designing choices in the data preparation and augmentation phases that can hinder the exportability of such a framework in the EEG domain.


# Future development
Considering how rapidly self-supervised learning is evolving, this library is expected to be constantly updated with novel self-supervised learning algorithms and deep learning models. Moreover, additional functionalities that can further improve the overall usability and enhance the comparison between different SSL strategies will be supported as well.

# CRediT Authorship Statement
FDP: Conceptualization, Writing - Original Draft, Software - Development, Software - Design, Software - Testing; AZ: Writing - Review & Editing, Software - design (dataloading and utils modules), Software - Testing; LFT: Writing - Review & Editing, Software - design (dataloading and utils modules), Software - Testing; MA: Funding Acquisition, Project Administration, Supervision, Writing - Review & Editing.

# Acknowledgements
This work was supported by the “Fondo per la promozione e lo sviluppo delle politiche del Programma Nazionale per la Ricerca (PNR) - STARS@UNIPD 2021, project: MedMax”. We would also like to thank Paolo Emilio Mazzon and the other members of the Padova Neuroscience Center for their useful suggestions during the development of the library.

# References
