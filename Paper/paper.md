---
title: 'SelfEEG: A Python library for Self-Supervised Learning in Electroencephalography'

tags:
  - Python
  - PyTorch
  - Deep Learning (DL)
  - Self-Supervised Learning (SSL)
  - Contrastive Learning (CL)
  - Electroencephalography (EEG)
  - Biomedical signals
authors:
  - name: 
      given-names: Federico 
      surname: Del Pup
    orcid: 0009-0004-0698-962X
    corresponding: true
    affiliation: "1, 2, 3"
  - name: 
      given-names: Andrea 
      surname: Zanola
    orcid: 0000-0001-6973-8634
    equal-contrib: true
    affiliation: 3
  - name: 
      given-names: Louis Fabrice
      surname: Tshimanga
    orcid: 0009-0002-1240-4830
    equal-contrib: true
    affiliation: "2, 3"
  - name: 
      given-names: Paolo Emilio
      surname: Mazzon
    affiliation: 3
  - name: 
      given-names: Manfredo
      surname: Atzori
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
SelfEEG is an open-source Python library developed to assist researchers in conducting Self-Supervised Learning (SSL) experiments on electroencephalography (EEG) data. 
Its primary objective is to offer a user-friendly but highly customizable environment, enabling users to efficiently design and execute self-supervised learning tasks on EEG data. 

SelfEEG covers all the stages of a typical SSL pipeline, ranging from data import to model design and training. 
It includes modules specifically designed to: split data at various granularity levels (e.g., session-, subject-, or dataset-based splits); effectively manage data stored with different configurations (e.g., file extensions, data types) during mini-batch construction; provide a wide range of standard deep learning models, data augmentations and SSL baseline methods applied to EEG data.

Most of the functionalities offered by selfEEG can be executed both on GPUs and CPUs, expanding its usability beyond the self-supervised learning area. 
Additionally, these functionalities can be employed for the analysis of other biomedical signals often coupled with EEGs, such as electromyography or electrocardiography data.

These features make selfEEG a versatile deep learning tool for biomedical applications and a useful resource in SSL, one of the currently most active fields of Artificial Intelligence.


# Statement of need
SelfEEG answers to the lack of Self-Supervised Learning (SSL) frameworks for the analysis of EEG data. 

In fact, despite the recent high number of publications (more than 20 journal papers in the last 4 years [@DelPup2023]), there are currently no frameworks or common standards for developing EEG-based SSL pipelines, contrary to other fields such as computer vision (see [LightlySSL](https://github.com/lightly-ai/lightly) or [ViSSL](https://github.com/facebookresearch/vissl)).

In the field of EEG data analysis, where it has been demonstrated that SSL can improve models’ accuracy and mitigate overfitting [@eegrafiei] [@banville], the absence of a self-supervised learning framework dedicated to EEG signals limits the development of novel strategies, reproducibility of results, and the progress of the field.

Thanks to selfEEG, researchers can instead easily build SSL pipelines, speeding up experimental design and improving the reproducibility of results. Reproducibility is a key factor in this area, as it enhances the comparison of different strategies and supports the creation of useful benchmarks.

SelfEEG was also developed considering the needs of deep learning researchers, for whom this library has been primarily designed. For this reason, selfEEG aims to preserve a high but easily manageable level of customization.


# Library Overview
SelfEEG is a comprehensive library for SSL applications to EEG data. It is built on top of PyTorch [@pytorch] and it includes several modules targeting all the steps required for developing EEG-based SSL pipelines.
In particular, selfEEG comprises the following modules:

- **dataloading**: a collection of functions and classes designed to support data splitting and the construction of efficient PyTorch dataloaders in the EEG context.
- **augmentation**: a collection of EEG data augmentation functions and other classes designed to combine them in more complex patterns.
- **models**: a collection of EEG deep learning models.
- **losses**: a collection of self-supervised learning losses.
- **ssl**:  a collection of self-supervised learning algorithms applied to EEG analysis with highly customizable fit methods.
- **utils**: a collection of utility functions and classes for various purposes, such as a PyTorch compatible EEG sampler and scaler.


# Related open-source projects
Despite several deep learning frameworks were developed for the analysis of EEG data, a library focused on the construction of self-supervised learning pipelines on EEG data is still not available to the best of our knowledge, hindering the advancement of the scientific knowledge and the progress in the field.
A comprehensive review of open-source projects related to neuroscientific data analysis is provided in [@app13095472]. 
Few examples are EEG-DL [@eegdl] and [torchEEG](https://github.com/torcheeg/torcheeg), which characterized for their completeness and spread among the neuroscientific community. 


# Future development
Considering how rapidly self-supervised learning is evolving, this library is expected to be constantly updated by the authors and the open-source community, especially by adding novel SSL algorithms, deep learning models, and functionalities that can enhance the comparison between different developed strategies. 
In particular, the authors plan to continue working on SelfEEG during the next years via several ongoing European and national projects.


# CRediT Authorship Statement
FDP: Conceptualization, Writing - Original Draft, Software - Development, Software - Design, Software - Testing; 
AZ: Writing - Review & Editing, Software - design (dataloading and utils modules), Software - Testing; 
LFT: Writing - Review & Editing, Software - design (dataloading and utils modules), Software - Testing;
PEM: Technical support, Writing - Review & Editing, Software - Testing; 
MA: Funding Acquisition, Project Administration, Supervision, Writing - Review & Editing.

# Acknowledgements
This work was supported by the STARS@UNIPD funding program of the University of Padova, Italy, through the project: MEDMAX.
This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement no 101137074 - HEREDITARY.
We would also like to thank the other members of the Padova Neuroscience Center for their support during the project development.

# References
