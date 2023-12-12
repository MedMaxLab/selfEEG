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
  - name: Paolo Emilio Mazzon
    affiliation: 3
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
SelfEEG is an open-source Python library developed to assist researchers in conducting Self-Supervised Learning (SSL) experiments on electroencephalography (EEG) data. 
Its primary objective is to offer a user-friendly but highly customizable environment, enabling users to efficiently design and execute self-supervised learning tasks on EEG data. 

SelfEEG covers all the stages of a typical SSL pipeline, ranging from data import to model design and training. 
It includes modules specifically designed to: split data at various granularity levels (e.g., session-, subject-, or dataset-based splits); effectively manage data stored with different configurations (e.g., file extensions, data types) during mini-batch construction; provide a wide range of standard deep learning models, data augmentations and SSL baseline methods applied to EEG data.

Most of the functionalities offered by selfEEG can be executed both on GPUs and CPUs, expanding its usability beyond the self-supervised learning area. 
Additionally, these functionalities can be employed for the analysis of other biomedical signals often coupled with EEGs, such as electromyography or electrocardiography data.

These features make selfEEG a versatile deep learning tool for biomedical applications and a useful resource in SSL, one of the currently most active fields of Artificial Intelligence.


# Statement of need 
SelfEEG answers to the lack of a Self-Supervised Learning (SSL) framework for EEG data.

Despite recent contributions in relation to this topic, there are currently no frameworks or common standards for developing EEG-based SSL pipelines, contrary to other fields like computer vision (see [LightlySSL](https://github.com/lightly-ai/lightly) or [ViSSL](https://github.com/facebookresearch/vissl)). 
The absence of a framework dedicated to EEG data is a limit for the development of novel strategies and the progress of the field.

By using SelfEEG, researchers can easily build an SSL pipeline, speeding up the experimental design and improving results' reproducibility, a key feature for enhancing the comparison between different strategies and supporting the creation of valuable benchmarks.

Self-supervised learning is an unsupervised learning approach that learns representations from unlabeled data, exploiting their intrinsic structure to provide supervision [@banville].
In the past 5 years, over 20 works applied SSL to EEG data analysis [@DelPup2023], demonstrating how this strategy can improve model’s accuracy and mitigate overfitting, especially when there is a limited amount of labeled data [@eegrafiei].

SelfEEG was developed by considering the needs of deep learning researchers, for whom this library is mainly designed. For this reason, the library still retains a high but easily manageable level of customizability.


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
