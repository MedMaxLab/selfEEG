<img src="Images/LibraryLogo.png" 
        alt="Picture" 
        width="200" 
        style="display: block; margin: 0 auto" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# What is selfEEG?
-----------------
selfEEG is a pytorch-based library designed to facilitate self-supervised learning (SSL) on EEG data. In selfEEG, you can find different functions and classes which helps you build an SSL pipeline, from the creation of the dataloaders, to the model's fine-tuning, passing by the definitions of custom data augmenters, models, and pretraining strategies.
SelfEEG comprises of the following modules:

1. **dataloading**, where you can find custom pytorch Dataset and Sampler classes as well as functions to split your dataset.
2. **augmentation**, where you can find lots of already implemented data augmentation with fully support on GPU, as well as other classes designed to combine them
3. **models**, where you can find lots of already implemented models widely used in the EEG analysis (e.g. DeepConvNet, EEGNet, ResNet, TinySleepNet, STNet, etc.)
4. **ssl**, where you can find already implemented self-supervised algorhitms (e.g. SimCLR, SimSiam, MoCo, BYOL, etc) with a highly customizable fit method as well as a custom earlyStopper and a function for fine-tuning.
5. **losses**, where you can find the implementation of the SSL losses.
6. **utils**, where you can other useful functions (torch port of a pchip interpolator, a GPU compatible normalizer, etc)

What makes selfEEG good? We have designed some modules keeping in mind EEG applications, but lots of functionalities can be easily exported on other types of signal data!

However, in selfEEG you will not find functions to effectively preprocess EEG data (although filtering and resampling can be performed with some of our functions).
If you want to preprocess EEG data in a really good way, take a look at:

- **MNE** (python based)
- **EEGLAB** (matlab based)
- **BIDSAlign** (an EEGLab extension provided by our team)

## installation
---------------
SelfEEG may be installed both via pip or conda:
```
pip install selfeeg
```
```
conda install selfeeg
```
Additionally, optinal but useful packages which we suggest to include in your environment, especially if you plan to work with jupyter, can be automatically installed with the following pip or conda commands:
```
pip install selfeeg[interactive]
```
```
conda install selfeeg --install-optional
```

## Dependencies
---------------
selfEEG depends on the following packages. If you want to use selfEEG via ``git clone``, be sure to install at least the following packages:

- pandas >=1.5.3
- scipy >=1.10.1
- torchaudio >=2.0.2
- torchvision >=0.15.2
- tqdm

The following list was extracted via ``pipdeptree`` ([github repo](https://github.com/tox-dev/pipdeptree/tree/main)). Packages like ``numpy`` or ``torch`` does not appear because they are dependencies of other listed packages.

Optional packages which we suggest to include in your environment are listed as follows:

- jupyterlab
- scikit-learn
- seaborn (or simply matplotlib)

## Usage
--------
in the Notebooks folder, you can find some notebooks (with "guide" in their name) which will explain how to use some modules. These notebooks are also included in the documentation.

## Contribution Guidelines
---------------
If you'd like to contribute to selfEEG, or simply want some suggestions on how to exploit this library in your SSL experiments, please consider writing a mail to our research team

[MedMax Team](mailto:manfredo.atzori@unipd.it&cc=federico.delpup@studenti.unipd.it,andrea.zanola@studenti.unipd.it,louisfabrice.tshimanga@unipd.it)

Our team is really open to new collaborations!

## Requests and bug tracker
---------------
If you have some requests or you have noticed some bugs, use the [GitHub issues](https://github.com/MedMaxLab/selfEEG/issues) page to report them. We will try to solve reported bugs as fast as possible.

## Authors and Citation
---------------
We have worked really hard to develop this library. If you use selfEEG during your research, please cite our work. It will help us to continue doing our research.

Contributors:
- Eng. Federico Del Pup
- M.Sc. Andrea Zanola
- M.Sc. Luise Fabrice Tshimanga: 
- Prof. Manfredo Atzori

## License
---------------
see 
[MIT Licence](LICENSE.txt)












