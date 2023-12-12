<img src="Images/LibraryLogo.png" 
        alt="Picture" 
        width="200" 
        style="display: block; margin: 0 auto" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# What is selfEEG?
selfEEG is a pytorch-based library designed to facilitate self-supervised learning (SSL) experiments on electroencephalography (EEG) data. In selfEEG, you can find different functions and classes which will help you build an SSL pipeline, from the creation of the dataloaders, to the model's fine-tuning, passing by the definitions of custom data augmenters, models, and pretraining strategies.
In particular, selfEEG comprises of the following modules:

1. **dataloading** - collection of custom pytorch Dataset and Sampler classes as well as functions to split your dataset.
2. **augmentation** - collection of data augmentation with fully support on GPU as well as other classes designed to combine them
3. **models** - collection of deep neural models widely used in the EEG analysis (e.g. DeepConvNet, EEGNet, ResNet, TinySleepNet, STNet, etc)
4. **ssl** - collection of self-supervised algorithms (e.g. SimCLR, SimSiam, MoCo, BYOL, etc) with a highly customizable fit method as well as a custom earlyStopper and a function for fine-tuning.
5. **losses** - collection of self-supervised learning losses.
6. **utils** - other useful functions to manage EEG data

What makes selfEEG good? We have designed some modules keeping in mind EEG applications, but lots of functionalities can be easily exported on other types of signal as well!

What will you not find in selfEEG? SelfEEG isn't an EEG preprocessing library. You will not find functions to preprocess EEG data in the best possible way (no IC rejection or ASR). However, some simple operations like filtering and resampling can be performed with functions implemented in the utils and augmentation modules.
If you want to preprocess EEG data in a really good way, we suggest to take a look at:

- **MNE** (python based)
- **EEGLAB** (matlab based)
- **BIDSAlign** (an EEGLab extension provided by our team)


## installation
SelfEEG may be installed both via pip:
```
pip install selfeeg
```
Additionally, optinal but useful packages which we suggest to include in your environment, especially if you plan to work with jupyter, can be automatically installed with the following pip command:
```
pip install selfeeg[interactive]
```

## Dependencies
selfEEG requires the following packages to correctly work. If you want to use selfEEG by forking and cloning the project, be sure to install them:

- pandas >=1.5.3
- scipy >=1.10.1
- torch >= 2.0.0
- torchaudio >=2.0.2
- torchvision >=0.15.2
- tqdm

The following list was extracted via ``pipdeptree`` ([github repo](https://github.com/tox-dev/pipdeptree/tree/main)). Packages like ``numpy`` doesn't appear because they are dependencies of other listed packages.

Optional packages which we suggest to include in your environment are:

- jupyterlab
- scikit-learn
- seaborn (or simply matplotlib)


## Usage
in the Notebooks folder, you can find some notebooks which will explain how to properly use some modules. These notebooks are also included in the **official documentation**.


## Contribution Guidelines
If you'd like to **contribute** to selfEEG, please take a look at our [contributing guidelines](CONTRIBUTING.md).

If you also have suggestions regarding novel features to add, or simply want some **support** or **suggestions** on how to exploit this library in your SSL experiments, please consider writing to our research team.

[MedMax Team](mailto:manfredo.atzori@unipd.it&cc=federico.delpup@studenti.unipd.it,andrea.zanola@studenti.unipd.it,louisfabrice.tshimanga@unipd.it)

Our team is open to new collaborations!


## Requests and bug tracker
If you have some requests or you have noticed some bugs, use the [GitHub issues](https://github.com/MedMaxLab/selfEEG/issues) page to report them. We will try to solve reported major bugs as fast as possible.


## Authors and Citation
We have worked really hard to develop this library. If you use selfEEG during your research, please cite our work. It will help us to continue doing our research. We are working on a research paper to submit to a journal. Until then, you can cite the following work.

```bibtex
@article{del2023applications,
    title={Applications of Self-Supervised Learning to Biomedical Signals: where are we now},
    author={Del Pup, Federico and Atzori, Manfredo},
    year={2023},
    publisher={TechRxiv}
}
```

Contributors:
- Eng. Federico Del Pup
- M.Sc. Andrea Zanola
- M.Sc. Louis Fabrice Tshimanga
- Eng. Paolo Emilio Mazzon
- Prof. Manfredo Atzori

## License
SelfEEG is released under the
[MIT Licence](LICENSE.md)












