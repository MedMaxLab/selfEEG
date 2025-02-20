<h1 align="center">
  <img src="https://github.com/MedMaxLab/selfEEG/blob/main/Images/LibraryLogo.png" width="300">
</h1><br>

[![PyPI](https://img.shields.io/pypi/v/selfeeg?label=PyPI&color=blue)](https://pypi.org/project/selfeeg/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/selfeeg.svg?color=blue)](https://anaconda.org/conda-forge/selfeeg)
[![Docs](https://img.shields.io/readthedocs/selfeeg)](https://readthedocs.org/projects/selfeeg/)
[![Unittest](https://github.com/MedMaxLab/selfEEG/actions/workflows/python-app.yml/badge.svg)](https://github.com/MedMaxLab/selfEEG/actions/workflows/python-app.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06224/status.svg)](https://doi.org/10.21105/joss.06224)
[![License](https://img.shields.io/badge/License-MIT-violet.svg)](https://github.com/MedMaxLab/selfEEG/blob/main/LICENSE.md)

# What is selfEEG?
selfEEG is a pytorch-based library designed to facilitate self-supervised learning
(SSL) experiments on electroencephalography (EEG) data.
In selfEEG, you can find different functions and classes that will help you build
an SSL pipeline, from the creation of the dataloaders to the model's fine-tuning,
covering other important aspects such as the definitions of custom data augmenters,
models, and pretraining strategies.
In particular, selfEEG comprises of the following modules:

1. **dataloading** - collection of custom pytorch Dataset and Sampler classes
   as well as functions to split your dataset.
3. **augmentation** - collection of data augmentation with fully support on GPU
   as well as other classes designed to combine them.
5. **models** - collection of deep neural models widely used in the EEG analysis
   (e.g., DeepConvNet, EEGNet, ResNet, TinySleepNet, STNet, etc)
7. **ssl** - collection of self-supervised algorithms with a highly customizable
   fit method  (e.g., SimCLR, SimSiam, MoCo, BYOL, etc) and other useful objects
   such as a custom earlyStopper or a fine-tuning function.
9. **losses** - collection of self-supervised learning losses.
10. **utils** - other useful functions to manage EEG data.

What makes selfEEG good? We have designed some modules keeping in mind EEG
applications, but lots of functionalities can be easily exported on other
types of signal as well!

What will you not find in selfEEG? SelfEEG isn't an EEG preprocessing library.
You will not find functions to preprocess EEG data in the best possible way
(no IC rejection or ASR). However, some simple operations like filtering and
resampling can be performed with functions implemented in the utils and
augmentation modules. If you want to preprocess EEG data in a really good way,
we suggest to take a look at:

- [**MNE**](https://mne.tools) (python based)
- [**EEGLAB**](https://sccn.ucsd.edu/eeglab) (matlab based)
- [**BIDSAlign**](https://github.com/MedMaxLab/BIDSAlign)
  (an EEGLab extension provided by our team)


## installation
SelfEEG may be installed via pip (recommended):
```
pip install selfeeg
```

SelfEEG can be also installed via conda by running the following command:
```
conda install conda-forge::selfeeg
```

Additionally, optional but useful packages that we suggest to include in your
environment, especially if you plan to work with jupyter, can be automatically
installed with the following pip command:
```
pip install selfeeg[interactive]
```

**Good practices**

Although the dependency list is pretty short, it is strongly suggested to install
selfEEG in a fresh environment. The following links provide a guide for creating a
new Python virtual environment or a new conda environment:

1. [new virtual environment](https://docs.python.org/3/library/venv.html)
2. [new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

In addition, if PyTorch, Torchvision and Torchaudio are not present in your
environment, the previous commands will install the CPU_only versions of such
packages.
If you have CUDA installed on your system, we strongly encourage you to first
install PyTorch, Torchvision and Torchaudio by choosing the
right configuration, which varies depending on your OS and CUDA versions;
then, install selfEEG.
The official PyTorch documentation provides an installation command selector,
which is available at the following [link](https://pytorch.org/get-started/locally/).



## Dependencies
selfEEG requires the following packages to correctly work.
If you want to use selfEEG by forking and cloning the project,
be sure to install them:

- pandas >=1.5.3
- scipy >=1.10.1
- torch >= 2.0.0
- torchaudio >=2.0.2
- torchvision >=0.15.2
- tqdm

The following list was extracted via
[pipdeptree](https://github.com/tox-dev/pipdeptree/tree/main).
Packages like ``numpy`` does not appear because they are dependencies
of other listed packages.

Optional packages which we suggest to include in your environment are:

- jupyterlab
- scikit-learn
- seaborn (or simply matplotlib)
- MNE-Python


## Usage
in the Notebooks folder, you can find some notebooks which will explain how to
properly use some modules.
These notebooks are also included in the
[official documentation](https://selfeeg.readthedocs.io/en/latest/index.html).


## Contribution Guidelines
If you'd like to **contribute** to selfEEG,
please take a look at our [contributing guidelines](CONTRIBUTING.md).

If you also have **suggestions** regarding novel features to add, or simply
want some **support**, please consider writing to our research team.

[MedMax Team](mailto:manfredo.atzori@unipd.it&cc=federico.delpup@studenti.unipd.it,andrea.zanola@studenti.unipd.it,louisfabrice.tshimanga@unipd.it)

Our team is open to new collaborations!


## Requests and bug tracker
If you have some requests or you have noticed some bugs,
use the [GitHub issues](https://github.com/MedMaxLab/selfEEG/issues) page to report
them. We will try to solve reported major bugs as fast as possible.


## Authors and Citation
We have worked really hard to develop this library.
If you use selfEEG during your research, please cite
[our work](https://doi.org/10.21105/joss.06224) published in the Journal of
Open Source Software (JOSS).
It would help us to continue our research.

```bibtex
@article{DelPup2024,
  title = {SelfEEG: A Python library for Self-Supervised Learning in Electroencephalography},
  author = {Del Pup, Federico and
            Zanola, Andrea and
            Tshimanga, Louis Fabrice and
            Mazzon, Paolo Emilio and
            Atzori, Manfredo},
  year = {2024},
  publisher = {The Open Journal},
  journal = {Journal of Open Source Software},
  volume = {9},
  number = {95},
  pages = {6224},
  doi = {10.21105/joss.06224},
  url = {https://doi.org/10.21105/joss.06224}
}
```

Contributors:
- Eng. Federico Del Pup
- M.Sc. Andrea Zanola
- M.Sc. Louis Fabrice Tshimanga
- Eng. Paolo Emilio Mazzon
- Prof. Manfredo Atzori

## License
SelfEEG is released under the [MIT Licence](LICENSE.md)
