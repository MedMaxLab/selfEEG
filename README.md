<img src="Images/LibraryLogo.png"
        alt="Picture"
        width="300"
        style="display: block; margin: 0 auto" />

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/selfeeg/">
    <img src="https://img.shields.io/pypi/v/selfeeg" alt="latest release" />
    </a>
    <a href="https://anaconda.org/pup_fede_cnd/selfeeg">
    <img src="https://anaconda.org/pup_fede_cnd/selfeeg/badges/version.svg" />
    </a>
    <a href="https://anaconda.org/pup_fede_cnd/selfeeg">
    <img src="https://anaconda.org/pup_fede_cnd/selfeeg/badges/latest_release_date.svg" />
    </a>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://img.shields.io/readthedocs/selfeeg">
    <img src="https://img.shields.io/readthedocs/selfeeg" alt="documentation build status" />
    </a>
    <a href="https://github.com/MedMaxLab/selfEEG/actions/workflows/python-app.yml">
      <img src="https://github.com/MedMaxLab/selfEEG/actions/workflows/python-app.yml/badge.svg" alt="GitHub Actions Testing Status" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/MedMaxLab/selfEEG/blob/main/LICENSE.md">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg"
        alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Publications</td>
  <td>
    <a href="https://joss.theoj.org/papers/ab7eaf53973996e7c8d49dada734de78">
        <img src="https://joss.theoj.org/papers/ab7eaf53973996e7c8d49dada734de78/status.svg">
    </a>
  </td>
</tr>
</table>


# What is selfEEG?
selfEEG is a pytorch-based library designed to facilitate self-supervised learning (SSL) experiments on electroencephalography (EEG) data. In selfEEG, you can find different functions and classes which will help you build an SSL pipeline, from the creation of the dataloaders, to the model's fine-tuning, passing by the definitions of custom data augmenters, models, and pretraining strategies.
In particular, selfEEG comprises of the following modules:

1. **dataloading** - collection of custom pytorch Dataset and Sampler classes as well as functions to split your dataset.
2. **augmentation** - collection of data augmentation with fully support on GPU as well as other classes designed to combine them.
3. **models** - collection of deep neural models widely used in the EEG analysis (e.g. DeepConvNet, EEGNet, ResNet, TinySleepNet, STNet, etc)
4. **ssl** - collection of self-supervised algorithms (e.g. SimCLR, SimSiam, MoCo, BYOL, etc) with a highly customizable fit method as well as a custom earlyStopper and a function for fine-tuning.
5. **losses** - collection of self-supervised learning losses.
6. **utils** - other useful functions to manage EEG data.

What makes selfEEG good? We have designed some modules keeping in mind EEG applications, but lots of functionalities can be easily exported on other types of signal as well!

What will you not find in selfEEG? SelfEEG isn't an EEG preprocessing library. You will not find functions to preprocess EEG data in the best possible way (no IC rejection or ASR). However, some simple operations like filtering and resampling can be performed with functions implemented in the utils and augmentation modules.
If you want to preprocess EEG data in a really good way, we suggest to take a look at:

- **MNE** (python based)
- **EEGLAB** (matlab based)
- **BIDSAlign** (an EEGLab extension provided by our team)


## installation
SelfEEG may be installed via pip (recommended):
```
pip install selfeeg
```
Additionally, optional but useful packages which we suggest to include in your environment, especially if you plan to work with jupyter, can be automatically installed with the following pip command:
```
pip install selfeeg[interactive]
```

SelfEEG can be also installed via conda by running the following command:
```
conda install -c Pup_Fede_Cnd -c pytorch selfeeg
```

**Good practice**

Although the dependency list is pretty short, it is strongly suggested to install selfEEG in a fresh environment. The following links provide a guide for creating a new Python virtual environment or a new conda environment:

1. [new virtual environment](https://docs.python.org/3/library/venv.html)
2. [new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

In addition, if PyTorch, Torchvision and Torchaudio are not present in your environment, the previous commands will install the CPU_only versions of such packages.
If you have CUDA installed on your system, we strongly encourage you to first install PyTorch, Torchvision and Torchaudio by choosing the
right configuration, which varies depending on your OS and CUDA versions; then install selfEEG. The official PyTorch documentation provides an installation command selector, which is available at the following [link](https://pytorch.org/get-started/locally/).



## Dependencies
selfEEG requires the following packages to correctly work. If you want to use selfEEG by forking and cloning the project, be sure to install them:

- pandas >=1.5.3
- scipy >=1.10.1
- torch >= 2.0.0
- torchaudio >=2.0.2
- torchvision >=0.15.2
- tqdm

The following list was extracted via ``pipdeptree`` ([github repo](https://github.com/tox-dev/pipdeptree/tree/main)). Packages like ``numpy`` does not appear because they are dependencies of other listed packages.

Optional packages which we suggest to include in your environment are:

- jupyterlab
- scikit-learn
- seaborn (or simply matplotlib)
- MNE-Python


## Usage
in the Notebooks folder, you can find some notebooks which will explain how to properly use some modules. These notebooks are also included in the [official documentation](https://selfeeg.readthedocs.io/en/latest/index.html).


## Contribution Guidelines
If you'd like to **contribute** to selfEEG, please take a look at our [contributing guidelines](CONTRIBUTING.md).

If you also have suggestions regarding novel features to add, or simply want some **support** or **suggestions** on how to exploit this library in your SSL experiments, please consider writing to our research team.

[MedMax Team](mailto:manfredo.atzori@unipd.it&cc=federico.delpup@studenti.unipd.it,andrea.zanola@studenti.unipd.it,louisfabrice.tshimanga@unipd.it)

Our team is open to new collaborations!


## Requests and bug tracker
If you have some requests or you have noticed some bugs, use the [GitHub issues](https://github.com/MedMaxLab/selfEEG/issues) page to report them. We will try to solve reported major bugs as fast as possible.


## Authors and Citation
We have worked really hard to develop this library. If you use selfEEG during your research, please cite our work. It will help us to continue doing our research. We are working on a research paper to submit to the Journal of Open Source Software. Until then, you can cite the following [ArXiv preprint](https://arxiv.org/abs/2401.05405):

```bibtex
@misc{delpup2023selfeeg,
      title={SelfEEG: A Python library for Self-Supervised Learning in Electroencephalography},
      author={Federico Del Pup and Andrea Zanola and Louis Fabrice Tshimanga and Paolo Emilio Mazzon and Manfredo Atzori},
      year={2023},
      eprint={2401.05405},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```

Alternatively, you can cite the following [IEEE article](https://ieeexplore.ieee.org/document/10365170):

```bibtex
@article{delpup2023,
  author={Del Pup, Federico and Atzori, Manfredo},
  journal={IEEE Access},
  title={Applications of Self-Supervised Learning to Biomedical Signals: a Survey},
  year={2023},
  volume={11},
  number={},
  pages={144180-144203},
  doi={10.1109/ACCESS.2023.3344531}}
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
