# Version X.X.X (only via git install)

**Functionality**

- **dataloading module**:
    - EEGdataset can preload the entire dataset.
- **models module**:
    - custom layers were moved in a new models.layer submodule
    - layer constraints now include MaxNorm, MinMaxNorm, UnitNorm, with axis selection like in Keras.
    - added Conv1d layer with norm constraint

**Maintenance**

* fixed typos on model module unittest.
* Added new tests for novel functionalities.


# Version 0.1.1 (latest)

This release includes all the revisions made during the Journal of Open Source 
Software (JOSS) peer-review.

**Functionality**

* fixed import problem for python<3.11 due to wrong syntax in the dataloading module.
* fixed small bugs in the dataloading module.

**Maintenance**

* Added more workflows
* Included (and run) a basic set of pre-commit hooks.

**Documentation**

* Added a tutorial with the EEGMMI dataset.


# Version 0.1.0

First release of selfEEG.
