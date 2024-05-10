# Version X.X.X (only via git install)

**Functionality**

- **overall**:
  - all functions have been aligned to the `lower_case_with_undersocres` format.
  - all classes have been aligned to the `CapitalizedWords` format.
- **dataloading module**:
    - EEGDataset can preload the entire dataset.
    - Fixed bugs
      (get_eeg_partition_number returns float values with some input arguments)
- **models module**:
    - custom layers were moved in a new models.layer submodule
    - layer constraints now include MaxNorm, MinMaxNorm, UnitNorm, with axis
      selection like in Keras.
    - added Conv1d layer with norm constraint and causal padding
- **ssl module**:
    - EarlyStopping now accepts a custom device to use during best weights recording

**Maintenance**

* Documentation notebooks have been fixed to the new naming format
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
