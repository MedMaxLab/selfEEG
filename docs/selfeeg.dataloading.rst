selfeeg.dataloading
*********************

The dataloading module offers functionalities for data splitting with various
desired settings, and for efficiently build PyTorch Dataloaders.

dataloading.load module
========================

Classes
---------------
.. currentmodule:: selfeeg.dataloading.load
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    EEGDataset
    EEGSampler

Functions
---------------
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: functiontemplate.rst

    get_eeg_partition_number
    get_eeg_split_table
    get_eeg_split_table_kfold
    check_split
    get_split
