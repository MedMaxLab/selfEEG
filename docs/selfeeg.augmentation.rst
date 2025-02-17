selfeeg.augmentation
*********************

This is the data augmentation module of the selfEEG library.
It is divided in two submodules:

- **Functional**: a collection of data augmentations compatible with numpy arrays
  and torch tensors moved to both CPU or GPU devices.
- **Compose**: a collection of classes designed to combine data augmentations in
  complex patterns.

augmentation.compose module
============================

Classes
---------------
.. currentmodule:: selfeeg.augmentation.compose
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    CircularAug
    DynamicSingleAug
    RandomAug
    SequentialAug
    StaticSingleAug


augmentation.functional module
===============================

Functions
---------------
.. currentmodule:: selfeeg.augmentation.functional
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: functiontemplate.rst

    add_band_noise
    add_eeg_artifact
    add_gaussian_noise
    add_noise_SNR
    change_ref
    channel_dropout
    crop_and_resize
    filter_bandpass
    filter_bandstop
    filter_highpass
    filter_lowpass
    flip_horizontal
    flip_vertical
    get_channel_map_and_networks
    get_filter_coeff
    identity
    masking
    moving_avg
    permutation_signal
    permute_channels
    random_FT_phase
    random_slope_scale
    scaling
    shift_frequency
    shift_horizontal
    shift_vertical
    warp_signal
