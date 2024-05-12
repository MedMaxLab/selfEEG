selfeeg.utils
**************

This module simply gathers functions and classes for various purposes. For example, you can find a torch implementation of the Scipy's pchip interpolation function (used for resampling) or pytorch EEG scaler with a soft clipping option. Both the cited functions are compatible with GPU tensors.

Classes
---------------

.. currentmodule:: selfeeg.utils.utils
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    RangeScaler

Functions
--------------

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: functiontemplate.rst

    check_models
    count_parameters
    create_dataset
    get_subarray_closest_sum
    scale_range_soft_clip
    torch_pchip
