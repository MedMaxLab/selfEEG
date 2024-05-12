selfeeg.models
****************

This module collects various Deep Learning models and custom layers.
It is divided in two submodules:

- **layers**: a collection custom layers with the possibility to add norm constraints.
- **zoo**: a collection of deep learning models proposed for EEG applications.

models.layers module
======================

Classes
-----------

.. currentmodule:: selfeeg.models.layers
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ConstrainedConv1d
    ConstrainedConv2d
    ConstrainedDense
    DepthwiseConv2d
    SeparableConv2d


models.zoo module
===================

Classes
---------

.. currentmodule:: selfeeg.models.zoo
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    DeepConvNet
    DeepConvNetEncoder
    EEGInception
    EEGInceptionEncoder
    EEGNet
    EEGNetEncoder
    EEGSym
    EEGSymEncoder
    ResNet1D
    ResNet1DEncoder
    ShallowNet
    ShallowNetEncoder
    StagerNet
    StagerNetEncoder
    STNet
    STNetEncoder
    TinySleepNet
    TinySleepNetEncoder
