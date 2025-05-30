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
    FilterBank
    SeparableConv2d


models.encoders module
========================

Classes
---------

.. currentmodule:: selfeeg.models.encoders
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    DeepConvNetEncoder
    EEGConformerEncoder
    EEGInceptionEncoder
    EEGNetEncoder
    EEGSymEncoder
    FBCNetEncoder
    ResNet1DEncoder
    ShallowNetEncoder
    StagerNetEncoder
    STNetEncoder
    TinySleepNetEncoder


models.zoo module
===================

Classes
---------

.. currentmodule:: selfeeg.models.zoo
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ATCNet
    DeepConvNet
    EEGConformer
    EEGInception
    EEGNet
    EEGSym
    FBCNet
    ResNet1D
    ShallowNet
    StagerNet
    STNet
    TinySleepNet
