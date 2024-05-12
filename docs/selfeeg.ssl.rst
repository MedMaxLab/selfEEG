selfeeg.ssl
**************

The ssl module collects different self-supervised learning algorithms applied
for the analysis of EEG Data. Each algorithm included an already implemented
fit and test method, to speed up the pretraining process. In addition, this
module includes an implementation of the fine-tuning function that can also be
used as a standalone fit method.

It is divided in two submodules:

- **Base**: a submodule that include the basic objects and functions for both the pretraining and fine-tuning process.
- **Compose**: a collection of contrastive learning algorithms.

ssl.base module
===================

Classes
---------------

.. currentmodule:: selfeeg.ssl.base
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    EarlyStopping
    SSLBase

Functions
---------------

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: functiontemplate.rst

    evaluate_loss
    fine_tune

ssl.contrastive module
=========================

Classes
---------------

.. currentmodule:: selfeeg.ssl.contrastive
.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    BarlowTwins
    BYOL
    MoCo
    SimCLR
    SimSiam
    VICReg
