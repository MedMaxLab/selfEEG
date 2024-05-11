selfeeg.ssl
===========

The ssl module collects different self-supervised learning algorithms applied
for the analysis of EEG Data. Each algorithm included an already implemented
fit and test method, to speed up the pretraining process.
In addition, this module includes an implementation of the fine-tuning
function that can also be used as a standalone fit method.

It is divided in two submodules:

- **Base**: a submodule that include the basic objects and functions for both the pretraining and fine-tuning process.
- **Compose**: a collection of contrastive learning algorithms.

ssl.base module
---------------
.. automodapi:: selfeeg.ssl.base
  :no-inheritance-diagram:
  :no-inherited-members:
  :no-main-docstr:
  :noindex:
  :no-heading:

ssl.contrastive module
----------------------
.. automodapi:: selfeeg.ssl.contrastive
  :no-inheritance-diagram:
  :no-inherited-members:
  :no-main-docstr:
  :noindex:
  :no-heading:
