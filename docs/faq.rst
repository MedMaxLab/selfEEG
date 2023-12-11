FAQs
====

General
-------

1) **Does selfEEG support training on GPUs for MacOS devices?**

The library is built on top of PyTorch, which support training on GPUs for macOS devices through the mps backend
(`mps backend <https://pytorch.org/docs/stable/notes/mps.html>`_). Note that only macOS devices with Apple Silicon M series SoC are supported, so older Intel models are excluded. In addition, it is worth to note that the mps backend still not cover all the functionalities implemented in CUDA (a coverage matrix can be found here: `matrix  <https://qqaatw.dev/pytorch-mps-ops-coverage/>`_), and some already implemented are yet to be optimized. This applies only for few things, so you will probably not notice these limitations.



Dataloading module
------------------

1) **I have a dataset which stores EEG data as 3D arrays, with the first dimension being associated to the trial number. Does the dataloading module support data provided in this way?**

Yes, the dataloading module can handle 3d Arrays (the DEAP and SHU datasets have 3d array for example) both for the calculation of the total number of samples and for the sample extraction. Just be sure to not change the loading and transform function.


2) **I have a single dataset with EEG data acquired from a certain number of subjects within multiple sessions? How can I split the data so to be sure that EEGs from a specific session are placed only in a single (train/validation/test) subset?**

You can split data at the session level with the ``GetEEGSplitTable`` function. You just need to:

- Give to the ``dataset_id_extractor`` a function to get the subject ID
- Give to the ``subject_id_extractor`` a function to get the session ID
- set ``split mode`` to 1. 

The point is that this function support splits at two granularity levels, with the second being able to identify unique IDs only when coupled with the first level. In this case, it is reasonable to assume that different subjects can have the same session ID, but there not exist duplicate (subject, session) ID pairs.

The names `subjects` and `dataset ID` were decided only for convention. However, these are just names and the function will not check if the IDs extracted from the file name really refer to the specific dataset or subject. You can give anything you want as long as the previous reasoning about the identification of unique pairs is satisfied.

3) **Can I implement a Leave One Subject Out cross-validation?**

Of course. You just need to call the ``GetEEGSplitTableKfold`` function, setting validation split to subject mode and setting the number of folds equals to the number of subjects. Remember to add a subject_id extractor if needed and, if you have enough data to create a separate test set, to also set the test split mode to subject and adjust the number of folds according to the number of subjects minus ones put in the test set  



Augmentation module
-------------------

1) **Should I set the batch_equal argument to True or False?**

Setting ``batch_equal`` to True has a dual effect. On the one hand, it increases mini-batch heterogeneity, potentially improving the quality of the representations; on the other hand, it slows down model training because broadcast cannot be exploited in its full power. It's up to you to decide which aspect to give priority, depending on your experimental design

2) **Is an augmentation always faster on GPU devices?**

Most of the time, augmentations executed on GPUs are faster compared to one on CPUs. However, it is worth to note that three main factors can affect the computational time of augmentations: the GPU device (cuda or mps), the ``batch_equal`` argument, and the object type (numpy array or tensor).

If you want to check how augmentations perform on different configurations, see the following table, which reported a benchmark test run on the Padova Neuroscience Center Server (GPU Tesla V100) with a 3D array of size (64*61*512). Alternatively, you can run the benchmarking test and check how augmentations specifically perform on your device. 

.. csv-table:: Augmentation Benchmark
    :file: _static/bench_table.csv
    :header-rows: 1
    :widths: 15, 14, 14, 14, 14, 14, 14
    :class: longtable



SSL module
----------

1) **The SSL module implements only Contrastive Learning algorithms, is it possible to use selfEEG to create predictive or generative pretraining task?**

It depends on the type of pretraining task you want to define. However, by defining the right dataloader and loss, and give them to the fine-tuning tuning function of the ssl module, it is possible to construct simple predictive or generative pretraining task. For example, a simple strategy can be:

1. Define an EEGDataset class without the label extraction.
2. Define a custom Augmenter.
3. Define a custom ``collite_fn`` function to give to the PyTorch Dataloader.
4. Define the loss function and other training parameters.
5. Run the fine-tuning function with the Dataloader, loss, and other training parameters.

The important step here is the definition of the ``collite_fn`` function (see `here <https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278?u=ptrblck>`_ on how to create custom collite_fn functions), which is used to create the pretraining target. For example:

1. Reconstructive pretraining (generative): create an augmented batch with the augmenter, then return the augmented batch as the input, and the original batch as the target.
2. Predict if the sample was augmented (predictive): apply an augmentation to a random number of samples before constructing the batch, then return the constructed batch and a binary label (1: augmented sample, 0: original sample)  
3. Predict the type of augmentation applied (predictive): Similar to point 2, with a multiclass label.



















