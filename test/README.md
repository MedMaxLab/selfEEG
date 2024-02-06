# SelfEEG TEST

This folder contains all the tests you must run to assess the correct functionality of the selfEEG library.

All the tests were implemented with **unittest**, a standard testing framework.

To run the whole test, you just need to go to the main working directory (if you are currently in this folder just type `cd ..` ), then type:

    python -m unittest discover test "*_test.py"

which basically tells python to search for all the unittest tests included in python files inside the test folder and subdirectories whose name ends with "*_test.py*". In addition, you can use the test notebooks included in the Notebooks folder of this repo, which we used as the the starting point for the unittest implementation and can be modified according to your needs, especially if you want to perform additional tests.

If you encounter any problem during the test, or think additional assertions should be added in existing test methods, please raise an issue on GitHub, so that we can work on that.

Also, remember to add a test of any functionality you are working on and check that any other already implemented will not break. To see which basic assertion you must include in a test for a novel functionality, read the following list and take a look at the already implemented code.

To conclude, if you are working on a new data augmentation, please add its benchmarking test in the Augmentation_benchmark.py file in the extra_material folder, just for completeness.


## Basic Assertions

Here is reported a basic list of assertion you must include in your tests based on the selfEEG's module the new functionality will be included. This list is not exhaustive, since some functions will probably need additional checks to understand if they proper work.

### General

All the functions must check that any allowed combination of input arguments will not raise an unexpected error. To speed up the procedure, you can create a dictionary of possible values per arguments and create the input iterator with the `makegrid` method included in each unittest.TestCase class. See already implemented testing pipelines.


### Dataloading

**loading functions**:

- all the files are loaded as excpected
- the output table dimension is the expected one

**split functions**

- the split is done correctly according to the given setting
- a subset label (train/validation/test/excluded) is assigned to all the dataset files
- the output table has the expected dimension


### Augmentation

**Funtional**

- the augmentation will return a tensor with the same size as the input one
- the range of values is the expected one
- no NaN values are returned
- additional test to check that the augmentation is performed correctly

**Composition**

- the composition is done correctly according to the desired order and modality


### Models

- No tensors with abnormal or NaN values are returned
- Layer constraints are respected


### Losses

- The output loss is not NaN
- The returned loss is the expected one (it's best to look at official implementations if possible, good papers usually link a GitHub repo to reproduce the reported results)


### SSL

- The forward method return a tensor with the expected size, without NaNs or strange values
- The fit and test methods will not throw an error for at least two epochs


## Devices where tests went well

1. MacBook Pro 14 inch M2-Pro, MacOS 13.6.2, mps backend
2. Padova Neuroscience Center Server, Ubuntu 18.04.1, Tesla V100 GPU
3. Custom built PC, Windows 10 22H2, NVIDIA RTX 2080 super
