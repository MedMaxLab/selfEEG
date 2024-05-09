# Notebooks

This folder gathers all the notebooks created during the development of selfEEG. We decided to include them in the official repository since they might be useful for developers who wish to contribute to the library. Notebooks can be used to implement or test things without applying changes directly to the **selfeeg** or **test** main folders. In particular:

1. All "_guide" notebooks are the ones included in the official documentation, under the tutorials section. They can be modified or run without worrying about changing the documentation page. If you think additional tutorials must be included, try to implement things here before changing notebooks in the doc folder.
2. All notebooks in the test_notebook folder are the first versions of the library unittests. They are not as complete as the final versions of the library unittests, but they can be useful if you want to add further checks on preexisting functionalities or if you want assess if tests on novel functionalities run smoothly before including them in the official test folder.

> [!NOTE]
> Test notebooks will no longer be updated.

4. Augmentation_benchmark is the notebook version of the extra_material augmentation benchmark Python script. You can use it to test how fast a specific augmentation is executed on your device instead of running the whole benchmark.
