# Extra Materials

This folder contains additional files that integrate information presented in the documentation. In particular:

1. **Augmentation_benchmark.py**: A file that run a benchmark test for the augmentation module. The file run 10 calls per function with multiple configurations (tensor/numpy array, GPU/CPU, batch_equal True/False), leaving NaN for those configurations not executable. The total execution time per configuration will be stored in the bench_table.csv file. The file also accepts an optional arguments to set the number of repetition of the 10 calls per configuration (total calls will be 10*repetition). Higher the number repetitions, higher the total execution time. To run the code, simply use the following command, changing X with an integer for the number of repetitions (use -h instead of -r X for additional help)

    ``python Augmentation_benchmark.py -r X``

2. **bench_table.csv**: A table reporting the results of our augmentation benchmark performed on the Padova Neuroscience Server with a Tesla V100 and 100 repetitions. Remember that higher repetitions means higher reported times, since table values referes to the total time. Higher repetitions also means better assessment of the performance differences
3. **geogebra-export.gbb**: A geogebra file used to design the custom range scaler with soft clipping. You can import it in the online version of geogebra and see how values are mapped from the input "uV" range to the desired output.
