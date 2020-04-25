## Index generation
In a 10-fold cross-validation, samples of a dataset are split into 10 subsets, where 8-folds are used for training, 1 fold is used for validation and the remaining one is used for testing. The folds rotate among each other for each trial of a cross-validation.

This behavior is applied through the index generator by creating 10 subsets of indices following the index assigned to a sound clip in Dataset.hdf5 file generated through the dataset transformer.
