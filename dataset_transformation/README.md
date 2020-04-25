## Dataset transformer
The transformation involves the generation of a single file containing the intermediate representation of a signal, i.e. the spectrogram.

The transformer loads each file in order from the dataset, applied the intermediate representation and stores the resulting transformation to a single hdf5 file for the whole dataset.

The parameters of the Fast Fourier Transform for the generation of the spectrograms are indicated in this module.
