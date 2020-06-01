# Masked Conditional Neural Networks for Audio Classification
Developed by: M. Sodano, S. Frontera (2020).

Supervisors: prof. A. Uncini, prof. D. Comminiello.

Achievement: Neural Networks exam.\
<br>



University Project that concerns the design of a Masked Conditional Neural Network for Audio Classification. The code was written in Google Colaboratory and exported as Python code: imports and related changes may be needed to run it.

## Masked Conditional Neural Network
The MCLNN allows to consider the temporal dimension of data inferring on a window of frames. Each frame is conditioned on n preceeding and n succeeding frames (i.e., the window has width <img src="https://render.githubusercontent.com/render/math?math=d = 2n %2B 1">). 

<p align="center">
<img src="https://camo.githubusercontent.com/e495380dbec9406294b86a72544581ea8a8ae002/68747470733a2f2f6965656578706c6f72652e696565652e6f72672f6d6564696173746f72655f6e65772f494545452f636f6e74656e742f6d656469612f383231313030322f383231353436322f383231353538382f383231353538382d6669672d332d736f757263652d6c617267652e676966" width="350">
</p>

The mask is a binary matrix that enforces a sparsness, so that each hidden unit belonging to the network processes only a limited part of the data. It is characterized by two parameters: the bandwidth (number of consecutive ones in a given column) and overlap (number of overlapping ones in two adjacent columns). The bandwidth indicates, for each neuron, which are the part of data to be processed; the overlap indicates the parts of data that are processed by two consecutive neurons.
<p align="center">
<img src="https://camo.githubusercontent.com/493fa05246198d91d95552b56a58e46b107959b3/68747470733a2f2f6965656578706c6f72652e696565652e6f72672f6d6564696173746f72655f6e65772f494545452f636f6e74656e742f6d656469612f383231313030322f383231353436322f383231353538382f383231353538382d6669672d342d736f757263652d6c617267652e676966" width="600">
</p>

The binary mask is enforced through a redefinition of the weight matrix as W ← W * M (where * indicates the element-wise product), so that only some entries are actually processed.

## Implementation
The code has been written and run in Google Colaboratory. It performs a feature extraction that builds spectrograms from the audio files by segmenting the data, performing the Fast Fourier Transform, re-connecting the segments.

A spectrogram is a graphical representation of the intensity of a sound as a function of time. Over the <img src="https://render.githubusercontent.com/render/math?math=x">-axis there is the time, and over the <img src="https://render.githubusercontent.com/render/math?math=y">-axis the frequency (linear or logarithmic scale). Each point of the plane has a color that represents the intensity of the sound in the given instant of time.

A 10 - Cross Validation is performed. Cumulative results are given according to both the probability vote (each sample is classified according to the most probable class) and the majority vote (each sample is classified according to the most voted class).

The used network is characterized by two masked layers, a dense layer and a softmax output layer, respectively with 220, 200 and 50 units and 35%, 35% and 10% dropout (obviously, the last layer will have as many units as the number of classes and no dropout). All the hidden neurons are provided with the PReLU activation function. The masks' hyperparameters are [40, 10] for the bandwidth and [10, 3] for the overlap.

Two musical datasets are used: [GTZAN](http://marsyas.info/downloads/datasets.html) and [ISMIR2004](http://ismir2004.ismir.net/genre_contest/). GTZAN audio tracks are collected from a variety of sources including CDs, radio, microphone recordings, in order to represent a variety of recording conditions. It consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks (i.e., it is balanced). ISMIR2004 contains the audio tracks from 8 genres: classical, electronic, jazz and blues, metal, punk, rock, pop, world. For the genre recognition contest, data were grouped into 6 heavily unbalanced classes: classical (640 samples), electronic (229 samples), jazz-blues (52 samples), metal-punk (90 samples), rock-pop (203 samples), world (244 samples), where in some cases two genres were merged into a single class. The total size of the data-set is 1458 tracks. They were characterized by different lengths: therefore, 30 s segments are extracted.

## Execution Requirements
Python (version 2.7.11) environment and packages:
- keras 2.2.4
- tensorflow_gpu 1.12.0
- tensorflow 1.14.0
- scipy 1.2.0
- scikit_learn 0.21.2
- scipy 1.0.1
- h5py 2.9.0
- matplotlib 3.0.2
- scikit_learn 0.19.2
- numpy 1.16.1

Moreover, the MCLNN code requires two .hdf5 files, one containing the samples and another of the indices.

*Dataset.hdf5*: a single file containing the intermediate representation (i.e. the spectrograms) of all the files of a dataset. Samples are the transformation of the complete clips.

*Index.hdf5*: these are 3 files: training, testing and validation. Each of the indices files hold the indices of the samples following their location in the Samples.hdf5. These files can be generated as many times as the number of cross-validation operation.

Therefore, first we need to run the Index Generator module, then the Dataset Transformation module, and finally the Main.

## Reference and Credits
Medhat, F., Chesmore, D. & Robinson, J. (2017). Masked Conditional Neural Networks for Audio Classification.. In A. Lintas, S. Rovetta, P. F. M. J. Verschure & A. E. P. Villa (eds.), *ICANN (2)* (p./pp. 349-358), : Springer. ISBN: 978-3-319-68612-7 

All the images belong to the authors.
