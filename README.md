# MCLNN_AudioClassification
University Project that concerns the design of a Masked Conditional Neural Network for Audio Classification.

## Masked Conditional Neural Network
The MCLNN allows to consider the temporal dimension of data inferring on a window of frames. Each frame is conditioned on n preceeding and n succeeding frames (i.e., the window has width <img src="https://render.githubusercontent.com/render/math?math=d = 2n %2B 1">
). The mask is a binary matrix that enforces a sparsness, so that each hidden unit belonging to the network processes only a limited part of the data.

## Implementation
The code has been written and run in Google Colaboratory. It performs a feature extraction that builds spectrograms from the audio files by segmenting the data, performing the Fast Fourier Transform, re-connecting the segments.

A spectrogram is a graphical representation of the intensity of a sound as a function of time. Over the <img src="https://render.githubusercontent.com/render/math?math=x">-axis there is the time, and over the <img src="https://render.githubusercontent.com/render/math?math=y">-axis the frequency (linear or logarithmic scale). Each point of the plane has a color that represents the intensity of the sound in the given instant of time.

A 10 - Cross Validation is performed. Cumulative results are given according to both the probability vote (each sample is classified according to the most probable class) and the majority vote (each sample is classified according to the most voted class).

The used network is characterized by two masked layers, a dense layer and a softmax output layer, respectively with 220, 200 and 50 units and 35%, 35% and 10% dropout (obviously, the last layer will have as many units as the number of classes and no dropout). All the hidden neurons are provided with the PReLU activation function. The masks' hyperparameters are <img src="https://render.githubusercontent.com/render/math?math=[40, 10]"> for the bandwidth and <img src="https://render.githubusercontent.com/render/math?math=[10, 3]"> for the overlap.

Two musical datasets are used: GTZAN and ISMIR2004. GTZAN audio tracks are collected from a variety of sources including CDs, radio, microphone recordings, in order to represent a variety of recording conditions. It consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks (i.e., it is balanced). ISMIR2004 contains the audio tracks from 8 genres: classical, electronic, jazz and blues, metal, punk, rock, pop, world. For the genre recognition contest, data were grouped into 6 heavily unbalanced classes: classical (640 samples), electronic (229 samples), jazz-blues (52 samples), metal-punk (90 samples), rock-pop (203 samples), world (244 samples), where in some cases two genres were merged into a single class. The total size of the data-set is 1458 tracks. They were characterized by different lengths: therefore, 30 s segments are extracted.

## Reference
Fady Medhat, David Chesmore, John Robinson, Masked Conditional Neural Networks for Audio Classification International Conference on Artificial Neural Networks and Machine Learning, ICANN 2017.
