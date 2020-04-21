# MCLNN_AudioClassification
University Project that concerns the design of a Masked Conditional Neural Network for Audio Classification.

## Masked Conditional Neural Network
The MCLNN allows to consider the temporal dimension of data inferring on a window of frames. Each frame is conditioned on n preceeding and n succeeding frames (i.e., the window has width <img src="https://render.githubusercontent.com/render/math?math=d = 2n %2B 1">
). The mask is a binary matrix that enforces a sparsness, so that each hidden unit belonging to the network processes only a limited part of the data.

## Implementation
The code has been written and run in Google Colaboratory. It performs a feature extraction that builds spectrograms from the audio files by segmenting the data, performing the Fast Fourier Transform, re-connecting the segments. \\

A spectrogram is a graphical representation of the intensity of a sound as a function of time. Over the x-axis there is the time, and over the y-axis the frequency (linear or logarithmic scale). Each point of the plane has a color that represents the intensity of the sound in the given instant of time.

## Reference
Fady Medhat, David Chesmore, John Robinson, Masked Conditional Neural Networks for Audio Classification International Conference on Artificial Neural Networks and Machine Learning, ICANN 2017.
