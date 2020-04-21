# MCLNN_AudioClassification
University Project that concerns the design of a Masked Conditional Neural Network for Audio Classification.

## Masked Conditional Neural Network
The MCLNN allows to consider the temporal dimension of data inferring on a window of frames. Each frame is conditioned on n preceeding and n succeeding frames (i.e., the window has width <img src="https://render.githubusercontent.com/render/math?math=d = 2n %2B 1">
). The mask enforces a systematic sparsness that follows a filterbank-like pattern and it automates the mixing-and-matching between different feature combinations at the input, analgous to the manual hand-crafting of features.

## Reference
Fady Medhat, David Chesmore, John Robinson, Masked Conditional Neural Networks for Audio Classification International Conference on Artificial Neural Networks and Machine Learning, ICANN 2017.
