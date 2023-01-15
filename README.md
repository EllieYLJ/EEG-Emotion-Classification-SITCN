# EEG-Emotion-Classification-SITCN

This is the core code of the reference: L.J. Yang, Y.X. Wang, X.H. Yang, C. Zheng, Stochastic Weight Averaging Enhanced Temporal Convolution Network for EEG-based Emotion Recognition. Biomedical Signal Processing and Control. 2023.

The code scripts are written in Matlab and Python.

# Requirements

1.Language: Python 3.7, Matlab R2018a

2.Libraries that Python programs depend on: Tensorflow, Numpy, Keras and Scikit-Learn

# Descriptions of the main code files

1./main.py. The main code of SITCN that extracts the features and classification from the EEG signals.

2./tcn.py,./swa,and ./margin_softmax. The directory includes the Code of the TCN SWA and AM-softmax.

3./DE.m. Extracting the DE features as the SITCN model inputs.

4./channelselection.m. Selecting key channels based on DE features and channel weights, and the subsequent averaging and sorting operations are done in Excel.


# Supplement
The functional connection matrix is plotted as a 3D topology by the MATLAB toolbox BrainNet Viewer compiled by Matlab2018a.
