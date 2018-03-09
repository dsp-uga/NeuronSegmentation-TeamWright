# Team Wright - P3

### Project Description
Goal of the project is to implement neuron segmentation on [NeuroFinder](http://neurofinder.codeneuro.org/) dataset. Dataset includes 19 training sets(includes images + neuron regions) and 9 testing sets(includes only images).

### Approach
We have tried three different approaches. 
- Starting with a simple [Naive Segmentation](https://github.com/dsp-uga/Wright/wiki/Naive-Segmentation), where an effort to find relation on how the neuron pixels and non-neuron pixels is made.
- [Non-negative Matrix Factorization](https://github.com/dsp-uga/Wright/wiki/Non-negative-Matrix-Factorization) method where NMF is applied on image data and then connected components which satisfy a condition are extracted as neuron segments.
- [Neural Networks](https://github.com/dsp-uga/Wright/wiki/Neural-Networks) in which a binary classification is applied using dense neural networks. Blob detection is performed on the output of network to get desired regions of interest.

### Running the project
For nmf implementation, can be run using command `python3 main.py`. It accepts two command line inputs:
- `"-d", "--dataset" <path to directory that includes neurofinder.**.**.test folders>`
- `"-o", "--output"  <path to directory to which output json files of each dataset are written>`

For nn implmentation, can be run using `python src/nn.py`.

### Requirements
- [Python](https://www.python.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit](http://scikit-learn.org/)
- [Thunder](http://thunder-project.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)

### Authors
See the [contributors](https://github.com/dsp-uga/Wright/blob/master/CONTRIBUTORS.md) for details.

### License
This project is licensed under [MIT License](https://github.com/dsp-uga/Wright/blob/master/LICENSE).

### Acknowledgement
- Thanks to Shannon Quinn for helping in clearing doubts in neural networks.
- Thanks to Nihal Soans and Dharamendra for suggestions on neural networks and nmf.
