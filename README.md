Author: Ian Edwards
Date: 11/06/2021
EEE4114F Course Project: Machine learning
=========================================
# ShapeClassification
A CNN to classify different types of basic shapes.

Instructions:
1) Install requirements and environment with:
"make install"
2) download dataset
3) move dataset to current folder
4) Run with "make classify"

Note:
The UI allows one to train a new model, see statistics for a loaded model or evaluate a model by passing in paths to images to be classified.
In this example, the dataset used the is the one shown bellow:
Possible dataset: https://drive.google.com/file/d/1w5XiUtJXmYU9gC7AVAzZYR42Adh9YSUI/view?usp=sharing .
With the location of the dataset being "./greayscale'

The model loaded can be changed in shapeClassifier.py with saveModelName variable.
When running shapeClassifier.py directly, one can input arguments such as batchsize, epochs and learning rate.
Note: The model created is only as good as the data given to it.


