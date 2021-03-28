# SpLearn: A CS6350 Machine Learning Library

SpLearn is a machine learning library developed by **Spencer Elkington** for
use during the CS6350: Machine Learning course at the University of Utah.

SpLearn is built to run off of the same general interface as the
[scikit-learn](https://scikit-learn.org/stable/) package commmonly used for
statistical analysis and modelling.

## Getting Started

You can run `run.sh` in order to install dependencies and begin a new Jupyter 
Lab session. Python files and notebooks can use `import splearn` to begin using
SpLearn models.

SpLearn is dependent on:
* [NumPy](https://numpy.org/): `pip install numpy`
* [Pandas](https://pandas.pydata.org/): `pip install pandas`

## Documentation

All machine learning models and methods in SpLearn use PyDocs to document
method signatures, along with weak typing provided. In general, training and
test data will accept Pandas DataFrame or Series objects.

The available SpLearn packages are:

* [SpLearn.Metrics](splearn/Metrics.py): A collection of metric methods used for
model testing and additional utility
* [SpLearn.DecisionTree](splearn/DecisionTree): A collection of DecisionTree
models
* [SpLearn.EnsembleLearning](splearn/EnsembleLearning): A collection of Ensemble
learning classifiers that make use of weaker modesl to boost accuracy.
* [SpLearn.Linear](splearn/LinearRegression): A collection of regression
algorithms to generate models with continuous outputs.
* [SpLearn.Perceptron](splearn/Perceptron): A Perceptron learning model used
to create boundaries for linear separability.