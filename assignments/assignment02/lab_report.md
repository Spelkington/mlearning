# The Influence on Random Tree Feature Subsetting on Marketing Success

**Authors: Spencer Elkington**

## Abstract

[TODO]

## Introduction

The Random Forest algorithm, first introduced by Tin Kam Ho in 1995, has become an essential tool in the kit of a data scientist to both predict and describe patterns in studied data. The algorithm seeks to use a weakly effective model, the **decision tree**, to create a far stronger predictive model. Rather than a singular decision tree grown from the data, the algorithm trains a number of trees on random subsets of observations and features from the data. When data is given to the model for prediction, each tree will vote on each observation, with the votes tallied and used to predict the outcome of each observation.

This allows two major insights into the data:

1. The ensemble of trees allows a far more flexible model that is able to more accurately predict nuances in the data; and
2. The random subsetting of features allows us to look closer at which features in the data may have the most predictive ability and should be looked into closer by other methods.

In this experiment, we are looking closely at how the number of features randomly selected for each tree influences the ability of the model to predict both **training** and **testing** data.

The data being used is a set of testing data from a bank regarding a marketing campaign. The bank would like to predict whether a given call will yield further business, provided details about the customer such as their demographic status and prior relationship to the bank.

The bare minimum expectation for a predictive model is that it is **better than simply predicting the most common outcome in the dataset**. In this case, where the data has vastly more negative ("no further yield") outcomes than positive ("has further yield"), the most effective bare-minimum model would simply predict that no observations will have further yield for the bank. On the training and testing datasets, this would result in the following model accuracies:

- **Training:** 0.8808 (88.08%) accuracy
- **Testing:**  0.8752 (87.52%) accuracy

In order for a model to be better than the bare minimum, **it must do better than these predictive accuracies.**

An analysis into the effects of feature subset size on predictibility is important because of an inherent trade-off between factors of time required for training, descriptiveness of the model on the data, and final accuracy of the model.

## Procedures

For this experiment, we will begin by instantiating **three** Random Forest ensemble models. Each one will train trees on a different subset size of features. In this case:

- **Random Forest 1:** 3 out of 16 features
- **Random Forest 2:** 5 out of 16 features
- **Random Forest 3:** 7 out of 16 features




## Results & Discussion


## Conclusions


## Appendices

*Note: This document functions as the lab report required for WRTG3014, but whose content is derived from the Ensemble Learning assignment for CS6430*