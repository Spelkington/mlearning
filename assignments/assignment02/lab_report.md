# The Influence on Random Tree Feature Subsetting on Marketing Success Data

**Authors: Spencer Elkington**

## Abstract

[TODO]


## Introduction

The Random Forest algorithm, first introduced by Tin Kam Ho in 1995, has become an essential tool in the kit of a data scientist to both predict and describe patterns in studied data. The algorithm seeks to use a weakly effective model, the **decision tree**, to create a far stronger predictive model.

Rather than a singular decision tree grown from the data, the Random Forest algorithm trains multiple trees on random subsets of observations and features from the data. When data is given to the model for prediction, each tree will vote on each observation, with the votes tallied and used to predict the outcome of each observation.

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

As the feature subset size becomes **smaller:**

- Each grown tree will **less** information about the overall data, causing them to be **less accurate** in individually predicting both training and test data.
- Each grown tree will have **less** features provided to them, making it **more apparent** which features have more predictive ability in the training data.
- Each grown tree will be **shorter**, making the algorithm's time to train **faster** for the same number of trees, all else held constant.


## Procedures

For this experiment, we will begin by instantiating **four** Random Forest ensemble models. Each one will train trees on a different subset size of features. In this experiment, the three Random Forest models will train its' constituent trees on:

1. 3 out of 16 features (~19%)
2. 5 out of 16 features (~31%)
3. 7 out of 16 features (~44%)
4. 9 out of 16 features (~56%)

Each Random Forest will instruct its' constituent trees to use the **gini index** method for calculating information gain, and to subset **observations** to 62% (derived from $1 - \frac{1}{e}$) of the training subset, with replacement.

Once the trees are instantiated, each will be directed to train constituents. Random Forest is expected to converge exponentially, so observations will be taken exponentially as well.

1. Each model will be trained up to **256** constituent trees; and
2. The accuracy of each model against both **test** and **training** data will be evaluated and recorded at each point where the number of constituents in each model reaches a **power of two.**

At the end of the experiment, each Random Forest model will have its' constituents available for further analysis.


## Results & Discussion

The recorded accuracies at each point are as shown:

[INSERT FIGURE]

This data shows a trend that **small feature subset values failed to converge above the bare minimum accuracy for an effective model** discussed in the Introduction. This is likely due to the limited information provided to each constituent tree limiting the accuracy.

[INSERT FIGURE]

An ensemble of equal-weighted limited-depth trees is reminiscient to the **Adaptive Boosting (AdaBoost)** algorithm for decision trees, with the critical (fatal) difference that this series of trees cannot have their final voting contributions weighted according to their model accuracy, as they would in the AdaBoost algorithm.

It is important to note that this experiment was run with bare-bones cross validation due to resource constrains (This experiment, on the hardware provided at writing, takes approximately 50 minutes to fully train). Ideally, this analysis would be re-run with notable key changes:

- K-fold cross validation would be used to refine the predictions of the models used.
- The experiment would be run multiple times with multiple random seeds used to subset the data, with results averaged out.
- Each model would be provided with **512** or **1028** base constituent models


## Conclusions

In this analysis of banking marketing data, Random Forest feature subset size variation results ranged from XX% to XX% accuracy on test data. Feature subsets below a range of X were shown to converge to accuracies at or below the bare minimum model effectiveness, implying that at least X amount of features are required for a Random Forest model to be an effective predictor on this data.


## Appendix & Notes

***Note:*** *This document functions as the lab report required for WRTG3014, but whose underlying analytical results are partially derived from the Ensemble Learning assignment for CS6430. Each assignment's reports, however, were developed entirely independently and without self-plagiarism.*