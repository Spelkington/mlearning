import numpy as np
import pandas as pd
from splearn.Metrics import accuracy_score

class AdaBooster:

    def __init__(self, learner_type):
        '''
        Instantiates an AdaBooster for the provided type of weak learner
        
        Parameters:
        
            * features (class): Any type of learning class that that has:

                - A no-argument initializer
                - A method that takes the form class.train(features, target, weights, **kwargs)
                - A method that takes the form class.predict(features, target)
        '''

        # Pre-store all of the necessary instance variables in order
        # to use them for both batch and iterative learning
        self.Learner = learner_type
        self.features: pd.DataFrame = None
        self.target: pd.Series = None
        self.weights: pd.Series = None
        self.learners = None
        self.learner_args = None
        self.votes = None
        self.classes = None

    
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        iterations: int,
        learning_rate = 1,
        learner_args = {},
    ):
        '''
        Trains an AdaBoost-ed version of the learner type provided in the construction of the AdaBooster object.
        
        Parameters:
        
            * features (DataFrame): A DataFrame of features
            * target (Series): A list of the target variables that the model is trained on. There MUST be ONLY
            two classes to predict from, as the AdaBooster is a binary classification booster.
            * iterations (int): The number of AdaBoost iterations to undergo.

        Optional:
            * learner_args (arguments): a list or dictionary of keyword values specific to the underlying weak
            learner type that is being AdaBoosted
        '''

        if len(target.unique()) > 2:
            raise Exception("AdaBooster can only be used on binary outputs.")

        # Store features and target for possible further training
        self.features = features
        self.target = target
        self.classes = target.unique()
        self.learner_args = learner_args

        self.learners = []
        self.votes  = []

        self.weights = pd.Series(np.array([1/len(target) for _ in range(len(target))]), name="weights")
        self.weights.index = target.index

        self.iterate(iterations, learning_rate)


    def iterate(self, iterations, learning_rate = 1):

        for _ in range(iterations):

            # Create a new learner and train it on the training data
            learner = self.Learner()
            learner.train(
                self.features,
                self.target,
                self.weights,
                **self.learner_args
            )
            
            preds = learner.predict(self.features)
            preds.index = self.target.index

            misses = np.invert(preds == self.target).astype(int)
            weighted_error = sum(misses * self.weights) / sum(self.weights)
            total_error    = sum(misses) / len(misses)

            vtwt = np.log( (1 - total_error)    / (total_error))
            stage = np.log( (1 - weighted_error) / (weighted_error) )

            self.learners.append(learner)
            self.votes.append(vtwt)

            for i in self.target.index:
                self.weights[i] = self.weights[i] * np.exp(misses[i] * np.abs(stage) * learning_rate)

            self.weights = self.weights / sum(self.weights)

    def predict(
        self,
        features,
    ):
        
        tally = np.zeros(len(features))

        vals = [0, 1]
        signs = [-1, 1]
        binary_map = dict(zip(vals, self.classes))
        signs_map  = dict(zip(self.classes, signs))

        for i, m in enumerate(self.learners):

            preds = m.predict(features)
            signs = preds.map(signs_map)
            
            preds = signs * self.votes[i]

            tally += preds

        for i, v in enumerate(tally):
            if v <= 0:
                tally[i] = 0
            else:
                tally[i] = 1
        tally = tally.astype(int)

        results = pd.Series(tally)
        results = results.map(binary_map)

        return results

        