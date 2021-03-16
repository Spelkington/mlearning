import numpy as np
import pandas as pd
from splearn.DecisionTree.DecisionTree import DecisionTree

class RandomForest:

    def __init__(self):
        '''
        Creates a new TreeBagger that can be trained using an array of
        DecisionTrees
        '''

        self.trees = []
        self.features = None
        self.target = None

    def train(
            self,
            features: pd.DataFrame,
            target: pd.Series,
            num_trees: int,
            seed: int,
            feature_frac: float = 0.68,
            subset_frac: float = 0.68,
            gain = "entropy",
    ):
        '''
        Trains N decision trees for bagging prediction
        
        Parameters:
        
            * features (DataFrame): Features used for training
            * target (Series): Predictions to be trained against
            * num_trees (int): Number of trees used in the prediction
            * gain: Type of gain used to train trees
            * seed (int): Lorem ipsum
        
        '''

        # Stage the internal variables for the machine
        self.features = features
        self.target = target
        self.gain = gain
        self.seed = seed
        self.subset_frac = subset_frac
        self.feature_frac = feature_frac

        classes = [0, 1]
        uniques = np.unique(target)
        self.binary_dict = dict(zip(classes, uniques))
        self.class_dict  = dict(zip(uniques, classes))

        self.iterate(num_trees)


    def predict(
        self,
        features: pd.DataFrame
    ):

        votes = np.zeros(len(features))
        
        for m in self.trees:

            preds = m.predict(features)
            preds = preds.map(self.class_dict)
            preds = (preds - 0.5) * 2
            
            votes += preds

        result = pd.Series((votes >= 0).astype(int))
        result = result.map(self.binary_dict)

        return result


    def __len__(self):
        return len(self.trees)

    def iterate(
        self,
        iterations
    ):
        
        for i in range(iterations):

            # Create a random subset of 68% of the data
            sub_feat = self.features.sample(
                frac = self.subset_frac,
                random_state = self.seed + len(self),
                replace = True,
                axis = "index"
            )

            sub_feat = sub_feat.sample(
                frac = self.feature_frac,
                random_state = self.seed + len(self),
                axis="columns"
            )

            sub_idx = sub_feat.index
            sub_targ = self.target[sub_idx]

            # Predict tree fully based on subset
            tree = DecisionTree()
            tree.train(
                sub_feat,
                sub_targ,
                gain=self.gain,
            )

            # Add tree to bagger
            self.trees.append(tree)