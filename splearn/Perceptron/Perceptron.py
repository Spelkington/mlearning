import numpy as np
import pandas as pd
from splearn.Metrics import accuracy_score

class BatchPerceptron:

    def __init__(
        self,
    ):
        self.weights: np.array = None
        self.trained: bool = False
        self.dim: int = 0
        self.labels   = None
        pass

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        epochs: int
    ):

        # Set the dimensionality and weights
        self.dim = len(features.columns)
        self.labels = features.columns
        self.weights = np.zeros(self.dim)

        self.add_batch(
            features,
            target,
            epochs
        )

        self.trained = True

    def add_batch(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        epochs: int,
        careful_step: bool = False
    ):

        if len(features.columns) != self.dim:
            raise Exception(
                f"The provided target dimension ({len(features.columns)}) is "
                f"outside the dimensionality of this Perceptron ({self.dim})."
            )

        # Convert targets to -1 and 1
        targ = np.round((target.to_numpy() - 0.5) * 2)
        data = features.to_numpy()

        for e in range(epochs):
            
            # Preds as -1 and 1
            preds = np.round((self.predict(features) - 0.5) * 2)
            signs  = (targ - preds) / 2

            mods = (data.T * signs)
            adj  = np.sum(mods, axis=1)

            if careful_step:
                adj *= 1 - accuracy_score(targ, preds)

            self.weights += adj

        pass

    def predict(
        self,
        features,
    ):

        if type(features) == pd.DataFrame:
            features = features.to_numpy()

        preds = (features * self.weights).sum(axis=1)
        preds = (preds >= 0).astype(int)

        return preds

class StdPerceptron:

    def __init__(
        self,
    ):
        self.weights: np.array = None
        self.trained: bool = False
        self.dim: int = 0
        self.labels   = None
        pass

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        epochs: int
    ):

        # Set the dimensionality and weights
        self.dim = len(features.columns)
        self.labels = features.columns
        self.weights = np.zeros(self.dim)

        self.add_batch(
            features,
            target,
            epochs
        )

        self.trained = True

    def add_batch(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        epochs: int,
    ):

        if len(features.columns) != self.dim:
            raise Exception(
                f"The provided target dimension ({len(features.columns)}) is "
                f"outside the dimensionality of this Perceptron ({self.dim})."
            )

        # Convert targets to -1 and 1
        targ = target.to_numpy()
        data = features.to_numpy()

        for e in range(epochs):
            
            for r in range(data.shape[0]):

                pred = int(np.dot(data[r], self.weights) <= 0)
                hit  = int(targ[r] == pred)
                hit  = (hit - 0.5) * 2

                if hit != 0:
                    self.weights += data[r] * hit

    def predict(
        self,
        features,
    ):

        if type(features) == pd.DataFrame:
            features = features.to_numpy()

        preds = (features * self.weights).sum(axis=1)
        preds = (preds >= 0).astype(int)

        return preds