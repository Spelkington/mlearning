import pandas as pd
import numpy as np

# def new_function(
#         arg1: type,
#         arg2: type
# ):
#     '''
#     Lorem ipsum
#     
#     Parameters:
#     
#         * param (type): Lorem ipsum
#     
#     Output:
#
#         * output (type): Lorem ipsum
#     '''
#
#     pass

def accuracy(
        targs: np.array,
        preds: np.array,
        wghts: np.array = None,
):
    '''
    Determines the accuracy of a given set of predictions
    
    Parameters:
    
        * targs (array-like): A list of the real target outcomes
        * preds (array-like): A list of the predictions made
        
        Optional:
        
        * wghts (array-like): A list of weights used to calculate a weighted
        error
    
    Output:

        * (float): A number in [0, 1] representing the accuracy of the predictions
        against the model
        
    '''

    hits = np.equals(targs, preds)