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

def accuracy_score(
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

    if wghts is None:
        wghts = np.zeros(len(targs)) + 1

    hits: np.array = (np.array(targs) == np.array(preds))
    hits = hits.astype(int)

    accuracy = np.dot(hits, wghts) / sum(wghts)

    return accuracy

def wt_unique(elements, weights):
    '''
    Calculates the unique values in an array, along with their weighted
    count.
    
    Parameters:
    
        * target (ndarray): a vector of output classes
        * weights (ndarray): a vector of sample weights
        
    Output:
    
        * (float): a float between 0 and 1 representing the order of the
            output vector
    '''

    uniques = np.unique(elements)
    e_weights = np.zeros(len(uniques))

    for i, e in enumerate(uniques):
        filter = (elements == e).astype(int)
        filt_wts = weights * filter
        e_weights[i] = sum(filt_wts)

    return uniques, e_weights
    

def maj_err(target, weights):
    '''
    Calculates the majority error within a given output data column
    
    Parameters:
    
        * target (ndarray): a vector of output classes
        
    Output:
    
        * (float): a float between 0 and 1 representing the order of the
            output vector
        
    '''
    
    e_weights = wt_unique(target, weights)[1]
    return 1 - (e_weights.max() / sum(weights))


def gini(target, weights):
    '''
    Calculates the gini error within a given output data column
    
    Parameters:
    
        * target (ndarray): a vector of output classes
        
    Output:
    
        * (float): a float between 0 and 1 representing the order of the
            output vector
        
    '''
    e_weights = wt_unique(target, weights)[1]
    wt_sum = sum(e_weights)
    return 1 - sum( [(wt / wt_sum)**2 for wt in e_weights] )

def entropy(target, weights):
    '''
    Calculates the entropy within a given output data column
    
    Parameters:
    
        * target (ndarray): a vector of output classes
        
    Output:
    
        * (float): a float between 0 and 1 representing the order of the
            output vector
        
    '''

    # Get the individual classes and counts
    # for every unique element of the target
    classes, e_weights = wt_unique(target, weights)

    # Calculate the probability of each class
    # by dividing all the counts by the length
    # of the vector
    probs = e_weights / sum(weights)

    # Zip the classes and probabilities into a dictionary
    class_probs = dict(zip(classes, probs))

    # Start an accumulation loop for the entropy
    ent = 0
    for p in class_probs.values():
        ent += p * np.log2(p)

    # Flip the sign and return
    return -1 * ent