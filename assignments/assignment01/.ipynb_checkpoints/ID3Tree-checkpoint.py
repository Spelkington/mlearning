import numpy as np
import pandas as pd

class TreeNode:
    
    def __init__(self):
        self.trained = False
        return
    
    def train(
        self,
        features,
        target,
        gain_func,
        depth_left = -1
    ):
        '''
        Trains this model based on the feature and target data provided.
        
        Parameters:
        
        features (DataFrame): a Pandas DataFrame of features on which the model will be
        trained. These should be categorical in nature.
        
        target (DataFrame/Series): a Pandas DataFrame or Series, one column which
        categorizes the feature data.
        
        gain_func: Any function that is able to take in a features and target
        parameter and output a dictionary where keys are column names and values
        are any sort of information gain.
        
        depth_left: the number of depth layers left before a node cannot create
        any more children. If the depth left is 0, rather than making child nodes,
        the TreeNode will forcefully declare itself as a leaf set to whatever
        target value is the majority within the target parameter.
        '''
        
        uniques, counts = np.unique(target, return_counts = True)
        
        if len(uniques) == 1:
            
            self.label = "leaf"
            self.value = uniques[0]
            
        elif depth_left == 0:
            
            self.label = "leaf"
            self.value = uniques[
                list(counts).index(max(counts))
            ]
            
        else:
            
            gains = gain_func(features, target)
            
            # Find the maximum information gain -
            # doing a loop is pretty shitty but I
            # couldn't remember the quickest way
            # to find the maximum value of a dic-
            # -tionary (:
            curr_max = -1
            max_key = None
            for key, val in gains.items():
                if val > curr_max:
                    curr_max = val
                    max_key = key
                    
            elements = features[max_key].unique()
            
            self.label = max_key
            self.value = {}
            
            for e in elements:
                new_child = TreeNode()
                
                filt = features[max_key] == e
                
                sub_feat = features[filt]
                sub_targ = target  [filt]
                
                sub_feat = sub_feat.drop(max_key, axis=1)
                
                new_child.train(
                    sub_feat,
                    sub_targ,
                    gain_func,
                    depth_left = depth_left - 1
                )
                
                self.value[e] = new_child
            
        self.trained=True
        
        return
    
    def predict(self, features, container):
        
        if self.label == "leaf":
            
            for i in features.index:
                container[i] = self.value
        
        else:
            
            for e, next_node in self.value.items():
            
                subset = features[features[self.label] == e]
                
                next_node.predict(subset, container)
                
        return
            
    
    def dump_nodes(self):
        
        dump = []
        
        if self.trained == False:
            return dump
        
        if self.label == "leaf":
            return dump
        
        for child in self.value.values():
            
            dump.append(child)
            
            for anc in child.dump_nodes():
                dump.append(anc)
            
        return dump

class ID3Tree:
    
    def __init__(self):
        self.trained = False
        pass
    
    def train(
        self,
        features,
        target,
        gain="entropy",
        max_depth=-1
    ):
        
        self.gain_func = None
        if   gain == "entropy":
            gain_func = ID3Tree.gen_gain_func(ID3Tree.entropy)
        elif gain == "majority error":
            gain_func = ID3Tree.gen_gain_func(ID3Tree.maj_err)
        elif gain == "gini":
            gain_func = ID3Tree.gen_gain_func(ID3Tree.gini)
        else:
            print("Invalid gain type - defaulting to entropy")
            gain_func = self.gen_gain_func(self.entropy)
        
        self.root = TreeNode()
        self.root.train(
            features,
            target,
            gain_func,
            depth_left = max_depth
        )
        
        return
    
    def predict(
        self,
        features,
    ):
        
        container = ["" for i in range(len(features))]
        
        # Sanitize indices on the way in
        features = features.reset_index(drop=True)
        
        np.array(
            self.root.predict(features, container)
        )
        
        result = np.array(container)
        
        # For all unpredictable numbers, replace with set mode
        uniques, counts = np.unique(result, return_counts = True)
        
        mode = uniques[
            list(counts).index(max(counts))
        ]
        
        result = np.where(result == "", mode, result)
        
        return result
    
    
    def dump_nodes(self):
        
        dump = self.root.dump_nodes()
        dump.append(self.root)
        
        return dump
    
    def entropy(target):
        '''
        Finds the entropy within a target variable.
        Takes in a 1D Numpy array and returns the
        entropy of that Series
        '''

        # Get the individual classes and counts
        # for every unique element of the target
        classes, counts = np.unique(target, return_counts=True)

        # Calculate the probability of each class
        # by dividing all the counts by the length
        # of the vector
        probs = counts / len(target)

        # Zip the classes and probabilities into a dictionary
        class_probs = dict(zip(classes, probs))

        # Start an accumulation loop for the entropy
        ent = 0
        for c, p in class_probs.items():
            ent += p * np.log2(p)

        # Flip the sign and return
        return -1 * ent
    
    def gen_gain_func(entropy_func):
        '''
        Determine the gain of each column
        in the dataset S. Assumes the last
        column is the target.
        '''
        
        def gains_func(features, target):

            # Find the entropy_func of the target,
            # for calculating information gain later
            targ_ent = entropy_func(target)

            # Start an empty dict to store gains
            # for every column
            col_gains = {}

            # Calculating individual column gains
            for column in features.columns:

                # Find # of unique elements in column
                unique_elements = np.unique(features[column])

                # Start column gain as target entropy_func,
                # since we'll be subtracting subset entropies
                # from this
                col_gain = targ_ent 

                # Cycle over set of all unique elements
                for e in unique_elements:

                    # Create a data subset of only entries where 
                    # the target column == the unique element
                    filt = features[column] == e
                    sub_feat = features[filt]
                    sub_targ = target  [filt]

                    # Determine the probability of getting an
                    # entry in this subset
                    prob = len(sub_feat) / len(features)

                    # Determine the entropy_func of the target
                    # column of the new subset
                    sub_ent = entropy_func(
                        sub_targ
                    )

                    # Subtract this subset's entropy_func from the
                    # full set entropy_func
                    col_gain -= prob * sub_ent

                # Store the column gain in the dictionary
                col_gains[column] = col_gain

            return col_gains
        
        return gains_func
    
    
    
    def maj_err(target):
        elements, counts = np.unique(target, return_counts=True)
        return 1 - (counts.max() / len(target))
    
    def gini(target):
        
        elements, counts = np.unique(target, return_counts=True)
        return 1 - sum([(count / len(target))**2 for count in counts])