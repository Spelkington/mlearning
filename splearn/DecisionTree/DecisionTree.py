import numpy as np
import pandas as pd
import splearn.Metrics as met
import multiprocessing as mp

class TreeNode:
    
    def __init__(self):
        self.trained = False
        self.label = None
        self.split = None
        self.value = None
        self.children = {}
        
        return
    
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        weights: pd.Series,
        gain_func,
        depth_left = -1,
        pool = None
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
        
        # If the number of unique classes in the target is one, we've hit a leaf
        # node. The value of this node will be that one unique class.
        if len(uniques) == 1:
            
            self.value = uniques[0]
            
        # If the depth left is zero, instead of making children, just make this
        # a leaf node and set the value to whatever the majority element in
        # the targets is.
        elif depth_left == 1 or len(features.columns) == 0:

            e_weights = np.zeros(len(uniques))

            for i, e in enumerate(uniques):
                filt = (target == e).astype(int)
                filt_wts = weights * filt
                e_weights[i] = np.sum(filt_wts)

            self.value = uniques[e_weights.argmax()]
            
        else:
            
            # Calculate the gains of each feature
            gains = gain_func(features, target, weights)

            
            # Find the maximum gain - doing a loop is pretty shitty but I
            # couldn't remember the quickest way to find the maximum value of
            # a dictionary (:
            keys = np.array(list(gains.keys()))
            vals = np.fromiter(gains.values(), dtype="float64")
            max_val = vals.max()
            max_key = keys[vals.argmax()]

            if max_val == 0:

                e_weights = np.zeros(len(uniques))

                for i, e in enumerate(uniques):
                    filt = (target == e).astype(int)
                    filt_wts = weights * filt
                    e_weights[i] = np.sum(filt_wts)

                self.value = uniques[e_weights.argmax()]

            else:

                # Set the label of this node to the max gain key
                self.label = max_key
                
                # In case an observation ever reaches this node but does not have
                # a further trained node to traverse to, set the value to the
                # majority node of this subset.
                self.value = uniques[
                    list(counts).index(max(counts))
                ]
                
                # Create node children
                if features[self.label].dtype == "int64":
                    
                    median = np.median(features[max_key])
                    self.split = median
                    
                    filt = features[max_key] > self.split
                    
                    # Calculate feature subsets
                    top_feat = features[filt]
                    bot_feat = features[np.invert(filt)]
                    
                    top_wght = weights[filt]
                    bot_wght = weights[np.invert(filt)]
                    
                    if len(top_feat) > 0:
                        
                        top_feat = top_feat.drop(max_key, axis=1)
                        top_targ = target[filt]
                        
                        top_child = TreeNode()
                        top_child.train(
                                top_feat,
                                top_targ,
                                top_wght,
                                gain_func,
                                depth_left = depth_left - 1
                        )
                        self.children[1] = top_child
                
                    if len(bot_feat) > 0:
                        
                        bot_feat = bot_feat.drop(max_key, axis=1)
                        bot_targ = target[np.invert(filt)]
                        
                        bot_child = TreeNode()
                        bot_child.train(
                                bot_feat,
                                bot_targ,
                                bot_wght,
                                gain_func,
                                depth_left = depth_left - 1
                        )
                        self.children[0] = bot_child
                    
                    
                # Split children into nodes by unique elements
                else:
                
                    # Get the unique elements of the maximum gain column
                    elements = features[max_key].unique()

                    # Create a new child for every element within the
                    # gain-iest feature
                    for e in elements:
                        new_child = TreeNode()

                        # Filter the subset by this element in the column
                        filt = features[max_key] == e

                        # Subset the features and target off of the filter,
                        # and drop the maximum gain column
                        sub_feat = features[filt]
                        sub_feat = sub_feat.drop(max_key, axis=1)
                        
                        sub_targ = target [filt]
                        sub_wght = weights[filt]

                        # Train the new child with the subset of targets and
                        # features, the provided gain function, and decrement the
                        # depth by 1.
                        new_child.train(
                            sub_feat,
                            sub_targ,
                            sub_wght,
                            gain_func,
                            depth_left = depth_left - 1
                        )

                        self.children[e] = new_child
            
        self.trained=True
        
        return
    
    def predict(self, features, container):
        
        # If the node doesn't have a label, it's a leaf node. Set all the
        # container items at the features' indices to the node value
        if self.label == None:
            for i in features.index:
                container[i] = self.value
        
        else:
            
            if features[self.label].dtype == "int64":
            
                filt = features[self.label] > self.split
                
                top_feat = features[filt]
                bot_feat = features[np.invert(filt)]
                
                set_list = [bot_feat, top_feat]
                
                for i, feats in enumerate(set_list):
                    if i in self.children.keys():
                        self.children[i].predict(feats, container)
                    else:
                        for i in feats.index:
                            container[i] = self.value
                
            else:
                
                # Keep track of the unique elements in the label column
                uniques = list(features[self.label].unique())

                # Predict all subsets for which this node has children
                for e, next_node in self.children.items():
                    subset = features[features[self.label] == e]
                    next_node.predict(subset, container)
                    if e in uniques:
                        uniques.remove(e)

                # If the feature element was not seen in the training,
                # set all of their predictions to the value of the node.
                for m in uniques:
                    subset = features[features[self.label] == m]
                    for i in subset.index:
                        container[i] = self.value
                
        return
            
    
    def dump_nodes(self):
        
        dump = []
        
        # Append the child to the dump array, and then recursively append
        # its' ancestors to the dump as well
        for child in self.children.values():
            dump.append(child)
            for anc in child.dump_nodes():
                dump.append(anc)
            
        return dump
    

class DecisionTree:
    '''
    Represents a decision tree that learns from classification and numerical
    data to create a tree classifier.
    '''
    
    def __init__(self):
        '''
        Instantiates a DecisionTree object
        '''
        
        self.trained = False
        pass
    
    def train(
        self,
        features,
        target,
        weights = None,
        gain="entropy",
        max_depth=-1
    ):
        '''
        Trains this model based on the feature and target data provided.
        
        Parameters:
        
        features (DataFrame): a Pandas DataFrame of features on which the model
        will be trained. These should be categorical in nature.
        
        target (DataFrame/Series): a Pandas DataFrame or Series, one column which
        categorizes the feature data.
        
        gain (string): 
            * "entropy"
            * "majority error"
            * "gini"
        
        max_depth (int): The maximum depth of the tree training
        '''
        
        if weights is None:
            weights = pd.Series(np.zeros(len(features)) + 1, name = "weights")
            weights.index = target.index
        
        self.gain_func = None
        if   gain == "entropy":
            gain_func = self.gen_gain_func(met.entropy)
        elif gain == "majority error":
            gain_func = self.gen_gain_func(met.maj_err)
        elif gain == "gini":
            gain_func = self.gen_gain_func(met.gini)
        else:
            print("Invalid gain type - defaulting to entropy")
            gain_func = self.gen_gain_func(met.entropy)
        
        self.root = TreeNode()
        self.root.train(
            features,
            target,
            weights,
            gain_func,
            depth_left = max_depth
        )
        
        return
    
    def predict(
        self,
        features,
    ):
        '''
        Predicts the outputs of a DataFrame of features based on the trained
        underlying DecisionTree
        
        Parameters:
        
        features (DataFrame): a Pandas DataFrame of features on which the model
        will be trained. These should be categorical in nature.
        '''
        
        # Create container to place predictions into
        container = [None for i in range(len(features))]
        
        # Sanitize indices on the way in
        features = features.reset_index(drop=True)
        
        # Predict the results of the given features, and place them into the
        # container, and numpy-ify it
        self.root.predict(features, container)
        result = pd.Series(container)
        
        return result
    
    
    def dump_nodes(self):
        
        dump = self.root.dump_nodes()
        dump.append(self.root)
        
        return dump
    
    
    def gen_gain_func(self, entropy_func):
        '''
        Determine the gain of each column in the dataset S. Assumes the last
        column is the target.
        
        Parameters:
        
            * entropy_func (func(vector): float): any function that inputs a
              vector of data and outputs a floating point value between zero and
              one representing how organized that data is.
        
        Return:
        
            * output (func(DataFrame, Series): dict): a function that takes in a
              DataFrame of features and Series of targets, and outputs a dic-
              tionary with the potential informationgain of each feature column.
              
        '''
        
        def gains_func(
            features: pd.DataFrame,
            target: pd.Series,
            weights: pd.Series
        ):

            # Find the entropy_func of the target,
            # for calculating information gain later
            targ_ent = entropy_func(target, weights)
            sum_weights = np.sum(weights)

            # Start an empty dict to store gains
            # for every column
            col_gains = {}

            # Calculating individual column gains
            for column in features.columns:

                # Start column gain as target entropy_func,
                # since we'll be subtracting subset entropies
                # from this
                col_gain = targ_ent
                
                if features[column].dtype == "int64":
                    
                    median = np.median(features[column])
                    
                    filt = features[column] > median
                    
                    top_targ = target[filt].to_numpy()
                    bot_targ = target[np.invert(filt)].to_numpy()
                    
                    top_wght = weights[filt].to_numpy()
                    bot_wght = weights[np.invert(filt)].to_numpy()
                    
                    top_prob = sum(top_wght) / sum_weights
                    top_entr = entropy_func(top_targ, top_wght)
                    top_entr *= top_prob
                    
                    bot_prob = sum(bot_wght) / sum_weights
                    bot_entr = entropy_func(bot_targ, bot_wght)
                    bot_entr *= bot_prob
                    
                    col_gain -= (bot_entr + top_entr)
                
                else:

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
                        sub_targ = target  [filt].to_numpy()
                        sub_wght = weights [filt].to_numpy()

                        # Determine the probability of getting an
                        # entry in this subset
                        prob = sum(sub_wght) / sum_weights

                        # Determine the entropy_func of the target
                        # column of the new subset
                        sub_ent = entropy_func(
                            sub_targ,
                            sub_wght
                        )

                        sub_ent *= prob

                        # Subtract this subset's entropy_func from the
                        # full set entropy_func
                        col_gain -= sub_ent

                # Store the column gain in the dictionary
                col_gains[column] = col_gain

            return col_gains
        
        return gains_func