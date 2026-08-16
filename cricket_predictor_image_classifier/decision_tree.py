import math
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import sys

# Set higher recursion depth for deep trees
#sys.setrecursionlimit(2000) 

# standard helper functions

def majority(labels):
    if len(labels) != 0: 
        return max(set(labels), key=list(labels).count) 
        #return the most common item in the list
    else:
        return None

def impurity_calculator(labels, criterion): 
    #takes the list of labels and criterion as input and returns impurity value
    n = len(labels)
    if criterion == "entropy":
        if n != 0:  #probs can be found only if set is non-empty
            label_counts = Counter(labels)
            ent = 0.0
            for c in label_counts.values():
                p = c / n
                #adding the entropy of each class
                ent -= p * math.log2(p) if p > 0 else 0 #to avoid log(0)
            return ent
        else:
            return 0.0
    elif criterion == "gini":
        if n != 0:
            label_counts = Counter(labels)
            gini_score = 1
            for c in label_counts.values():
                p = c / n
                #adding the gini impurity of each class
                gini_score -= p ** 2 # 0 probability doesnt affect gini
            return gini_score
        else:
            return 0.0
    else:
        print("Invalid Criterion")
        return

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct_count = (y_pred == y_true).sum() #count of correct predictions
    accuracy_val = correct_count/y_true.size # Percentage of correct predictions
    return accuracy_val 

def count_nodes(node):
    # Count total nodes in a tree/subtree
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    count = 1
    for child in node.children.values(): #recursively count in children
        count += count_nodes(child)
    return count

# end of helper functions

class Node:
    def __init__(self, class_label=None, feature=None, threshold=None, is_leaf=False):
        self.class_label = class_label      # Class label for leaf nodes (majority class)
        self.children = {}           # Child nodes
        self.feature = feature       # feature the node splits on
        self.threshold = threshold   # Threshold value for numeric splits or unique values for categorical
        self.is_leaf = is_leaf       # Whether the node is a leaf node
        
        
        # Add a stats dictionary to cache information for all 3 datasets
        # This is the core of the hint given in piazza
        self.stats = {} #used to store stats for train, val, test at the node level
        self.reset_stats() #set the stats to zero initially

    def reset_stats(self):
        #Helper to reset stats, used in populating.
        self.stats = {
            'train': {'samples': 0, 'correct_if_leaf': 0, 'correct_in_subtree': 0},
            'val':   {'samples': 0, 'correct_if_leaf': 0, 'correct_in_subtree': 0},
            'test':  {'samples': 0, 'correct_if_leaf': 0, 'correct_in_subtree': 0}
        }


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, target_col='result', criterion='entropy'):
        self.max_depth = max_depth #default depth = 10
        self.min_samples_split = min_samples_split # minimum number of samples needed to split a node
        self.root = None
        self.target_col = target_col
        self.criterion = criterion  # 'entropy' or 'gini'
        self.features = None #list of feature names
        self.feature_types = {} #feature name to type mapping
        self.categorical_features = [] # list of categorical feature names
        self.feature_to_idx = {}  # Map feature name to column index
        self.categorical_indices = []  # Indices of categorical features

    def fit(self, df):
        # Fit the decision tree to the training data given in a pd df

        # Extract feature names and target
        self.features = [c for c in df.columns if c != self.target_col]
        
        # Create feature name to index mapping
        self.feature_to_idx = {f: i for i, f in enumerate(self.features)}
        
        # Detect categorical vs numeric per column
        self.feature_types = {}
        self.categorical_features = []
        self.categorical_indices = []
        
        for i, f in enumerate(self.features):
            if pd.api.types.is_numeric_dtype(df[f]):
                self.feature_types[f] = 'numeric'
            else:
                self.feature_types[f] = 'categorical'
                self.categorical_features.append(f)
                self.categorical_indices.append(i)
        
        # Convert to numpy arrays for efficient computation
        X = df[self.features].values
        y = df[self.target_col].values
        
        # Build tree using numpy arrays
        self.root = self.build_tree(X, y, depth=0) # initialize just the root

    def build_tree(self, X, y, depth):
        # recursively build the decision tree using numpy arrays

        # Get majority class for this node
        majority_class = majority(y)
        
        # Stopping conditions -----------
        if len(X) == 0:
            # Return a leaf with a placeholder value, but it's empty
            # We'll rely on parent's majority for prediction if this path is taken
            return Node(class_label=None, is_leaf=True)
        
        if len(np.unique(y)) == 1:
            # All samples are of the same class
            return Node(class_label=np.unique(y)[0], is_leaf=True)
        
        if depth >= self.max_depth:
            # Max depth reached
            return Node(class_label=majority_class, is_leaf=True)
        
        if len(X) < self.min_samples_split:
            # Not enough samples to split
            return Node(class_label=majority_class, is_leaf=True)
        

        # Find best split
        gain, feature_idx, threshold, children, missing_count = self.select_optimal_split(X, y)

        if missing_count  == len(y):
            # All values are missing for the best feature, cannot split
            return Node(class_label=majority_class, is_leaf=True)
        
        if gain <= 0: # if gain is <= 0 
            return Node(class_label=majority_class, is_leaf=True)
        
        if feature_idx is None: # if no feature is found or 
            return Node(class_label=majority_class, is_leaf=True)
        
        # End of stopping conditions -------

        # Create internal node (store feature name, not index)
        feature = self.features[feature_idx]
        new_node = Node(feature=feature, threshold=threshold, class_label=majority_class, is_leaf=False)

        # Build children recursively from the split subsets
        for split_val, (X_new, y_new) in children.items():
            child_node = self.build_tree(X_new, y_new, depth + 1)
            new_node.children[split_val] = child_node
        
        return new_node
   
    def information_gain(self, parent, children):
        """Calculate information gain (or Gini gain) from a split"""
        gain = impurity_calculator(parent, self.criterion)

        for child in children:
            if len(child) > 0:
                child_impurity = impurity_calculator(child, self.criterion)
                gain -= (len(child) / len(parent)) * child_impurity

        return gain

    def split(self, X, y, feature_idx, split_type):
        """
        Splits the dataset based on the feature at feature_idx.
        Split continuous feature at median using numpy arrays.
        Splits categorical features by unique values.
        """
        feature_vals = X[:, feature_idx] 
        results = [0, {}, {}] 
        if split_type == 'numerical':
            results[2] = None #will be set to unique values later
            # Convert to float for numeric operations, handling missing values
            try:
                feature_vals = feature_vals.astype(float)
                # Handle missing values by using nanmedian
                if np.any(np.isnan(feature_vals)):
                    median = np.nanmedian(feature_vals)
                    # Replace NaN with median for splitting
                    feature_vals = np.where(np.isnan(feature_vals), median, feature_vals)
                else:
                    median = np.median(feature_vals)
            except (ValueError, TypeError):
                # If conversion fails, return no gain
                return results
            
            # Avoid splitting if all values are the same
            if len(np.unique(feature_vals)) == 1:
                return results

            # Get indices for the left split (value <= median) and right split (value > median)
            lind = np.where(feature_vals <= median)[0]
            rind = np.where(feature_vals > median)[0]


            # If one side is empty, no gain
            if X[lind].shape[0] == 0 or X[rind].shape[0] == 0:
                # print(f"Empty split: left={X_left.shape[0]}, right={X_right.shape[0]}, median={median}, unique_vals={len(np.unique(feature_vals))}")
                return results


            children_y_data = [y[lind], y[rind]] 
            results[0] = self.information_gain(y, children_y_data)

            results[1] = {
                "lessereq": (X[lind], y[lind]),
                "greater": (X[rind], y[rind])
            }
            results[2] = median
            
        else:
            
            # Handle missing values - convert to string '__MISSING__'
            feature_vals = np.where(pd.isna(feature_vals), '__MISSING__', feature_vals)
            feature_vals = feature_vals.astype(str)
            
            unique_vals = np.unique(feature_vals)
            
            if len(unique_vals) <= 1: #if only one unique value, no split
                return [0, {}, None]
            
            children_y = []
            split_children = {}
            
            for val in unique_vals:
                mask = feature_vals == val
                X_subset, y_subset = X[mask], y[mask]
                children_y.append(y_subset)
                split_children[val] = (X_subset, y_subset)
            
            results[0] = self.information_gain(y, children_y)
            results[1] = split_children
            results[2] = unique_vals

        return results

    def select_optimal_split(self, data_X, target_y):
        """
        Determines the feature and split value that yields the maximum information gain.
        
        data_X: numpy array of features (n_samples, n_features)
        target_y: numpy array of labels
        
        Returns: (gain, feature_index, split_value, child_data_sets, missing_count)
        """
        
        # Initialize a list to hold the best split results: 
        # [gain, feature_index, split_value, child_data_sets, missing_count]
        optimal_split_data = [-1, None, None, None, 0]    
        
        feature_index_counter = 0 # Use an explicit counter instead of range(n_features)
        
        while feature_index_counter < data_X.shape[1]: #iterate through features to find best split
            feature_idx = feature_index_counter
            
            # Determine the split strategy based on feature type
            is_categorical = feature_idx in self.categorical_indices
            # Perform Split and Get Metrics
            # g - gain; t - threshold/unique values; s_c - split children
            if is_categorical:
                g, s_c, t = self.split(data_X, target_y, feature_idx, split_type='categorical')
            else:
                g, s_c, t = self.split(data_X, target_y, feature_idx, split_type='numerical')

            # update optimal split if current is better
            if g > optimal_split_data[0]:
                # Count missing values in the selected feature
                feature_column = data_X[:, feature_idx]
                missing_count = np.sum(pd.isna(feature_column))
                
                # Reassign the entire result tuple in one line
                optimal_split_data = (g, feature_idx, t, s_c, missing_count)
                
            feature_index_counter += 1
        
        return optimal_split_data
    
    # Prediction helpers
    def predict_one(self, x_in, current_node=None):
        """
        Predicts a single sample using a structurally different traversal method.
        x_in: The data sample (Series, dict, or numpy array).
        """
        if current_node is None:
            current_node = self.root
            
        if current_node.is_leaf:
            return current_node.class_label
        
        # Consolidated Feature Value Retrieval
        feature_name = current_node.feature
        feature_val = None

        if isinstance(x_in, (pd.Series, dict)): # Handle pandas Series or dict input
            feature_val = x_in.get(feature_name) if isinstance(x_in, dict) else x_in[feature_name]
        else:
            # Assuming numpy array input, we need the feature index
            feature_idx = self.feature_to_idx.get(feature_name)
            if feature_idx is not None:
                feature_val = x_in[feature_idx]
            else:
                # Fallback if feature index not found (unlikely)
                return current_node.class_label 

        # --- 2. Traversal Logic ---
        next_child_key = None
        
        if feature_name in self.categorical_features:
            # Categorical feature logic
            # Handle NaN/None by assigning a unique key for the split, different from '__MISSING__'
            cat_val = str(feature_val) if pd.notna(feature_val) else 'MISSING_CATEGORY'
            
            if cat_val in current_node.children:
                return self.predict_one(x_in, current_node.children[cat_val])
            else:
                # Unseen category fallback
                return current_node.class_label

        else:
            # Numeric feature logic
            try:
                val_float = float(feature_val)
                
                # Explicitly check for NaN using the math library
                if math.isnan(val_float) or pd.isna(feature_val):
                    # If value is missing/NaN, fall back immediately
                    return current_node.class_label 

                # Check directly against node.threshold for numeric split
                threshold = current_node.threshold
                
                if val_float <= threshold:
                    next_child_key = "lessereq" # Corresponds to the '<= threshold' child
                else:
                    next_child_key = "greater" # Corresponds to the '> threshold' child
                
                # The standard split must exist, otherwise fallback
                if next_child_key in current_node.children:
                    return self.predict_one(x_in, current_node.children[next_child_key])
                
            except (ValueError, TypeError):
                # Invalid or non-numeric value passed for a numeric feature
                return current_node.class_label

        # Final fallback if traversal failed (e.g., node has no matching child key/branch)
        return current_node.class_label

    def predict(self, X):
        """
        Predict for multiple samples.
        X: can be a pandas DataFrame or numpy array
        """
        if isinstance(X, pd.DataFrame):
            # DataFrame - iterate over rows
            preds = []
            for _, row in X.iterrows():
                preds.append(self.predict_one(row))
            return np.array(preds)
        else:
            # Numpy array - iterate over rows
            preds = []
            for i in range(len(X)):
                preds.append(self.predict_one(X[i]))
            return np.array(preds)
    
    def accuracy(self, X, y):
        """
        Calculate accuracy on a dataset.
        X: pandas DataFrame or numpy array
        y: numpy array or pandas Series
        """
        predictions = self.predict(X)
        if isinstance(y, pd.Series):
            y = y.values
        return np.mean(predictions == y)