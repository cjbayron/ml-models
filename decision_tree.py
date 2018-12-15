'''
Building a Decision Tree using CART (from scratch)

Note: Code was tested only on dataset with numerical features.
Categorical features are not yet fully supported.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_confusion_matrix

import common.utils as ut

# get data from:
#   https://www.kaggle.com/c/otto-group-product-classification-challenge
TRN_DATA_PATH = 'datasets/otto-group-product-classification/train.csv'
NUM_SAMPLES = 5000
NUM_FEATS = 93

def visualize_data(feats, true_labels, preds):
    '''Display labeled data and clustered data
    '''
    print("Visualizing data...")
    red_feats = ut.reduce_to_2D_by_tsne(feats)

    label2col_map = ['red', 'orange', 'yellow', 'green', 'blue',
                     'violet', 'brown', 'gray', 'pink']

    label_list = np.unique(true_labels)

    _, ax = plt.subplots(ncols=2, figsize=(10, 5))
    graph_label_pair = zip(ax, [true_labels, preds])
    for graph, labels in graph_label_pair:
        for label in label_list:
            # get samples with label == label
            idxs = np.where(labels == label)
            # get components
            pc1, pc2 = red_feats['pc1'].values[idxs], red_feats['pc2'].values[idxs]
            # scatter plot w/ color based on labels
            graph.scatter(x=pc1, y=pc2, color=label2col_map[label-1],
                          alpha=0.5, label=label)

            graph.set_xlabel('PC1')
            graph.set_ylabel('PC2')

    ax[0].set_title('Labeled Products')
    ax[1].set_title('Predicted Labels')

    for graph in ax:
        graph.legend() # show legend
        graph.grid(True) # show gridlines

    plt.show()

def get_impurity(labels):
    '''Calculate Gini impurity
    '''
    num_labels = float(len(labels))
    imp = 0.0
    _, cnts = np.unique(labels, return_counts=True)
    for cnt in cnts:
        cnt = float(cnt)
        imp += float((cnt/num_labels)*(1-(cnt/num_labels)))

    return imp

def get_best_split_along_column(data, labels, feat_idx, categorical=False):
    '''Get best split using features in a single column
    '''
    feat_col = data[:, feat_idx]
    splitter_pool = np.unique(feat_col) # get splitters
    min_im = np.inf
    left_idxs = []
    right_idxs = []
    splitter = None
    for val in splitter_pool:
        if categorical:
            left_labels = labels[feat_col == val]
            right_labels = labels[feat_col != val]
        else:
            left_labels = labels[feat_col >= val]
            right_labels = labels[feat_col < val]

        # if all data is placed on only one side
        # then it is not a meaningful split so we skip
        if len(left_labels) == len(data) or len(right_labels) == len(data):
            continue

        avg_im = len(left_labels) * get_impurity(left_labels) + \
                 len(right_labels) * get_impurity(right_labels)

        if avg_im < min_im:
            min_im = avg_im
            left_idxs = (feat_col >= val)
            right_idxs = (feat_col < val)
            splitter = val

    if len(left_idxs) + len(right_idxs) > 0:
        min_im /= (len(left_idxs) + len(right_idxs))

    return min_im, splitter, left_idxs, right_idxs

class TreeNode():
    '''Node for a Decision Tree
    '''
    def __init__(self):
        self.labels = None
        self.left_node = None
        self.right_node = None
        self.is_leaf = False
        self.categorical = False
        self.splitter = None

    def build_tree(self, feats, labels):
        '''Build tree recursively
        '''
        self.labels = labels
        best_gain = 0
        best_left_idxs = []
        best_right_idxs = []
        best_splitter = None
        cur_imp = get_impurity(labels)
        for col in range(len(feats[0])):
            # Note: we assume all features are numerical instead of categorical
            imp, splitter, left_idxs, right_idxs = \
                get_best_split_along_column(feats, labels, col,
                                            categorical=False)

            gain = cur_imp - imp
            if gain > best_gain:
                best_gain = gain
                best_left_idxs = left_idxs
                best_right_idxs = right_idxs
                best_splitter = {'col': col, 'val': splitter}

        self.splitter = best_splitter
        if self.splitter is None:
            self.is_leaf = True
        else:
            self.left_node = TreeNode()
            self.right_node = TreeNode()

            self.left_node.build_tree(feats[best_left_idxs], labels[best_left_idxs])
            self.right_node.build_tree(feats[best_right_idxs], labels[best_right_idxs])

        return

    def classify(self, feats):
        '''Classify sample according to built tree
        '''
        if self.is_leaf is False and self.splitter is None:
            raise Exception("Decision tree not built!")

        if self.is_leaf:
            return np.random.choice(self.labels)

        else:
            val = self.splitter['val']
            col = self.splitter['col']

            if self.categorical:
                if feats[col] == val:
                    label = self.left_node.classify(feats)
                else:
                    label = self.right_node.classify(feats)

            else:
                if feats[col] >= val:
                    label = self.left_node.classify(feats)
                else:
                    label = self.right_node.classify(feats)

        return label

def main():
    '''Main
    '''
    global TRN_DATA_PATH, NUM_SAMPLES, NUM_FEATS

    # no need to rescale for decision tree
    feats, labels = ut.get_data_from_csv(TRN_DATA_PATH, rescale=False)

    if NUM_SAMPLES < len(feats):
        feats, labels = ut.sample(feats, labels, NUM_SAMPLES)

    feats = feats.values
    if NUM_FEATS < len(feats[0]):
        idxs = np.random.choice(range(len(feats[0])), NUM_FEATS, replace=False)
        feats = feats[:, idxs]

    trn_feats, tst_feats, trn_labels, tst_labels = train_test_split(feats,
                                                                    labels,
                                                                    test_size=0.20,
                                                                    stratify=labels)

    # build tree
    print("Building decision tree...")
    decision_tree = TreeNode()
    decision_tree.build_tree(trn_feats, trn_labels.values)
    print("Done!")

    print("Checking accuracy on training set...")
    predictions = []
    for sample in trn_feats:
        result = decision_tree.classify(sample)
        predictions.append(result)

    # for checking only. must be 100% accuracy on training set
    print("Training Set Results:\n", classification_report(trn_labels, predictions))


    print("Using tree to predict labels...")
    predictions = []
    for sample in tst_feats:
        result = decision_tree.classify(sample)
        predictions.append(result)

    print("Test Set Results:\n", classification_report(tst_labels, predictions))

    visualize_data(pd.DataFrame(tst_feats), tst_labels, predictions)

    # display confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(tst_labels, predictions, normalize=True)
    plt.show()

    return 0

if __name__ == "__main__":
    main()
