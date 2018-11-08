import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# get data from:
#   https://www.kaggle.com/c/otto-group-product-classification-challenge
trn_data_path = 'datasets/otto-group-product-classification/train.csv'
tst_data_path = 'datasets/otto-group-product-classification/test.csv'

showHist = False
showHeatmap = False

num_samples = 10000

def show_hist(labels):
    # plot histogram
    labels.hist(bins=9)
    plt.title('Product Classes Distribution')
    plt.ylabel('No. of samples')
    plt.xlabel('Class')
    plt.show()

def sample(feats, labels, num_samples):
    splits = int(len(feats) / num_samples)
    sampler = StratifiedKFold(n_splits=splits, shuffle=True)
    for _, sampled_idxs in sampler.split(feats, labels):
        feats, labels = feats.iloc[sampled_idxs], labels.iloc[sampled_idxs]
        break

    return feats, labels

if __name__ == "__main__":

    data = pd.read_csv(trn_data_path)
    feats = data.drop('target', axis=1).drop('id', axis=1).astype(np.int)
    # feature scaling
    feats = pd.DataFrame(preprocessing.scale(feats))
    labels = data['target'].map(lambda s : s.lstrip('Class_')).astype(np.int)

    if showHist:
        show_hist(labels)

    if showHeatmap:
        # plot heatmap
        corr = feats.corr()
        sns.heatmap(corr, cmap='coolwarm', fmt='0.2f')
        plt.show()

    if num_samples < len(feats):
        feats, labels = sample(feats, labels, num_samples)

    if showHist:
        show_hist(labels)

    trn_feats, tst_feats, trn_labels, tst_labels = train_test_split(feats,
                                                                    labels,
                                                                    test_size = 0.20,
                                                                    stratify=labels)

    svm_clf = SVC(gamma='scale', decision_function_shape='ovo')
    print("Fitting SVM classifier on %s samples..." % len(trn_feats))
    svm_clf.fit(trn_feats, trn_labels)
    print("Done.")

    print("Testing SVM classifier on %s samples." % len(tst_feats))
    tst_preds = svm_clf.predict(tst_feats)

    if showHist:
        show_hist(tst_labels)
        show_hist(pd.DataFrame(tst_preds))
    
    print("Results:\n", classification_report(tst_labels, tst_preds))