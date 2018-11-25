'''
SVM Multi-class Classifier with PCA and t-SNE visualization
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import common.utils as ut

# get data from:
#   https://www.kaggle.com/c/otto-group-product-classification-challenge
TRN_DATA_PATH = 'datasets/otto-group-product-classification/train.csv'

SHOW_HIST = False
SHOW_HEATMAP = False
VISUALISE_RESULTS_PCA = False
VISUALISE_RESULTS_TSNE = True

NUM_SAMPLES = 10000

def show_hist(labels):
    '''Display label histogram
    '''
    labels.hist(bins=9)
    plt.title('Product Classes Distribution')
    plt.ylabel('No. of samples')
    plt.xlabel('Class')
    plt.show()

def show_heatmap(feats):
    '''Plot heatmap to explore correlation between features
    '''
    corr = feats.corr()
    sns.heatmap(corr, cmap='coolwarm', fmt='0.2f')
    plt.show()

def visualize_data(feats, preds, dim_reduction='tsne'):
    '''Visualize data with predicted labels in two-dimensions
    '''
    print("Visualizing data...")
    red_feats = None
    if dim_reduction == 'tsne':
        red_feats = ut.reduce_to_2D_by_tsne(feats)
    elif dim_reduction == 'pca':
        red_feats = ut.reduce_to_2D_by_pca(feats)

    red_feats['label'] = preds.astype(str)
    sc_plot = ggplot(red_feats, aes(x='pc1', y='pc2', color='label')) \
              + geom_point(size=70, alpha=0.5) \
              + ggtitle("Predicted Product Classes")
    # show plot
    sc_plot.show()

if __name__ == "__main__":

    FEATS, LABELS = ut.get_data_from_csv(TRN_DATA_PATH)

    if SHOW_HIST:
        show_hist(LABELS)

    if SHOW_HEATMAP:
        show_heatmap(FEATS)

    if NUM_SAMPLES < len(FEATS):
        FEATS, LABELS = ut.sample(FEATS, LABELS, NUM_SAMPLES)

    if SHOW_HIST:
        # to ensure that the sampled data
        #   follows same distribution as original data
        show_hist(LABELS)

    TRN_FEATS, TST_FEATS, TRN_LABELS, TST_LABELS = train_test_split(FEATS,
                                                                    LABELS,
                                                                    test_size=0.20,
                                                                    stratify=LABELS)

    # svm training
    SVM_CLF = SVC(gamma='scale', decision_function_shape='ovo')
    print("Fitting SVM classifier on %s samples..." % len(TRN_FEATS))
    SVM_CLF.fit(TRN_FEATS, TRN_LABELS)
    print("Done.")

    # svm testing
    print("Testing SVM classifier on %s samples." % len(TST_FEATS))
    TST_PREDS = SVM_CLF.predict(TST_FEATS)

    if SHOW_HIST:
        show_hist(TST_LABELS)
        show_hist(pd.DataFrame(TST_PREDS))

    print("Results:\n", classification_report(TST_LABELS, TST_PREDS))

    if VISUALISE_RESULTS_TSNE:
        visualize_data(TST_FEATS, TST_PREDS, dim_reduction='tsne')

    if VISUALISE_RESULTS_PCA:
        visualize_data(TST_FEATS, TST_PREDS, dim_reduction='pca')
