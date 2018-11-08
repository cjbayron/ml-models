# SVM Multi-class Classifier with PCA and t-SNE visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# get data from:
#   https://www.kaggle.com/c/otto-group-product-classification-challenge
trn_data_path = 'datasets/otto-group-product-classification/train.csv'

showHist = False
showHeatmap = False
visualizeResultsPCA = False
visualizeResultsTSNE = True

num_samples = 10000

def get_data_from_csv(data_path):
    data = pd.read_csv(data_path)
    feats = data.drop('target', axis=1).drop('id', axis=1).astype(np.int)
    # feature scaling
    feats = pd.DataFrame(preprocessing.scale(feats))
    labels = data['target'].map(lambda s : s.lstrip('Class_')).astype(np.int)

    return feats, labels

def show_hist(labels):
    # plot histogram
    labels.hist(bins=9)
    plt.title('Product Classes Distribution')
    plt.ylabel('No. of samples')
    plt.xlabel('Class')
    plt.show()

def show_heatmap(feats):
    # plot heatmap
    corr = feats.corr()
    sns.heatmap(corr, cmap='coolwarm', fmt='0.2f')
    plt.show()

def sample(feats, labels, num_samples):
    splits = int(len(feats) / num_samples)
    sampler = StratifiedKFold(n_splits=splits, shuffle=True)
    for _, sampled_idxs in sampler.split(feats, labels):
        feats, labels = feats.iloc[sampled_idxs], labels.iloc[sampled_idxs]
        break

    return feats, labels

def reduce_to_2D_by_pca(df):
    print("Reducing dimension by PCA...")
    pcg = PCA(n_components=2)
    # get principal components
    pcs = pcg.fit_transform(df.values)
    print("Done.")
    # "compressed" data
    comp_data = pd.DataFrame()
    comp_data['pc1'] = pcs[:,0]
    comp_data['pc2'] = pcs[:,1]

    print("PCA Variance: ", pcg.explained_variance_ratio_)
    return comp_data

def reduce_to_2D_by_tsne(df):
    print("Reducing dimension by t-SNE...")
    tsne_g = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_res = tsne_g.fit_transform(df.values)
    print("Done.")
    # "compressed" data
    comp_data = pd.DataFrame()
    comp_data['pc1'] = tsne_res[:,0]
    comp_data['pc2'] = tsne_res[:,1]

    return comp_data

def visualize_data(feats, preds, dim_reduction='tsne'):
    print("Visualizing data...")
    red_feats = None
    if dim_reduction == 'tsne':
        red_feats = reduce_to_2D_by_tsne(feats)
    elif dim_reduction == 'pca':
        red_feats = reduce_to_2D_by_pca(feats)

    red_feats['label'] = preds.astype(str)
    sc_plot = ggplot(red_feats, aes(x='pc1', y='pc2', color='label')) \
              + geom_point(size=70, alpha=0.5) \
              + ggtitle("Predicted Product Classes")
    # show plot
    sc_plot.show()

if __name__ == "__main__":

    feats, labels = get_data_from_csv(trn_data_path)

    if showHist:
        show_hist(labels)

    if showHeatmap:
        show_heatmap(feats)

    if num_samples < len(feats):
        feats, labels = sample(feats, labels, num_samples)

    if showHist:
        # to ensure that the sampled data 
        #   follows same distribution as original data
        show_hist(labels)

    trn_feats, tst_feats, trn_labels, tst_labels = train_test_split(feats,
                                                                    labels,
                                                                    test_size = 0.20,
                                                                    stratify=labels)

    # svm training
    svm_clf = SVC(gamma='scale', decision_function_shape='ovo')
    print("Fitting SVM classifier on %s samples..." % len(trn_feats))
    svm_clf.fit(trn_feats, trn_labels)
    print("Done.")

    # svm testing
    print("Testing SVM classifier on %s samples." % len(tst_feats))
    tst_preds = svm_clf.predict(tst_feats)

    if showHist:
        show_hist(tst_labels)
        show_hist(pd.DataFrame(tst_preds))
    
    print("Results:\n", classification_report(tst_labels, tst_preds))


    if visualizeResultsTSNE:
        visualize_data(tst_feats, tst_preds, dim_reduction='tsne')

    if visualizeResultsPCA:
        visualize_data(tst_feats, tst_preds, dim_reduction='pca')
