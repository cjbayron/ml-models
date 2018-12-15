'''
Utility functions
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold

def get_data_from_csv(data_path, rescale=True):
    '''Read data from CSV file and perform feature scaling
    '''
    data = pd.read_csv(data_path)
    feats = data.drop('target', axis=1).drop('id', axis=1).astype(np.int)
    if rescale is True:
        # feature scaling
        feats = preprocessing.scale(feats)

    feats = pd.DataFrame(feats)
    labels = data['target'].map(lambda s: s.lstrip('Class_')).astype(np.int)

    return feats, labels

def sample(feats, labels, num_samples):
    '''Randomly sample from available data (feats, labels)
    while maintaining original label distribution
    '''
    splits = int(len(feats) / num_samples)
    sampler = StratifiedKFold(n_splits=splits, shuffle=True)
    for _, sampled_idxs in sampler.split(feats, labels):
        feats, labels = feats.iloc[sampled_idxs], labels.iloc[sampled_idxs]
        break

    return feats, labels

def reduce_to_2D_by_tsne(df):
    '''Reduce dimension using t-SNE
    '''
    print("Reducing dimension by t-SNE...")
    tsne_g = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_res = tsne_g.fit_transform(df.values)
    print("Done.")
    # "compressed" data
    comp_data = pd.DataFrame()
    comp_data['pc1'] = tsne_res[:, 0]
    comp_data['pc2'] = tsne_res[:, 1]

    return comp_data

def reduce_to_2D_by_pca(df):
    '''Reduce dimension using PCA
    '''
    print("Reducing dimension by PCA...")
    pcg = PCA(n_components=2)
    # get principal components
    pcs = pcg.fit_transform(df.values)
    print("Done.")
    # "compressed" data
    comp_data = pd.DataFrame()
    comp_data['pc1'] = pcs[:, 0]
    comp_data['pc2'] = pcs[:, 1]

    print("PCA Variance: ", pcg.explained_variance_ratio_)
    return comp_data
