# K-means
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import euclidean

# get data from:
#   https://www.kaggle.com/c/otto-group-product-classification-challenge
trn_data_path = 'datasets/otto-group-product-classification/train.csv'
num_samples = 2000
color_pool = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet',
              'brown', 'gold', 'black', 'pink', 'tan', 'lime', 'magenta',
              'teal', 'lavender', 'khaki', 'aqua', 'fuchsia', 'ivory']

def get_data_from_csv(data_path):
    data = pd.read_csv(data_path)
    feats = data.drop('target', axis=1).drop('id', axis=1).astype(np.int)
    # feature scaling
    feats = pd.DataFrame(preprocessing.scale(feats))
    labels = data['target'].map(lambda s : s.lstrip('Class_')).astype(np.int)

    return feats, labels

def sample(feats, labels, num_samples):
    splits = int(len(feats) / num_samples)
    sampler = StratifiedKFold(n_splits=splits, shuffle=True)
    for _, sampled_idxs in sampler.split(feats, labels):
        feats, labels = feats.iloc[sampled_idxs], labels.iloc[sampled_idxs]
        break

    return feats, labels

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

def visualize_data(feats, labels, ca):
    print("Visualizing data...")
    red_feats = reduce_to_2D_by_tsne(feats)

    label2col_map = ['red', 'orange', 'yellow', 'green', 'blue',
                     'violet', 'brown', 'gray', 'pink']

    label_list = np.unique(labels)
    cluster_list = np.unique(ca)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    for label in label_list:
        # get samples with label == label
        idxs = np.where(labels==label)
        # get components
        pc1, pc2 = red_feats['pc1'].values[idxs], red_feats['pc2'].values[idxs]
        # scatter plot w/ color based on labels
        ax[0].scatter(x=pc1, y=pc2, color=label2col_map[label-1],
                      alpha=0.5, label=label)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[0].set_title('Labeled Products')

    for i, cluster in enumerate(cluster_list):
        # get samples assigned to cluster
        idxs = np.where(ca==cluster)
        # get components
        pc1, pc2 = red_feats['pc1'].values[idxs], red_feats['pc2'].values[idxs]
        # scatter plot w/ color based on cluster
        ax[1].scatter(x=pc1, y=pc2, color=color_pool[i],
                      alpha=0.5, label=cluster)

    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC2')
    ax[1].set_title('Grouped Products')

    for graph in ax:
        graph.legend() # show legend
        graph.grid(True) # show gridlines

    plt.show()

def calculate_cost(X, u):
    '''
    Calculate k-means distortion/cost

    X- samples assigned to cluster
    u - cluster mean
    '''
    cost = 0
    cost += np.sum([(euclidean(x, u)**2) for x in X])
    return cost

def k_means(k, X, max_iter=100, print_every_iter=5):
    '''
    K-means from scratch

    k - no. of clusters
    X - samples
    '''
    if k > len(X):
        raise Exception("No. of clusters > no. of samples!")
    # randomly initialize means
    u_idxs = np.random.choice(range(len(X)), size=k, replace=True)
    U = X[u_idxs]
    # initialize cluster array
    c = np.array([-1] * len(X))
    
    for j in range(max_iter):
        U_prev = np.array(U)
        # cluster assignment
        for i, x in enumerate(X):
            u_dists = [euclidean(x, u)
                       for u in U]
            c[i] = np.argmin(u_dists)

        # update means
        cost = 0
        for k_idx in range(k):
            c_xs = X[np.where(c==k_idx)]
            if len(c_xs) == 0:
                # re-initialize mean/cluster
                idx = np.random.choice(range(len(X)))
                U[k_idx] = X[idx]
            else:
                U[k_idx] = np.mean(c_xs)

            # get cost
            cost += calculate_cost(c_xs, U[k_idx])

        if (j+1) % print_every_iter == 0:
            print("k = %d. Iter %d. Cost: %.2f" % (k, j+1, cost))

        # if no update, then minima is already found
        if np.array_equal(U, U_prev):
            break
    
    return c, cost

def show_elbow(K, costs):
    plt.subplots(figsize=(8, 5))
    plt.plot(K, costs, 'o-')
    plt.xticks(K)
    plt.grid(True)
    plt.xlabel('K', fontsize=15)
    plt.ylabel('Cost', fontsize=15)
    plt.title('Elbow method')
    plt.show()

def main():
    feats, labels = get_data_from_csv(trn_data_path)

    if num_samples < len(feats):
        feats, labels = sample(feats, labels, num_samples)

    tries = 3
    min_k = 3
    max_k = 12

    K = list(range(min_k, max_k + 1))
    K_ca = []
    K_costs = []

    for k in K:
        best_cost = 0.0
        best_ca = None
        for t in range(tries):
            print("-- Trial %d --" % (t+1))
            # get cluster assignment and cost
            ca, cost = k_means(k=k, X=feats.values)
            if t == 0:
                best_cost = cost
                best_ca = ca
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_ca = ca

        K_costs.append(best_cost)
        K_ca.append(best_ca)

    # elbow method to select k
    show_elbow(K, K_costs)
    # get input
    try:
        selected_k = int(input('Choose k (range: %d - %d): ' % (min_k, max_k)))
        if k < min_k or k > max_k:
            raise ValueError
    except ValueError:
        print('Invalid k given!')
        return 0

    idx = K.index(selected_k)
    ca = K_ca[idx]
    print("Selected k: %d. Final Cost: %.2f" % (selected_k, K_costs[idx]))

    # visualize results
    visualize_data(feats, labels, ca)

if __name__ == "__main__":
    main()