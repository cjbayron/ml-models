'''
K-means Clustering (from scratch) with elbow method selection
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

import common.utils as ut

# get data from:
#   https://www.kaggle.com/c/otto-group-product-classification-challenge
TRN_DATA_PATH = 'datasets/otto-group-product-classification/train.csv'
NUM_SAMPLES = 2000
COLOR_POOL = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet',
              'brown', 'gold', 'black', 'pink', 'tan', 'lime', 'magenta',
              'teal', 'lavender', 'khaki', 'aqua', 'fuchsia', 'ivory']

def visualize_data(feats, labels, ca):
    '''Display labeled data and clustered data
    '''
    print("Visualizing data...")
    red_feats = ut.reduce_to_2D_by_tsne(feats)

    label2col_map = ['red', 'orange', 'yellow', 'green', 'blue',
                     'violet', 'brown', 'gray', 'pink']

    label_list = np.unique(labels)
    cluster_list = np.unique(ca)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    for label in label_list:
        # get samples with label == label
        idxs = np.where(labels == label)
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
        idxs = np.where(ca == cluster)
        # get components
        pc1, pc2 = red_feats['pc1'].values[idxs], red_feats['pc2'].values[idxs]
        # scatter plot w/ color based on cluster
        ax[1].scatter(x=pc1, y=pc2, color=COLOR_POOL[i],
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
    U = X[np.random.choice(range(len(X)), size=k, replace=True)]
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
            c_xs = X[np.where(c == k_idx)]
            if len(c_xs) == 0:
                # re-initialize mean/cluster
                U[k_idx] = X[np.random.choice(range(len(X)))]
            else:
                U[k_idx] = np.mean(c_xs)

            # get cost
            cost += calculate_cost(c_xs, U[k_idx])

        cost /= len(X)
        if (j+1) % print_every_iter == 0:
            print("k = %d. Iter %d. Cost: %.6f" % (k, j+1, cost))

        # if no update, then minima is already found
        if np.array_equal(U, U_prev):
            break

    return c, cost

def show_elbow(K, costs):
    '''Graph "elbow" for selection of k
    '''
    plt.subplots(figsize=(8, 5))
    plt.plot(K, costs, 'o-')
    plt.xticks(K)
    plt.grid(True)
    plt.xlabel('K', fontsize=15)
    plt.ylabel('Cost', fontsize=15)
    plt.title('Elbow method')
    plt.show()

def main():
    '''Main
    '''
    feats, labels = ut.get_data_from_csv(TRN_DATA_PATH)

    if NUM_SAMPLES < len(feats):
        feats, labels = ut.sample(feats, labels, NUM_SAMPLES)

    tries = 5
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
        return 1

    idx = K.index(selected_k)
    ca = K_ca[idx]
    print("Selected k: %d. Final Cost: %.6f" % (selected_k, K_costs[idx]))

    # visualize results
    visualize_data(feats, labels, ca)
    return 0

if __name__ == "__main__":
    main()
