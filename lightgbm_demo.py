'''
Demonstration of Microsoft's LightGBM
- dataset preparation and parameter configuration
'''
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from misc import process_mbti_data as preproc

# raw data contains twitter posts
RAW_DATA_PATH = 'datasets/mbti-myers-briggs-personality-type-dataset/mbti_1.csv'
# processed data contains features extracted from the posts
# (num words per sentence/post, occurence of 100 words)
PROC_DATA_PATH = 'datasets/mbti-myers-briggs-personality-type-dataset/mbti_processed.csv'

GEN_NEW_PROC_DATA = False

label_cols = ['E_I', 'N_S', 'F_T', 'J_P']

def main():
    '''Main
    '''
    if not os.path.exists(PROC_DATA_PATH) or GEN_NEW_PROC_DATA:
        raw_dataset = pd.read_csv(RAW_DATA_PATH)
        preproc.gen_processed_data(raw_dataset, PROC_DATA_PATH)

    dataset = pd.read_csv(PROC_DATA_PATH)

    # normalize input features
    feats = dataset.drop(columns=label_cols, axis=1).astype(np.float64)
    feats = preprocessing.scale(feats)

    print("Input Features shape: ", feats.shape)

    # train classifier for each label
    for label_name in label_cols:
        labels = dataset[label_name].astype(np.int)

        trn_feats, tst_feats, trn_labels, tst_labels = train_test_split(feats,
                                                                        labels,
                                                                        test_size=0.10,
                                                                        stratify=labels)

        trn_data = lgb.Dataset(trn_feats, label=trn_labels)
        val_data = lgb.Dataset(tst_feats, label=tst_labels)

        # set params
        params = {'objective': 'binary',
                  'num_threads': 2,
                  'metric': 'binary',
                  'verbosity': 0,
                  'max_bin': 65535}

        num_boost_round = 1000
        learning_rates = (([0.01] * 50) + ([0.001] * 50)) * int(num_boost_round / 100)

        lgb.train(params,
                  trn_data,
                  num_boost_round=num_boost_round,
                  valid_sets=[trn_data, val_data],
                  verbose_eval=100,
                  #early_stopping_rounds=10,
                  learning_rates=learning_rates)

if __name__ == "__main__":
    main()
