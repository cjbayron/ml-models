'''
Module for extracting features from Myers-Brigges Personality Dataset
'''
from collections import Counter
import itertools
import threading

import numpy as np
import pandas as pd

# for getting the top words
NUM_TOP_WORDS = 100
# for counting words
NUM_THREADS = 4

# global variables
new_feats = pd.DataFrame()

# classes
class CountWordsThread(threading.Thread):
    ''' Thread for counting word occurence
    '''
    def __init__(self, keywords, all_user_words, keyword_idx):
        threading.Thread.__init__(self)
        self.keywords = keywords
        self.all_user_words = all_user_words
        # for thread identification
        self.keyword_idx = keyword_idx

    def run(self):
        global new_feats
        for word in self.keywords:
            # get occurence of word for each user
            word_freq = [sum(np.array(user_words) == word)
                         for user_words in self.all_user_words]

            new_feats['occ_' + word] = word_freq

            print("Processing for word %d: '%s' complete!" % (self.keyword_idx, word))
            self.keyword_idx += 1

# functions
def gen_input_cols(dataset):
    ''' Generate feature columns
    '''
    global new_feats

    # --- 1st feature ---
    # generate number of words per sentence
    num_words_per_sent = np.zeros(len(dataset))
    all_words = []
    all_user_words = [] # store here words used by each user
    for ix, post in enumerate(dataset['posts']):
        sents = post.split('|||')

        nw = 0
        user_words = []
        for sent in sents:
            # split into words, then remove empty elements
            ws = list(filter(None, sent.lower().split(' ')))
            # get num of words
            nw += len(ws)
            # push to list of words of user
            user_words.extend(ws)

        all_user_words.append(user_words)

        nw /= len(sents)
        num_words_per_sent[ix] = nw

    # push to feature columns
    new_feats['num_words_per_sent'] = num_words_per_sent

    # --- 2nd feature ---
    # flatten all_user_words
    all_words = list(itertools.chain(*all_user_words))
    # get most common
    counts = Counter(all_words).most_common(n=NUM_TOP_WORDS)
    # convert to np array to allow index commas
    keywords = np.array(counts)[:, 0]

    # separator for dividing task into subsets
    idx_sep = int(NUM_TOP_WORDS / NUM_THREADS)
    seps = [0]
    seps.extend([idx_sep*(i+1) for i in range(NUM_THREADS)])
    seps = np.array(seps)
    for i in range((NUM_TOP_WORDS % NUM_THREADS)):
        seps[i+1:] += 1

    countWordsThreads = []
    print("Counting word occurence using %d threads..." % (NUM_THREADS))
    for thd_idx in range(NUM_THREADS):
        thread = CountWordsThread(keywords[seps[thd_idx]:seps[thd_idx+1]],
                                  all_user_words,
                                  seps[thd_idx])
        # assign each thread a subset to process
        countWordsThreads.append(thread)
        countWordsThreads[thd_idx].start()

    for thd_idx in range(NUM_THREADS):
        countWordsThreads[thd_idx].join()

    # TO CONSIDER: aside from most common maybe also consider mid words???
    # cause maybe the mid words are more significant words

def gen_output_cols(dataset):
    ''' Generate output columns
    '''
    # output: E/I, N/S, F/T, J/P = 0/1
    # we treat each characteristic as mutually exclusive
    # we use 0/1 to for the quality in each characteristic
    zeros_chars = ['E', 'N', 'F', 'J']
    ones_chars = ['I', 'S', 'T', 'P']

    for ix, char in enumerate(ones_chars):
        char_vals = np.zeros(len(dataset['type']))
        char_vals[[char in typ for typ in dataset['type']]] = 1

        char_str = zeros_chars[ix] + '_' + ones_chars[ix]
        new_feats[char_str] = char_vals

def gen_processed_data(dataset, path):
    ''' Generate features from data
    '''
    global new_feats

    gen_input_cols(dataset)
    gen_output_cols(dataset)

    new_feats.to_csv(path, index=False)
