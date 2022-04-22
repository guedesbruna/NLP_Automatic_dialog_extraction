import pandas as pd
import numpy as np
import re
from collections import Counter
import statistics
import sys
from itertools import islice
from random import randint
import random 
from more_itertools import locate
import pickle


##### AUX FUNCT
def random_chunk(lista, min_chunk=1, max_chunk=5): 
    '''https://stackoverflow.com/questions/21439011/best-way-to-split-a-list-into-randomly-sized-chunks'''

    it = iter(lista) #The iter() function creates an object which can be iterated one element at a time. These objects are useful when coupled with loops like for loop, while loop.
    while True:
        nxt = list(islice(it,randint(min_chunk,max_chunk))) # islice(iterable, start, stop, step), randint(lower_bound, higher_boundary)
        if nxt:
            yield nxt
        else:
            break            


##### AUX FUNCT
def create_hid_baseline(all_da_seq):
    '''create hidden state comparison for probability split'''

    prob_hmm = all_da_seq
    for e, i in enumerate(all_da_seq):
        for f, j in enumerate(i):
            for g, k in enumerate(j):
                if g == 0:
                    prob_hmm[e][f][g] = 'New'
                else:
                    prob_hmm[e][f][g] = 'Current'
    
    return prob_hmm



def rand_baseline(unigram, df):
    '''Create random baseline by Uniformely Random splitting a list into chunks varying between min and max chunk size.
    Outputs a list with the split DAs, a list with the hidden state equivalents, and a dataframe to visualize dialog, da, and hidden'''
    
    random_baseline = []

    for dialog in unigram:
        random_baseline.append(list(random_chunk(dialog[2:-2])))

    rand_hid_st = create_hid_baseline(random_baseline)
    rand_bas_df = pd.DataFrame(list(zip(random_baseline,rand_hid_st, df['new_text'])), columns=['DA','rand_split','Text'])
    
    return random_baseline, rand_hid_st, rand_bas_df



def split_prob_baseline(unigram, df, prob=0.3):
    '''Create splits in list and append as sublists with a probability between each 2 DAs.
    Outputs list with DA split, list with hidden equivalents and dataframe to visualize dialog, da, and hidden '''

    all_da_seq_rand = []

    for e, dialog in enumerate(unigram):
        dialog = dialog[2:-2]
        indexes = []
        for ind in range(len(dialog)):
            coin = random.uniform(0, 1)  #every time toss coin for rand number between 0 and 1
            if coin <= prob: #ex 0.3:
                indexes.append(ind)
        split_da_lab_rand = [dialog[i : j] for i, j in zip([0] +  indexes, indexes + [None])]
        all_da_seq_rand.append(split_da_lab_rand)


    prob_hmm = create_hid_baseline(all_da_seq_rand)
    prob_bas_df = pd.DataFrame(list(zip(all_da_seq_rand,prob_hmm, df['new_text'])), columns=['DA','rand_split','Text'])
    
    return all_da_seq_rand, prob_hmm, prob_bas_df



def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv

    df = pd.read_csv(argv[1], index_col=0) #pd.read_csv('./data_TM2/processed/processed_utterances_sentence_DA_labeling.csv', index_col=0)
        # prob = argv[2] #only needed if prob != 0.3

    a_file = open("input_hmm.pkl", "rb")
    input_hmm = pickle.load(a_file)

    unigram = input_hmm.get('unigram')


    random_baseline, rand_hid_st, rand_bas_df = rand_baseline(unigram, df)
    all_da_seq_rand, prob_hmm, prob_bas_df = split_prob_baseline(unigram, df)

    print('done!')


if __name__ == '__main__':
    main()


#type in terminal:
#python3 baselines.py './data_TM2/processed/processed_utterances_sentence_DA_labeling.csv'


