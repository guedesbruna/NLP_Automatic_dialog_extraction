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
import copy 

from ngrams import flatten
from statistics_and_plot import sorted_seq_and_counts, plot_literal_count_per_seq, plot_grouped_per_len_seq


random.seed(237)  #check if seed is working
print(random.random()) # must always be the same number: 0.5714025946899135


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

    prob_hmm = copy.deepcopy(all_da_seq)
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

    flat_random_baseline = flatten(random_baseline)
    flat_random_baseline = flatten(flat_random_baseline)

    flat_rand_hid_st = flatten(rand_hid_st)
    flat_rand_hid_st = flatten(flat_rand_hid_st)

    rand_bas_df = pd.DataFrame(list(zip(flat_random_baseline,flat_rand_hid_st, df['new_text'])), columns=['DA','rand_split','Text'])
    
    # print(random_baseline)
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

    flat_prob_baseline = flatten(all_da_seq_rand)
    flat_prob_baseline = flatten(flat_prob_baseline)

    flat_prob_hid_st = flatten(prob_hmm)
    flat_prob_hid_st = flatten(flat_prob_hid_st)


    prob_bas_df = pd.DataFrame(list(zip(flat_prob_baseline,flat_prob_hid_st, df['new_text'])), columns=['DA','rand_split','Text'])
    
    return all_da_seq_rand, prob_hmm, prob_bas_df



def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv

    df = pd.read_csv(argv[1], index_col=0) #pd.read_csv('./data_TM2/processed/processed_utterances_sentence_DA_labeling.csv', index_col=0)
    model_name =  ['random_baseline', 'prob_baseline'] #argv[2]

    a_file = open("./generated_files/input_hmm.pkl", "rb")
    input_hmm = pickle.load(a_file)

    unigram = input_hmm.get('unigram')

    random_baseline, rand_hid_st, rand_bas_df = rand_baseline(unigram, df)
    prob_baseline, prob_hid_st, prob_bas_df = split_prob_baseline(unigram, df)

    results = {'rand_baseline_split': random_baseline, 'rand_baseline_hidden_states':rand_hid_st, 'df_visualization_splits_and_text_rand':rand_bas_df,
               'prob_baseline_split': prob_baseline, 'prob_baseline_hidden_states':prob_hid_st, 'df_visualization_splits_and_text_prob':prob_bas_df }

    #save new results to file
    with open('./generated_files/baseline_splits_hidden_and_dfvisualization.pkl', 'wb') as fp:
        pickle.dump(results, fp)

    #from statistics_and_plot script
    for model in model_name:
        if model == 'random_baseline':
            da_seq = random_baseline
            hid_seq = rand_hid_st
        elif model == 'prob_baseline':
            da_seq = prob_baseline
            hid_seq = prob_hid_st

        sorted_d, seq_sizes, seq_to_plot = sorted_seq_and_counts(da_seq)
        plot_literal_count_per_seq(seq_sizes, model)
        plot_len_grouped = plot_grouped_per_len_seq(sorted_d, seq_to_plot, model)

        with open('./generated_files/sorted_dict_'+ model + '.pkl', 'wb') as f:
            pickle.dump(sorted_d, f)




    print('done!')


if __name__ == '__main__':
    main()


#type in terminal:
# python3 baselines.py '../data_TM2/processed/processed_utterances_sentence_DA_labeling.csv'
