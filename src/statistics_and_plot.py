import pandas as pd
import numpy as np
import re
from collections import Counter
import statistics
from more_itertools import locate
import operator
import matplotlib.pyplot as plt
import pickle
import sys



def manual_inspection(i, df, hmm_seq, unique_ids):### Manual Inspection
    '''choose index of dialog to inspect.
    Returns a dataframe with columns DA, hidden sequence and text'''
    #sanity check first
    print(len(hmm_seq[i][0]) == len(hmm_seq[i][1]) == len(df['new_text'].loc[df['conversation_id'] == unique_ids[i]])) # nao tem pq sem mesmo index ja que hmm foi sorted
    print(len(hmm_seq[i][0]))
    print(len(hmm_seq[i][1]))
    print(len(df['new_text'].loc[df['conversation_id'] == unique_ids[i]]))


    inspection = pd.DataFrame(list(zip(hmm_seq[i][0], hmm_seq[i][1], df['new_text'].loc[df['conversation_id'] == unique_ids[i]])),columns =['DA', 'HMM', 'Text'])
    return inspection


### AUX FUNCT
def count_and_order_seq(all_da_seq):
    seq_counts = {}

    for seq in all_da_seq:
        for sub_seq in seq:
            sub_seq = ' '.join(sub_seq)
            seq_counts[sub_seq] = seq_counts[sub_seq] + 1 if sub_seq in seq_counts else 1
            
    sorted_d = dict(sorted(seq_counts.items(), key=operator.itemgetter(1),reverse=True))
    return sorted_d


def sorted_seq_and_counts(hmm_seq):

    all_da_seq = []
    all_hid_seq = []
    count = 0

    for e, i_ in enumerate(hmm_seq):
        s_da = hmm_seq[e][0]
        s_hid = hmm_seq[e][1]
        
        if len(s_da)==len(s_hid): #sanity check that all contain same len
            #https://thispointer.com/python-how-to-find-all-indexes-of-an-item-in-a-list/
            index_new = list(locate(s_hid, lambda a: a == 'New')) 
            #https://www.geeksforgeeks.org/python-custom-list-split/?ref=lbp 
            split_hidden_lab = [s_hid[i : j] for i, j in zip([0] +  index_new, index_new + [None])]
            split_da_lab = [s_da[i : j] for i, j in zip([0] +  index_new, index_new + [None])]

            all_hid_seq.append(split_hidden_lab)
            all_da_seq.append(split_da_lab)
        else: count+=1

    sorted_d = count_and_order_seq(all_da_seq)

    seq_to_plot = []

    for key, value in sorted_d.items():
        seq_to_plot.append(key.split(' '))
        
    max_len = len(max(seq_to_plot, key=len))
    min_len = len(min(seq_to_plot, key=len))

    seq_sizes = {}

    for seq in seq_to_plot:
        for leng in range(max_len):
            if len(seq) == leng:
                seq_sizes[leng] = seq_sizes[leng] + sorted_d.get(' '.join(seq)) if leng in seq_sizes else sorted_d.get(' '.join(seq))

    # seq_sizes = sorted(seq_sizes.items(), key=operator.itemgetter(1),reverse=True)

    print('minimum len: '+ str(min_len))
    print('maximum len: '+ str(max_len))
    print('counts of number of sequences of each len (ranking): ' + str(seq_sizes))   #(sorted(seq_sizes.items(), key=operator.itemgetter(1),reverse=True)))

    return sorted_d, seq_sizes, seq_to_plot ###save sorted_d in json/pickle



# # #to see only sequences of specific len I can use the below:
# # for e, seq in enumerate(seq_to_plot):
# #     if len(seq) ==10:
# #         print(seq, sorted_d.get(' '.join(seq)))



##### FAZER FUNCAO AQUI, VOU USAR O MESMO PRAS BASELINES
def plot_literal_count_per_seq(seq_sizes):
    names = list(seq_sizes.keys())
    values = list(seq_sizes.values())

    plt.bar(range(len(seq_sizes)), values, tick_label=names)
    plt.savefig('plot_literal_count_per_seq.png')
    plt.show()

    # return plt.bar(range(len(seq_sizes)), values, tick_label=names)


##### FAZER FUNCAO AQUI, VOU USAR O MESMO PRAS BASELINES
def plot_grouped_per_len_seq(sorted_d, seq_to_plot):
    
    #isso seria pra plotear todos os elementos do dict
    sorted_names = list(sorted_d.keys())
    sorted_values = list(sorted_d.values())
    sorted_len_each_seq = []

    for s in seq_to_plot:
    #     print(len(s), s)
        sorted_len_each_seq.append(len(s))

    df_plot = pd.DataFrame(list(zip(sorted_names, sorted_values, sorted_len_each_seq)),columns =['Name', 'values', 'seq_len'])

    df_plot.drop(0, axis=0, inplace=True)
    df_plot.head(10)

    print('sequence sizes in different ranges of sequence count:')
    print('number of sequences with more than 1000 counts and their length: ')
    print(df_plot['seq_len'].loc[df_plot['values'] > 1000].value_counts())
    print('number of sequences between 500 and 1000 counts and their length: ')
    print(df_plot['seq_len'].loc[(df_plot['values'] > 500) & (df_plot['values'] < 1000)].value_counts())
    print('number of sequences with less than 500 counts and their length: ')
    print(df_plot['seq_len'].loc[df_plot['values'] < 500].value_counts())

    ##### FAZER FUNCAO AQUI, VOU USAR O MESMO PRAS BASELINES
    print(df_plot['seq_len'].max())
    print(df_plot['seq_len'].min())
    print('mean len: '+ str(df_plot['seq_len'].mean()))
    
    fig, ax = plt.subplots()
    df_plot.hist('seq_len', ax=ax)
    fig.savefig('hist.png')
    


def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv

    df = pd.read_csv(argv[1], index_col=0) #pd.read_csv('./data_TM2/processed/processed_utterances_sentence_DA_labeling.csv', index_col=0)

    #open files i'll need
    a_file = open("hmm_results.pkl", "rb")
    hmm_seq = pickle.load(a_file)

    sec_file = open("input_hmm.pkl", "rb")
    input_hmm = pickle.load(sec_file)
    unique_ids = input_hmm.get('unique_ids') # same as input_hmm['unique_ids']

    i = 10
    inspection_df = manual_inspection(i, df, hmm_seq, unique_ids)
    sorted_d, seq_sizes, seq_to_plot = sorted_seq_and_counts(hmm_seq)
    plot_literal_count_per_seq(seq_sizes)
    plot_len_grouped = plot_grouped_per_len_seq(sorted_d, seq_to_plot)
    
    #save new results to file
    with open('sorted_hmm_dict.pkl', 'wb') as fp:
        pickle.dump(sorted_d, fp)
    
    print('done!')



if __name__ == '__main__':
    main()


#type in terminal:
#python3 statistics_and_plot.py './data_TM2/processed/processed_utterances_sentence_DA_labeling.csv'