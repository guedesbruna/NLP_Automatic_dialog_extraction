# import os
import pandas as pd
import numpy as np
# from pandas.io.json import json_normalize
import re
# from nltk.tokenize import word_tokenize
# import sklearn
from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score
import sys

def metrics_DA_labeling(df):
    '''To compare gold labels with synthetically annotated data
    Input is the dataframe with binary gold labels and binary syntethical labels'''
    
    #sanity check that columns of gold and synt are aligned:

    print(df.columns[20])
    print(df.columns[20+14])

    # df.iloc[:,[9,23]]

    DA = ['repair_initiator','greeting','request_summary','confirmation','receipt','disconfirmation',
        'sequence_closer','completion_check','hold_request','partial_request', 'detail_request', 'grant', 'answer'] 
    kappa = []
    prec_rec_f1 = []
    count_g = []
    count_s = []

    for e,i in enumerate(range(9,22)):
        gold = np.asarray([0 if val != DA[e] else 1 for val in df.iloc[:, i]])
        synt = np.asarray([0 if val != DA[e] else 1 for val in df.iloc[:, (i+14)]])
        kappa.append(cohen_kappa_score(gold, synt))
        prec_rec_f1.append(precision_recall_fscore_support(gold, synt, average='macro', zero_division=0))
        unique_g, counts_g = np.unique(gold, return_counts=True)
        unique_s, counts_s = np.unique(synt, return_counts=True)
        count_g.append(counts_g[1])
        count_s.append(counts_s[1])

    # print('kappa: ' + str(np.mean(kappa)))  
    # kappa_scores = list(zip(DA,kappa))

    kappa_scores = pd.DataFrame(kappa, index = DA, columns = ['Kappa'])
    # print(kappa_scores)

    metrics = pd.DataFrame(prec_rec_f1, index = DA, columns = ['Precision','Recall','F1_macro','Support'])
    metrics = metrics.drop(columns = 'Support')
    # metrics = metrics.insert(loc=0,column=count_g, value='Count_Gold')
    metrics['Cohen\'s Kappa'] = kappa_scores
    metrics['Count Gold'] = count_g
    metrics['Count Synt'] = count_s
    metrics.sort_values(by='F1_macro', ascending=False)

    return metrics

def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv

    #set path to file    
    p = argv[1]
    df = pd.read_csv(p, index_col=0)

    metrics = metrics_DA_labeling(df)

    metrics.to_csv('./statistics_on_DA_labeling/metrics_DA_labeling.csv')
    print('done!')
    return metrics

if __name__ == '__main__':
    main()

#type in the terminal
#python3 2.metrics_DA_labeling.py ./data_TM2/GOLD_and_SYNT_annotated_data.csv