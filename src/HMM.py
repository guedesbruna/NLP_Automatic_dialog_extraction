import pandas as pd
import numpy as np
import re
from collections import Counter
import statistics
import pickle
import sys
from statistics_and_plot import manual_inspection, sorted_seq_and_counts_hmm, plot_literal_count_per_seq, plot_grouped_per_len_seq


def viterbi(obs, states, start_p, trans_p, emit_p, emit_p_bi, emit_p_tri): 
    ''' # from wikipedia, code inclusive: https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
    Code should receive emition prob of uni, bi and trigrams
    This also requires as input sentences (i.e: obs) with padding=2 on each side'''

    V = [{}]

    for st in states:
        V[0] [st] = {"prob": start_p[st] * emit_p[st] [obs[0]], "prev": None} #### OLD
        V.append({})
        V[1] [st] = {"prob": start_p[st] * emit_p[st] [obs[1]], "prev": None} #### NEW


    # Run Viterbi when t > 1
    for t in range(2, len(obs)-2): 
        V.append({})
        for st in states: #New and Current            
            max_tr_prob = V[t - 1] [states[0]] ["prob"] * trans_p[states[0]] [st] #0.026402178674778336*0.1 and same*0.9 ######### NEW
            prev_st_selected = states[0] 
            
            for prev_st in states[1:]: #(states[1:]) >>> Current
                tr_prob = V[t - 1] [prev_st] ["prob"] * trans_p[prev_st] [st] # a diferenca aqui em relacao ao max_tr_prob é que aqui se calcula o previous state
                if tr_prob > max_tr_prob: #if other state is higher than state 0 (i.e if Current > New)
                    max_tr_prob = tr_prob #then new max is Current instead on New. Here we store the actual probability
                    prev_st_selected = prev_st #and previous state is updated. Here we store name of state

            # emit_p for trigram,bigram and unigram if new and if current
            if st == 'New':
                max_prob = max_tr_prob * statistics.fmean([emit_p[st] [obs[t]], emit_p_bi[st].get((obs[t],obs[t+1])), emit_p_tri[st].get((obs[t],obs[t+1],obs[t+2]))]) 
            elif st == 'Current':
                max_prob = max_tr_prob * statistics.fmean([emit_p[st] [obs[t]], emit_p_bi[st].get((obs[t-1],obs[t])), emit_p_tri[st].get((obs[t-2],obs[t-1],obs[t]))]) 
            
            
            V[t] [st] = {"prob": max_prob, "prev": prev_st_selected}

    V = V[:-1]
    # for line in dptable(V):
       # print(line)

    optimal = []
    max_prob = 0.0
    best_st = None
    
    # Get most probable state and its backtrack
    for st, data in V[-1].items(): # V[-1] is the last dic. ex: {'New': {'prob': 6.983898154776834e-19, 'prev': 'Current'}, 'Current': {'prob': 1.5713770848247866e-18, 'prev': 'New'}
        if data["prob"] > max_prob: #6.983898154776834e-19 > 0.0 (in first step)
            max_prob = data["prob"] #update max prob to compare with second state in next iteration
            best_st = st
    optimal.append(best_st) #appending all best states
    previous = best_st #updating previous as best state to be used later


    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1): # start=len(V)-2, stop=-1, step=-1. Calculates from last step to first ###### NEW
        optimal.insert(0, V[t + 1] [previous] ["prev"])
        previous = V[t + 1] [previous] ["prev"]
    optimal = optimal[1:]
    
    # print ("The steps of states are " + " ".join(optimal) + " with highest probability of %s" % max_prob)
    return optimal, max_prob



def dptable(V):
    # Print a table of steps from dictionary
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state] ["prob"]) for v in V)



def inputs_hmm(uni, bi_fut, bi_past, tri_fut, tri_past):

    states = ['New', 'Current']
    start_p = {'New':0.9999,'Current':0.0001}

    Cur = 0.62 #+1e-70
    Ne = 1 - Cur

    trans_p = {'New': {'New':0.1,'Current':0.9}, 'Current': {'New': Ne,'Current':Cur}} #the higher current, more it appears. 0.6 too litle, 0.7 too much
    emit_uni = {}
    emit_bi = {}
    emit_tri = {}

    emit_uni['New'] = uni
    emit_uni['Current'] = uni 
    emit_bi['New'] = bi_fut 
    emit_bi['Current'] = bi_past 
    emit_tri['New'] = tri_fut 
    emit_tri['Current'] = tri_past 

    return states, start_p, trans_p, emit_uni, emit_bi, emit_tri


def all_dialog_hmm(unigram, uni, bi_fut, bi_past, tri_fut, tri_past):
    states, start_p, trans_p, emit_uni, emit_bi, emit_tri = inputs_hmm(uni, bi_fut, bi_past, tri_fut, tri_past)

    hmm_seq = []
    count = 0
    for dialog in unigram:
        try:
            seq, p = viterbi(dialog, states, start_p, trans_p, emit_uni, emit_bi, emit_tri)
            # hmm_seq.append([dialog[2:-1], seq, p]) #remove extra symbols from dialog (<s>, <ss>, <e>)
            hmm_seq.append([dialog[2:-2], seq[:-1], p]) #remove extra symbols from dialog (<s>, <ss>, <e>, <ee>)
        except Exception:
            count+=1
            pass

    return hmm_seq, count


def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv
    
    df = pd.read_csv(argv[1], index_col=0) #pd.read_csv('./data_TM2/processed/processed_utterances_sentence_DA_labeling.csv', index_col=0)

    model_name = 'HMM'

    a_file = open("./generated_files/input_hmm.pkl", "rb")
    input_hmm = pickle.load(a_file)
    unique_ids = input_hmm.get('unique_ids') # same as input_hmm['unique_ids']

    hmm_seq, count = all_dialog_hmm(input_hmm['unigram'], input_hmm['uni'], input_hmm['bi_fut'], input_hmm['bi_past'], input_hmm['tri_fut'], input_hmm['tri_past'])
    print('Count of dialogues that failed and went to exception: ' + str(count))

    i = 10
    inspection_df = manual_inspection(i, df, hmm_seq, unique_ids)

    sorted_d, seq_sizes, seq_to_plot = sorted_seq_and_counts_hmm(hmm_seq)
    plot_literal_count_per_seq(seq_sizes, model_name)
    plot_len_grouped = plot_grouped_per_len_seq(sorted_d, seq_to_plot, model_name)
    
    #save new results to file
    with open('./generated_files/sorted_dict_'+ model_name+'.pkl', 'wb') as fp:
        pickle.dump(sorted_d, fp)

    
    with open('./generated_files/hmm_results.pkl', 'wb') as f:
        pickle.dump(hmm_seq, f)
    
    print('done!')



if __name__ == '__main__':
    main()


#type in terminal:
#python3 HMM.py '../data_TM2/processed/processed_utterances_sentence_DA_labeling.csv'