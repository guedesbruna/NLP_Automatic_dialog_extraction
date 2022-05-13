import pandas as pd
import numpy as np
import re
from collections import Counter
import statistics
import pickle
import sys
from statistics_and_plot import manual_inspection, sorted_seq_and_counts_hmm, plot_literal_count_per_seq, plot_grouped_per_len_seq


def viterbi(obs, states, start_p, trans_p, emit_p, emit_p_bi, emit_p_tri): 
    ''' #code based on : https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
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



def inputs_hmm(uni, bi_fut, bi_past, tri_fut, tri_past, cur):

    states = ['New', 'Current']
    start_p = {'New':0.9999,'Current':0.0001}

    # Cur = 0.62 #+1e-70
    Ne = 1 - cur

    trans_p = {'New': {'New':0.1,'Current':0.9}, 'Current': {'New': Ne,'Current':cur}} #the higher current, more it appears. 0.6 too litle, 0.7 too much
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


def all_dialog_hmm(unigram, uni, bi_fut, bi_past, tri_fut, tri_past, cur_st):
    states, start_p, trans_p, emit_uni, emit_bi, emit_tri = inputs_hmm(uni, bi_fut, bi_past, tri_fut, tri_past, cur_st)

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


#AUX FUNCT
def append_value(dict_obj, key, value):
    if key in dict_obj:
        # Key exist in dict. Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else: # As key is not in dict, add key-value pair
        dict_obj[key] = value

def sensitivity(unigram, uni, bi_fut, bi_past, tri_fut, tri_past):
    
    hmm_seq_range = {}

    for st_value in np.arange(0.1, 1.0, 0.05):

        hmm_seq, count = all_dialog_hmm(unigram, uni, bi_fut, bi_past, tri_fut, tri_past, st_value) 
        hmm_seq_range[str(round(st_value, 3))] = hmm_seq
        
        count_n_patterns_per_dial = [] #n of patterns per dialog. unique list, each element is number of patterns per dialog
        count_len_patterns = [] #len of patterns for full dataset. unique list, each element is len of one pattern, does not distinguish per dialog

        for i in range(len(hmm_seq)):
            indices = [index for index, element in enumerate(hmm_seq[i][1]) if element == 'New']
            count_n_patterns_per_dial.append(len(indices))
            leng = len(hmm_seq[i][1])
            for e in range(len(indices)-1):
                count_len_patterns.append(indices[e+1]-indices[e]) #subtract index of next new minus current new to see size of pattern

        
        append_value(hmm_seq_range, 'mean_n_patterns_per_dialog', np.mean(count_n_patterns_per_dial)) #literal mean
        append_value(hmm_seq_range, 'norm_mean_n_patterns_per_dialog', np.mean(count_n_patterns_per_dial)/leng) #normalized dividing by size of dialog
        append_value(hmm_seq_range, 'max_n_patters', np.max(count_n_patterns_per_dial))
        append_value(hmm_seq_range, 'min_n_patters', np.min(count_n_patterns_per_dial))
        append_value(hmm_seq_range, 'var_n_patters', np.var(count_n_patterns_per_dial))

        append_value(hmm_seq_range, 'mean_len_patterns', np.mean(count_len_patterns))
        append_value(hmm_seq_range, 'norm_mean_len_patterns', np.mean(count_len_patterns)/leng) #normalized dividing by size of dialog

    return hmm_seq_range


def main():

    model_name = 'HMM'

    a_file = open("./generated_files/input_hmm.pkl", "rb")
    input_hmm = pickle.load(a_file)
    unique_ids = input_hmm.get('unique_ids') # same as input_hmm['unique_ids']
    unigram = input_hmm.get('unigram')
    print(len(unigram))

    hmm_seq_range = sensitivity(unigram, input_hmm['uni'], input_hmm['bi_fut'], input_hmm['bi_past'], input_hmm['tri_fut'], input_hmm['tri_past'])
    
    #save new results to file
    with open('./generated_files/dict_sensitivity_change_prob_matrix_'+ model_name +'.pkl', 'wb') as fp:
        pickle.dump(hmm_seq_range, fp)

    print('done!')


if __name__ == '__main__':
    main()


#type in terminal:
#python3 HMM_sensitivity.py 
