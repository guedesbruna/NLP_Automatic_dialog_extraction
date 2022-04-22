# ## N-grams from Jurafsky

import pandas as pd
import numpy as np
import re
from collections import Counter
import statistics
import sys


# df = pd.read_csv('./data_TM2/processed/processed_utterances_sentence_DA_labeling.csv', index_col=0)


import ast

def add_special_tok(df):
    '''Correct list fromat and add special tokens <UNK>, <s> and <ss> for initial padding and <e> and <ee> for padding the end of sentence.
    Outputs the updated df, the list of all DA and unique dialog ids.'''

    all_DA = []
    for row in range(len(df['all_DA'])):
        inst = ast.literal_eval(df['all_DA'][row])
        inst = [i.strip("[]") for i in inst]
        all_DA.append(inst)
        
    df['all_DA'] = all_DA


    df.drop(df.iloc[:, 8:-1], inplace = True, axis = 1)
    df = df.explode('all_DA')
    df['all_DA'] = df['all_DA'].fillna('<UNK>')
    df['all_DA'].head()


    #solution for all dialogues. After This data will be prepared to serve as input
    unique_ids = df['conversation_id'].unique()

    start = '<s>'
    start_dou = '<ss>'
    end_dou = '<ee>'
    end ='<e>'
    full_DA = []  

    for dialog in unique_ids:
        dialog_DA = []   
        temp = df.loc[df['conversation_id'] == dialog]
        for x in range(len(temp)):
            dialog_DA.append(temp['all_DA'].iloc[x])

        #insert begin and end tokens for each dialog. This is for bigram only. for tri would need two symbols in the begin and 2 in the end
        dialog_DA.insert(0, start)
        dialog_DA.insert(1, start_dou)
        dialog_DA.append(end_dou)
        dialog_DA.append(end)
        full_DA.append(dialog_DA)

    return df, full_DA, unique_ids




###AUX FUNCT
def flatten(lista):
    '''Receives a list of lists and flattens to a single list
    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists'''

    return [item for sublist in lista for item in sublist]





import nltk
from nltk.util import ngrams

# Modeling phase
def ngrams_lm(full_DA):
    '''Modeling ngrams from unigram to fourgram.
    from this tutorial: https://medium.com/swlh/language-modelling-with-nltk-20eac7e70853#b9bf
    Outputs the unigram list, a dict with all ngrams(from 1 to 4) and a dict of probabilities of format key=ngram, value=prob'''

    unigram=[]
    bigram=[]
    trigram=[]
    fourgram=[]
    tokenized_sent = flatten(unigram)

    #unigram, bigram, trigram, and fourgram models are created
    for sequence in full_DA:
        unigram.append(sequence)

    bigram.extend(list(ngrams(sequence, 2)))  
    trigram.extend(list(ngrams(sequence, 3)))
    fourgram.extend(list(ngrams(sequence, 4)))

    # ngrams step to calculate probs:
                
    ngrams_all = {1:[], 2:[], 3:[], 4:[]} #from unigram to fourgram in this case
    for i in range(4):
        for each in unigram:
            for j in ngrams(each, i+1):
                ngrams_all[i+1].append(j);
    ngrams_voc = {1:set([]), 2:set([]), 3:set([]), 4:set([])} #set() method is used to convert any of the iterable to sequence of iterable elements with distinct elements

    for i in range(4):
        for gram in ngrams_all[i+1]:
            if gram not in ngrams_voc[i+1]:
                ngrams_voc[i+1].add(gram)
    total_ngrams = {1:-1, 2:-1, 3:-1, 4:-1}
    total_voc = {1:-1, 2:-1, 3:-1, 4:-1}

    for i in range(4):
        total_ngrams[i+1] = len(ngrams_all[i+1])
        total_voc[i+1] = len(ngrams_voc[i+1])                       
        
    ngrams_prob = {1:[], 2:[], 3:[], 4:[]}

    for i in range(4):
        for ngram in ngrams_voc[i+1]:
            tlist = [ngram]
            tlist.append(ngrams_all[i+1].count(ngram))
            ngrams_prob[i+1].append(tlist)
        
    for i in range(4):
        for ngram in ngrams_prob[i+1]:
            ngram[-1] = (ngram[-1])/(total_ngrams[i+1]+total_voc[i+1])             


    #Prints top 10 unigram, bigram, trigram, fourgram
    # print("Most common n-grams without stopword removal and with add-1 smoothing: \n")
    for i in range(4):
        ngrams_prob[i+1] = sorted(ngrams_prob[i+1], key = lambda x:x[1], reverse = True)
        

    return unigram, ngrams_all, ngrams_prob, tokenized_sent #unigram, bigram, trigram, fourgram




def ngram_prediction(tokenized_sent, ngrams_prob):
    ''' Calculates probability of unseen sequences
    Outputs a dict with predictions and a the ngram probability dict updated with probabilities of unseen sequences.'''

    ngram = {1:[], 2:[], 3:[]}#to store n-grams formed from the strings

    for i in range(1, 4):
        ngram[i] = list(ngrams(tokenized_sent, i))[-1]
    
    print("String: ", ngram)
    
    for j in range(4):
        ngrams_prob[j+1] = sorted(ngrams_prob[j+1], key = lambda x:x[1], reverse = True)
    
    pred = {1:[], 2:[], 3:[]}
    for k in range(3):
        count = 0
        for each in ngrams_prob[k+2]:
            if each[0][:-1] == ngram[k+1]:#to find predictions based on highest probability of n-grams                   
                count +=1
                pred[k+1].append(each[0][-1])
                if count ==5:
                    break
        if count<5:
            while(count!=5):
                pred[k+1].append("NA")#if no word prediction is found, replace with NOT FOUND
                count +=1
                
    return pred, ngrams_prob


import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary


def perplexity(unigram):
    ''' Perplexity is An intrinsic evaluation metric is one that measures the quality of a model independent of any application.
    The perplexity of a language model on a test set is the inverse probability of the test set, normalized by the number of words. 
    Thus the higher the conditional probability of the word sequence, the lower the perplexity, and maximizing the perplexity is equivalent
    to maximizing the test set probability according to the language model. 
    Here for unlabeled data, we calculate perprexity in the whole dataset using as input the unigrams.
    It outputs a file with the perplexity for each 
    https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954#55043954?newreg=b97aa34187184c90988f9a75e51898c2 '''

    unigram_train = unigram[0:17000]

    n = 2
    train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="<e>") for t in unigram_train]
    words = [word for sent in unigram_train for word in sent]
    words.extend(["<s>", "<e>"])
    padded_vocab = Vocabulary(words)
    model = MLE(n)
    model.fit(train_data, padded_vocab)

    unigram_test = unigram[17000:] # if you want to test just small sample
    print(len(unigram_test))

    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="<e>") for t in unigram_test]

    #for each bigram MLE estimate (tuples estimate). maximizing a likelihood function so that, under the assumed statistical model, the observed data is most probable.
    #if you don't want to see MLE per tuple comment print line
    for test in test_data:
        print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test]) # will generate one estimator per input. Here each input contains all DA labels for one dialog sequence.
        # pass

    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="<e>") for t in unigram_test]
    
    with open('perplexity.txt', 'w') as f:
        f.write('Perplexity for each dialog tested: \n')
        for i, test in enumerate(test_data):
    #         f.write("PP({0}):{1}".format(unigram_test[i], model.perplexity(test))) #to see whole dialog instead of index uncomment this line and comment line below
            f.write("PP({0}):{1}".format(i, model.perplexity(test)))
            f.write('\n')
    
###AUX FUNCT
def past_ngrams(da_tuples, da_triples, bi, tri):
    '''To include all possible pairs, even if they have a zero prob. In that case assign a very low prob.'''

    # ## Correct past bi/trigrams (including all possible examples, now it has only non zero examples)

    # bi_past same as bi, but if there is no match prob will be set to 1e-10 
    #same has to be done for tri_past and tri

    bi_past = {} #use da_tuples list ad basis, since contains all possibilities
    tri_past = {} #use da_triples list ad basis, since contains all possibilities
            
    for dupla in da_tuples: #list containing all possible tuples of bigram
        bi_past[dupla] = 1e-10 
        for k in bi.keys():
            if k == dupla:
                bi_past[dupla] = bi.get(k)
                
    for tripla in da_triples: #list containing all possible tuples of bigram
        tri_past[tripla] = 1e-10 
        for ka in tri.keys():
            if ka == tripla:
                tri_past[tripla] = tri.get(ka)

    # #sanity check
    # print(len(bi_past) == len(da_u)**2, len(bi_past))
    # print(len(tri_past) == len(da_u)**3, len(tri_past))

    return bi_past, tri_past

###AUX FUNCT
def future_ngrams(da_u,bi_past, tri_past, uni):
    ''' Future bi/trigrams.
    Calculate for t, the p(t|t+1) for bigrams and p(t|t+1)*p(t+1|t+2)'''

    bi_fut = {}
    for A in da_u:
        for B in da_u:
            bi_fut[(A,B)] = bi_past.get((A,B))*uni.get(A)/uni.get(B)
            
    tri_fut = {}
    for e, A in enumerate(da_u):
        for f, B in enumerate(da_u):
            for g, C in enumerate(da_u):
                tri_fut[(A,B,C)] = tri_past.get((A,B,C))*bi_past.get((A,C))/bi_past.get((B,C))
    
    return bi_fut, tri_fut

###AUX FUNCT
def all_possible_ngrams(da_u):
    '''create list of all possible unique bigrams and trigrams. order matters
    count occurrences for ngrams.
    Output format:  bi: {('A_confirmation', 'A_sequence_closer'): 0.042966175117006435, ...
    tri: {('U_answer', 'A_confirmation', 'A_sequence_closer'): 0.018695486830967774, ...'''

    #create list of all possible unique bigrams and trigrams. order matters
    da_tuples = []

    for fir in range(len(da_u)):
        for sec in range(len(da_u)):
            da_tuples.append((da_u[fir], da_u[sec]))


    da_triples = []
    for fir in range(len(da_u)):
        for sec in range(len(da_u)):
            for thir in range(len(da_u)):
                da_triples.append((da_u[fir],da_u[sec], da_u[thir]))


    # #sanity check bi and tri
    # print(len(da_tuples) == len(da_u)*len(da_u), len(da_tuples))
    # print(len(da_triples) == len(da_u)**3, len(da_triples))


    ### I THINK THE BELOW IS NOT USED LATER ON
    # #count occurrences for bigrams
    # bi_next = {}

    # for row in range(len(df)-1):
    #     k = (df['all_DA'][row], df['shifted_DA_bi'][row])            
    #     bi_next[k] = bi_next[k] + 1 if k in bi_next else 1
        

    # #count occurrences for trigrams
    # tri_next = {}

    # for row in range(len(df)-1):
    #     k = (df['all_DA'][row], df['shifted_DA_bi'][row], df['shifted_DA_tri'][row])            
    #     tri_next[k] = tri_next[k] + 1 if k in tri_next else 1
    
    return(da_tuples, da_triples)

def emission_probs(df, ngrams_prob):

    uni = {}
    bi = {}
    tri = {}
    four = {}

    for key, value in ngrams_prob.items():
        for lista in value:            
            if key ==1:
                lista[0] = str(lista[0])
                lista[0] = re.sub(r"['(),]", "", lista[0])
                uni[lista[0]] = lista[1]
            if key ==2:
                bi[lista[0]] = lista[1]
            if key ==3:
                tri[lista[0]] = lista[1]
            if key ==4:
                four[lista[0]] = lista[1]


    # ### Make emission probabilities for hidden state New

    # - count +=1 for that combination of 2 labels every time one label is followed by next. 
    # - I want to calculate for t, the p(t|t+1) for bigrams and p(t|t+1)*p(t+1|t+2)
    # - format: 
    #     - bi: {('A_confirmation', 'A_sequence_closer'): 0.042966175117006435,
    #     - tri: {('U_answer', 'A_confirmation', 'A_sequence_closer'): 0.018695486830967774,


    # unique da counting special symbols
    da_u = [] #uni.keys()
    [da_u.append(k) for k in uni.keys()]

    #create shifted DA for future bigram
    df['shifted_DA_bi'] = df['all_DA'].shift(-1)
    #create shifted DA for future trigram
    df['shifted_DA_tri'] = df['all_DA'].shift(-2)


    #count occurrences of each DA
    df = df.reset_index(drop=True)
    count_DA = df['all_DA'].value_counts().reset_index(name='Counts')
    count_DA.iloc[0, 0]
    count_da = {}
    for e in range(len(count_DA)):
        count_da[count_DA.iloc[e, 0]] = count_DA.iloc[e, 1]


    # #create list of all possible unique bigrams and trigrams. order matters
    # da_tuples = []

    # for fir in range(len(da_u)):
    #     for sec in range(len(da_u)):
    #         da_tuples.append((da_u[fir], da_u[sec]))


    # da_triples = []
    # for fir in range(len(da_u)):
    #     for sec in range(len(da_u)):
    #         for thir in range(len(da_u)):
    #             da_triples.append((da_u[fir],da_u[sec], da_u[thir]))


    # # #sanity check bi and tri
    # # print(len(da_tuples) == len(da_u)*len(da_u), len(da_tuples))
    # # print(len(da_triples) == len(da_u)**3, len(da_triples))



    # #count occurrences for bigrams
    # bi_next = {}

    # for row in range(len(df)-1):
    #     k = (df['all_DA'][row], df['shifted_DA_bi'][row])            
    #     bi_next[k] = bi_next[k] + 1 if k in bi_next else 1
        

    # #count occurrences for trigrams
    # tri_next = {}

    # for row in range(len(df)-1):
    #     k = (df['all_DA'][row], df['shifted_DA_bi'][row], df['shifted_DA_tri'][row])            
    #     tri_next[k] = tri_next[k] + 1 if k in tri_next else 1

        

    # # ## Correct past bi/trigrams (including all possible examples, now it has only non zero examples)

    # # bi_past same as bi, but if there is no match prob will be set to 0.00000000000001 or sth very low
    # #same has to be done for tri_past and tri

    # bi_past = {} #use da_tuples list ad basis, since contains all possibilities
    # tri_past = {} #use da_triples list ad basis, since contains all possibilities
            
    # for dupla in da_tuples: #list containing all possible tuples of bigram
    #     bi_past[dupla] = 0.0000000001 
    #     for k in bi.keys():
    #         if k == dupla:
    #             bi_past[dupla] = bi.get(k)
                
    # for tripla in da_triples: #list containing all possible tuples of bigram
    #     tri_past[tripla] = 0.0000000001 
    #     for ka in tri.keys():
    #         if ka == tripla:
    #             tri_past[tripla] = tri.get(ka)

    # # #sanity check
    # # print(len(bi_past) == len(da_u)**2, len(bi_past))
    # # print(len(tri_past) == len(da_u)**3, len(tri_past))


    # # ## Future bi/trigrams
    # bi_fut = {}
    # for A in da_u:
    #     for B in da_u:
    #         bi_fut[(A,B)] = bi_past.get((A,B))*uni.get(A)/uni.get(B)
            
    # tri_fut = {}
    # for e, A in enumerate(da_u):
    #     for f, B in enumerate(da_u):
    #         for g, C in enumerate(da_u):
    #             tri_fut[(A,B,C)] = tri_past.get((A,B,C))*bi_past.get((A,C))/bi_past.get((B,C))

    # # #sanity check
    # # print(len(bi_fut) == len(bi_past))
    # # print(len(tri_fut) == len(tri_past))

    da_tuples, da_triples = all_possible_ngrams(da_u)
    bi_past, tri_past = past_ngrams(da_tuples, da_triples, bi, tri)
    bi_fut, tri_fut = future_ngrams(da_u,bi_past, tri_past, uni)

    return df, bi_past, tri_past, bi_fut, tri_fut



def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv

    df = pd.read_csv(argv[1], index_col=0) #pd.read_csv('./data_TM2/processed/processed_utterances_sentence_DA_labeling.csv', index_col=0)

    df, full_DA, unique_ids = add_special_tok(df)
    unigram, ngrams_all, ngrams_prob, tokenized_sent = ngrams_lm(full_DA)
    # pred, ngrams_prob = ngram_prediction(tokenized_sent, ngrams_prob)
    perplexity(unigram)
    df, bi_past, tri_past, bi_fut, tri_fut = emission_probs(df, ngrams_prob)


    print('done!')

if __name__ == '__main__':
    main()


#type in terminal:
#python3 ngrams.py './data_TM2/processed/processed_utterances_sentence_DA_labeling.csv'