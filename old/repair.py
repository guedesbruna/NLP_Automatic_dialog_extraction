import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import re
from nltk.tokenize import word_tokenize
import sklearn
from gensim.test.utils import get_tmpfile
from sklearn.metrics import pairwise
import os
import gensim
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data_TM2/synt_annotated_data.csv', index_col=0)


train, test = train_test_split(df,train_size=393180, shuffle=False) #split taking into account full dialogue. Not splitting in middle of dialogue
train, val = train_test_split(train,train_size=0.001, shuffle=False) #só pra testar coisas 0.001 funciona bem! 0.01 20s

def read_corpus(fname, tokens_only=False):
        for i, line in enumerate(fname): #### FOR MY CODE PUT (fname) ##### default is (f)
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = list(read_corpus(train['new_text']))
test_corpus = list(read_corpus(test['new_text'], tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=80)

#Build a vocabulary
model.build_vocab(train_corpus)

#train model
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#save model and load again
fname = get_tmpfile("my_doc2vec_model")

model.save(fname)
model = gensim.models.doc2vec.Doc2Vec.load(fname)


#selecting window and implementing cosine similarity
#questions to be answered:
#is this taking all dialogues or one? if only one loope through all
#define window width experimentally
#define threshold for cosine similarity


conversation1 = df.loc[df['conversation_id'] == 'dlg-00e32998-0b0f-47f1-a4f0-2ce90f1718d0']

sentences_window = {}
nrows = len(conversation1['new_text'])
sentences = conversation1['new_text']
rep_initiators = np.array(conversation1['DA_rep_init'])
window_width = 2

for k,sentence,rep_init in zip(range(nrows),sentences,rep_initiators):
    if rep_init== "repair_initiator":
        sentences_window[sentence] = sentences[k-window_width:k+window_width+1]

rep=pd.DataFrame(sentences_window)#.transpose()
repair_tok = list(read_corpus(rep.iloc[:,0], tokens_only=True))

inf_vecs = []
[inf_vecs.append(model.infer_vector(ut)) for ut in repair_tok]  

threshold = 0.2
paraphrasing = []

for e in range(len(inf_vecs)):
    counter = e+1
    while counter < len(inf_vecs):
        pair_sim = pairwise.cosine_similarity(inf_vecs[e].reshape(1, -1), inf_vecs[counter].reshape(1, -1))
        print(e, counter, pair_sim)
        if pair_sim >= threshold:
            print('YES', str(e), str(counter), str(pair_sim))
            paraphrasing.append((e,counter,pair_sim))
        counter +=1

paraphrasing