import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import re
import sys


# df = pd.read_csv('./data_TM2/processed_utterances_all_tasks_TESTE2.csv', index_col=0)

# def sentence_ut(df):
#     # #essa parte é de antes porem com split incluindo simbolo do split
#     new_text = []
#     for ut in df['text']:
#         new_ut = re.split('(\.)', ut)
#         new_text.append(new_ut)
#     df['new_text'] = new_text

#     #to keep delimiter in sentence
#     #not very efficient way of doing it, apparently a better way using regex symbols

#     new_lista = []
#     for row in df['new_text']:
#     #     print(row)
#         for e, word in enumerate(row):
#             if word == '.':
#                 new_lista.append(row[e-1]+row[e])
#                 new_lista.remove(row[e-1])
#             else:
#                 new_lista.append(word)
    
#     df['new_lista'] = new_lista
#     df = df.explode('new_lista')
#     df = df.reset_index(drop=True)

#     return df


def da_indicators(df):
    greeting = [' hi','hi.','hello','bye'] 
    repair_initiator = ['sorry','repeat', 'understand', 'mean'] 
    #repair is what comes after the repair initiator
    request_summary = ['correct?','confirm'] #when you ask an information to be confirmed
    confirmation = ['yes','correct.','correct ', 'that\'s it', 'indeed'] #confirmation é a resposta que confirma: ex yes | correct | that's it
    sentence_closer = ['awesome','great','perfect','exactly','that\'s all', 'thanks', 'thank you'] 
    question = ['\?']
    receipt = ['You are welcome', 'you\'re welcome']

    #choose if i'll deal with whole iterances or them split

    temp=df.text.fillna("0")
    df['DA_indicators'] =  np.where(temp.str.contains('|'.join(repair_initiator), case=False), "repair_initiator",
                                np.where(temp.str.contains('|'.join(greeting), case=False), "greeting",
                                np.where(temp.str.contains('|'.join(confirmation), case=False),  "confirmation",
                                np.where(temp.str.contains('|'.join(question), case=False),  "question",
                                np.where(temp.str.contains('|'.join(sentence_closer), case=False), "sentence_closer", "x")))))

    temp2=df.new_text.fillna("0")
    df['DA_indicators_sent'] =  np.where(temp2.str.contains('|'.join(repair_initiator), case=False), "repair_initiator",
                                np.where(temp2.str.contains('|'.join(greeting), case=False), "greeting",
                                np.where(temp2.str.contains('|'.join(confirmation), case=False),  "confirmation",
                                np.where(temp2.str.contains('|'.join(question), case=False),  "question",
                                np.where(temp2.str.contains('|'.join(sentence_closer), case=False), "sentence_closer", "x")))))

    return df

def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv

    #set path to file    
    p = argv[1]
    df = pd.read_csv(p, index_col=0)

    # df_sent = sentence_ut(df)
    df_final = da_indicators(df)

    # print(df_final.head())
    df_final.to_csv('./data_TM2/processed_utterances_sentence_DA_labeling.csv')
    return df_final

if __name__ == '__main__':
    main()

#type in the terminal
#python3 DA_labeling.py ./data_TM2/processed_utterances_all_tasks.csv