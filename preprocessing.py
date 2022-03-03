import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from nltk import sent_tokenize
import re
import sys


def process_raw(df): #, name):
    dialogs = []

    for e,i in enumerate(df['utterances']):
        for j in i:
            new_df=pd.json_normalize(j)
            new_df.insert(0, 'task', df['task'][e])
            new_df.insert(1, 'conversation_id', df['conversation_id'][e])
            new_df.insert(2, 'instruction_id', df['instruction_id'][e])
        
            dialogs.append(new_df)

    large_df = pd.concat(dialogs, ignore_index=True)
#   large_df.to_csv('./data_TM2/processed_utterances_'+name+'.csv')
    
    return large_df


def unique(df_list):
    large_frames = []

    for df_ in df_list:
        large_df = process_raw(df_)
        large_frames.append(large_df)

    all_tasks = pd.concat(large_frames, ignore_index=True)
    all_tasks.to_csv('./data_TM2/concatenated_tasks.csv')
    
    return all_tasks


def tokenize(df):
    tokenized = []
    for ut in df['text']:
        tokenized.append(sent_tokenize(ut))
        
    df['new_text'] = tokenized
    df = df.explode('new_text')
    df = df.reset_index(drop=True)

    return df


def main(argv=None):
    
    #For picking up commandline arguments
    if argv is None:
        argv = sys.argv

    df_t1 = pd.read_json('./data_TM2/flights.json')
    df_t2 = pd.read_json('./data_TM2/food-ordering.json')
    df_t3 = pd.read_json('./data_TM2/hotels.json')
    df_t4 = pd.read_json('./data_TM2/movies.json')
    df_t5 = pd.read_json('./data_TM2/music.json')
    df_t6 = pd.read_json('./data_TM2/restaurant-search.json')
    df_t7 = pd.read_json('./data_TM2/sports.json')

    # create extra column in the beginning for type of task before merging
    df_t1.insert(0, 'task', 'flights')
    df_t2.insert(0, 'task', 'food-ordering')
    df_t3.insert(0, 'task', 'hotels')
    df_t4.insert(0, 'task', 'movies')
    df_t5.insert(0, 'task', 'music')
    df_t6.insert(0, 'task', 'restaurant-search')
    df_t7.insert(0, 'task', 'sports')

    df_list = [df_t1, df_t2, df_t3, df_t4, df_t5, df_t6, df_t7]
    all_tasks = unique(df_list)
    final_df = tokenize(all_tasks)

    final_df.to_csv('./data_TM2/processed_utterances_all_tasks.csv')

    return final_df

if __name__ == '__main__':
    main()
