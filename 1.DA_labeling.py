import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import re
from nltk.tokenize import word_tokenize
import sys
import ast



def da_indicators(df):
    temp=df.speaker.fillna("0")
    df['speaker'] =  np.where(temp.str.contains('USER'), "U", 'A')
    # df['speaker'].unique()

    greeting = [' hi','hi.','hello','yo ', 'hey', 'How can I help you?'] 
    repair_initiator = ['Could you please repeat','Can you repeat','Could you say that again?','say again','what did you say','what else did you say?','I can\'t hear you.','repeat please','you said','sorry?','what do you mean?']
    request_summary = [' correct?','confirm']  
    confirmation = ['yes',' correct.',' correct ', 'that\'s it', 'indeed', 'yeah', 'that\'s right', 'you got it', 'yep', 'sure', 'okay', 'all right',  'sure', 'sounds good', 'super.', 'super!', 'alright', 'I don\'t mind', 'I can help you', 'sounds really good', 'got it.'] 
    sequence_closer = ['awesome','great','perfect','exactly','that\'s all','that is all', 'thanks', 'thank you', 'pleasure', 'excellent', 'okay', 'excellent', 'too bad', 'oh well', 'have a good day', 'enjoy', 'until next time', 'good day', 'good luck', 'bye', 'til next time']
    receipt = ['You are welcome', 'you\'re welcome']
    disconfirmation = [' no ', 'wrong', 'incorrect', 'not really']
    completion_check = ['is that all?','anything else I can help you with?', 'anything else?'] 
    partial_request = ['\?']
    detail_request = ['\?']
    hold_request = ['one moment, please',' hold on', 'hold', 'just a moment', 'let me check for you', 'let me see', 'one sec' ]
    grant_answer = ['.']
    temp2=df.new_text.fillna("0")

    #choose if i'll deal with whole iterances or them split

    df['DA_rep_init'] =  np.where(temp2.str.contains('|'.join(repair_initiator), case=False), df.speaker+"_repair_initiator", '')
    df['DA_greet'] =  np.where(temp2.str.contains('|'.join(greeting), case=False), df.speaker+"_greeting", '')
    df['DA_req_sum'] =  np.where(temp2.str.contains('|'.join(request_summary), case=False), df.speaker+"_request_summary", '')
    df['DA_conf'] =  np.where(temp2.str.contains('|'.join(confirmation), case=False), df.speaker+"_confirmation", '')
    df['DA_receipt'] =  np.where(temp2.str.contains('|'.join(receipt), case=False), df.speaker+"_receipt", '')
    df['DA_disconf'] =  np.where(temp2.str.contains('|'.join(disconfirmation), case=False), df.speaker+"_disconfirmation", '')
    df['DA_closer'] =  np.where(temp2.str.contains('|'.join(sequence_closer), case=False), df.speaker+"_sequence_closer", '')           
    df['DA_comp_check'] =  np.where(temp2.str.contains('|'.join(completion_check), case=False),  df.speaker+"_completion_check", '')
    df['DA_hold'] =  np.where(temp2.str.contains('|'.join(hold_request), case=False),  df.speaker+"_hold_request", '')
    df['DA_partial_req'] = np.where(((df['speaker'] == 'U') & temp2.str.contains('|'.join(partial_request), case=False)), df.speaker+'_partial_request', '')
    df['DA_detail_req'] = np.where(((df['speaker'] == 'A') & temp2.str.contains('|'.join(detail_request), case=False)), df.speaker+'_detail_request', '')
    df['DA_grant'] = np.where(((df['speaker'] == 'A') & (temp2.str.contains('|'.join(grant_answer), case=False) & (df['DA_partial_req'].shift(1) == 'U_partial_request') )), df.speaker+'_grant', '')                                     
    df['DA_answer'] = np.where(((df['speaker'] == 'U') & (temp2.str.contains('|'.join(grant_answer), case=False) & (df['DA_detail_req'].shift(1) == 'A_detail_request') )), df.speaker+'_answer', '')                                     

    df['all_DA'] = df[['DA_rep_init', 'DA_greet','DA_req_sum', 'DA_conf', 'DA_receipt', 'DA_disconf', 'DA_closer', 'DA_comp_check', 'DA_hold', 'DA_partial_req', 'DA_detail_req', 'DA_grant', 'DA_answer']].agg(' '.join, axis=1)

    df['all_DA'] = [word_tokenize(da) for da in df['all_DA']]

    df.loc[:, ['speaker', 'new_text', 'all_DA']].to_csv('analysis_multiple_DA_cases_3rditeration.csv')
    counts = df['all_DA'].value_counts()

    counts.to_csv('count_overlaping_DAs_3iteration.csv')
    # data = {'counts':counts}
    # df_counts = pd.DataFrame(data)

    labeled_ut_count = df["all_DA"].apply(lambda x: 1 if len(x) == 0 else 0).sum()
        
    print('number of utterances without label: '+ str(labeled_ut_count))
    print('total number of utterances in selected dataset: ' + str(len(df['all_DA'])))
    print('percentage of data synthetically labeled: ' + str(1- (labeled_ut_count/len(df['all_DA']))))
    
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

    df_final.to_csv('./data_TM2/processed_utterances_sentence_DA_labeling.csv')
    print('done!')
    return df_final

if __name__ == '__main__':
    main()

#type in the terminal
#python3 1.DA_labeling.py ./data_TM2/processed_utterances_all_tasks.csv