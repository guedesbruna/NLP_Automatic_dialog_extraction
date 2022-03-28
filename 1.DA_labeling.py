import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import re
from nltk.tokenize import word_tokenize
import sys



def da_indicators(df):
    greeting = [' hi','hi.','hello','yo ', 'hey', 'How can I help you?'] 
    # repair_initiator = ['sorry','repeat', 'understand', 'mean?', 'what?', 'example?', 'could you say that again?', 'say again', 'what did you say?', 'I can\'t hear you', 'I didn\'t listen', 'say it again']

    repair_initiator = ['Could you please repeat','Can you repeat','Could you say that again?','say again','what did you say','what else did you say?','I can\'t hear you.','repeat please','you said','sorry?','what do you mean?']
    # repair = ['I meant']
    request_summary = [' correct?','confirm']  
    confirmation = ['yes',' correct.',' correct ', 'that\'s it', 'indeed', 'yeah', 'that\'s right', 'you got it', 'yep', 'sure', 'okay', 'all right',  'sure', 'sounds good', 'super.', 'super!', 'alright', 'I don\'t mind', 'I can help you', 'sounds really good', 'got it.'] 
    sequence_closer = ['awesome','great','perfect','exactly','that\'s all','that is all', 'thanks', 'thank you', 'pleasure', 'excellent', 'okay', 'excellent', 'too bad', 'oh well', 'have a good day', 'enjoy', 'until next time', 'good day', 'good luck', 'bye', 'til next time']
    receipt = ['You are welcome', 'you\'re welcome']
    disconfirmation = [' no ', 'wrong', 'incorrect', 'not really']
    #NEW
    completion_check = ['is that all?','anything else I can help you with?', 'anything else?'] 
    partial_request = ['\?']
    detail_request = ['\?']
    hold_request = ['one moment, please',' hold on', 'hold', 'just a moment', 'let me check for you', 'let me see', 'one sec' ]
    grant_answer = ['.']
    temp2=df.new_text.fillna("0")

    #choose if i'll deal with whole iterances or them split

    df['DA_rep_init'] =  np.where(temp2.str.contains('|'.join(repair_initiator), case=False), "repair_initiator", '')
    # df['DA_rep'] =  np.where(temp2.str.contains('|'.join(repair), case=False), "repair", '')
    df['DA_greet'] =  np.where(temp2.str.contains('|'.join(greeting), case=False), "greeting", '')
    df['DA_req_sum'] =  np.where(temp2.str.contains('|'.join(request_summary), case=False), "request_summary", '')
    df['DA_conf'] =  np.where(temp2.str.contains('|'.join(confirmation), case=False), "confirmation", '')
    df['DA_receipt'] =  np.where(temp2.str.contains('|'.join(receipt), case=False), "receipt", '')
    df['DA_disconf'] =  np.where(temp2.str.contains('|'.join(disconfirmation), case=False), "disconfirmation", '')
    df['DA_closer'] =  np.where(temp2.str.contains('|'.join(sequence_closer), case=False), "sequence_closer", '')           
    df['DA_comp_check'] =  np.where(temp2.str.contains('|'.join(completion_check), case=False),  "completion_check", '')
    df['DA_hold'] =  np.where(temp2.str.contains('|'.join(hold_request), case=False),  "hold_request", '')
    df['DA_partial_req'] = np.where(((df['speaker'] == 'USER') & temp2.str.contains('|'.join(partial_request), case=False)), 'partial_request', '')
    df['DA_detail_req'] = np.where(((df['speaker'] == 'ASSISTANT') & temp2.str.contains('|'.join(detail_request), case=False)), 'detail_request', '')
    df['DA_grant'] = np.where(((df['speaker'] == 'ASSISTANT') & (temp2.str.contains('|'.join(grant_answer), case=False) & (df['DA_partial_req'].shift(1) == 'partial_request') )), 'grant', '')                                     
    df['DA_answer'] = np.where(((df['speaker'] == 'USER') & (temp2.str.contains('|'.join(grant_answer), case=False) & (df['DA_detail_req'].shift(1) == 'detail_request') )), 'answer', '')                                     

    df['all_DA'] = df[['DA_rep_init', 'DA_greet','DA_req_sum', 'DA_conf', 'DA_receipt', 'DA_disconf', 'DA_closer', 'DA_comp_check', 'DA_hold', 'DA_partial_req', 'DA_detail_req', 'DA_grant', 'DA_answer']].agg(' '.join, axis=1)

    df['all_DA'] = [word_tokenize(da) for da in df['all_DA']]

    df.loc[:, ['speaker', 'new_text', 'all_DA']].to_csv('analysis_multiple_DA_cases_3rditeration.csv')
    # df.loc[:, ['speaker', 'new_text', 'all_DA']]
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

    # print(df_final.head())
    df_final.to_csv('./data_TM2/processed_utterances_sentence_DA_labeling.csv')
    print('done!')
    return df_final

if __name__ == '__main__':
    main()

#type in the terminal
#python3 DA_labeling.py ./data_TM2/processed_utterances_all_tasks.csv