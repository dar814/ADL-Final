import pandas as pd
import copy

def ABCD_to_choice(l):
    if l=='A'or l=='1':
        return 0
    elif l=='B'or l=='2':
        return 1
    elif l=='C'or l=='3':
        return 2
    elif l=='D'or l=='4':
        return 3
    else:
        print(l,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def read_format_data(path):
    d=pd.read_json(path,lines=True)
    data=copy.deepcopy(d[d['question'].apply(lambda x:x['choices']).apply(len)==4])
    data=data.reset_index(drop=True)
    easy_train_question=pd.json_normalize(data['question'])['stem']
    easy_train_choice_0=pd.json_normalize(data['question'])['choices'].apply(lambda x:x[0]['text'])
    easy_train_choice_1=pd.json_normalize(data['question'])['choices'].apply(lambda x:x[1]['text'])
    easy_train_choice_2=pd.json_normalize(data['question'])['choices'].apply(lambda x:x[2]['text'])
    easy_train_choice_3=pd.json_normalize(data['question'])['choices'].apply(lambda x:x[3]['text'])
    data['question']=easy_train_question
    data['choice_0']=easy_train_choice_0
    data['choice_1']=easy_train_choice_1
    data['choice_2']=easy_train_choice_2
    data['choice_3']=easy_train_choice_3
    data['answerKey']=data['answerKey'].apply(lambda x: ABCD_to_choice(x)).astype(int)
    
    return data

pd.set_option('display.max_columns',7)

train_easy_data=read_format_data(path="ARC-Easy\\ARC-Easy-Train.jsonl")
valid_easy_data=read_format_data(path="ARC-Easy\\ARC-Easy-Dev.jsonl")
test_easy_data=read_format_data(path="ARC-Easy\\ARC-Easy-Test.jsonl")

train_hard_data=read_format_data(path="ARC-Challenge\\ARC-Challenge-Train.jsonl")
valid_hard_data=read_format_data(path="ARC-Challenge\\ARC-Challenge-Dev.jsonl")
test_hard_data=read_format_data(path="ARC-Challenge\\ARC-Challenge-Test.jsonl")

train_data=pd.concat([train_easy_data,train_hard_data],ignore_index=True)
valid_data=pd.concat([valid_easy_data,valid_hard_data],ignore_index=True)

Train_data=pd.concat([train_data,valid_data],ignore_index=True)
test_data=pd.concat([test_easy_data,test_hard_data],ignore_index=True)

Train_data.to_csv('train.csv',index=False)
test_data.to_csv('test.csv',index=False)