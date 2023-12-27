import pandas as pd
import copy

train_df = pd.read_parquet('cycic_multiplechoice_train.parquet')
train_MC=train_df[['run_id','correct_answer','question','answer_option0','answer_option1','answer_option2','answer_option3','answer_option4']]
train_MC.to_csv('train.csv',index=False)

valid_df = pd.read_parquet('cycic_multiplechoice_validation.parquet')
valid_MC=valid_df[['run_id','correct_answer','question','answer_option0','answer_option1','answer_option2','answer_option3','answer_option4']]
valid_MC.to_csv('test.csv',index=False)