#processes OpenSmile .txt files for IEMOCAP egemaps features
#creates a pandas data frame with feature labels as columns and wav file names as indexes

import pandas as pd
import os

#change as needed
data_path = "./IEMOCAP_egemaps"
txt_file = "Ses01F_impro01_F000.wav_egemaps.txt"

data_dict = {}
for file in os.listdir(data_path):
    with open(os.path.join(data_path, file)) as f:
        txt = f.readlines()
    feat_labels = txt[3:-5]
    feat_labels = [l.strip(' \n@attribute') for l in feat_labels]
    data = txt[-1].split(',')
    data = data[1:-1]
    data_dict[file] = data
feat_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=feat_labels)

feat_df.to_csv('IEMOCAP_egemaps.csv')

#test with one file:

# test_dict = {}
# with open(os.path.join(data_path, txt_file)) as f:
#     txt = f.readlines()
# feat_labels = txt[3:-5]
# feat_labels = [l.strip(' \n@attribute') for l in feat_labels]
# #print(feat_labels)
# data = txt[-1].split(',')
# data = data[1:-1]
# test_dict[txt_file] = data
# feat_df = pd.DataFrame.from_dict(test_dict, orient='index', columns=feat_labels)
# print(feat_df)
#
# feat_df.to_csv('test_features.csv')


#88 features
#print(len(data[1:-1]))
#print(len(txt[3:-5]))
