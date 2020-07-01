#process word-level data into a dataframe/csv
#aligns utterances with timing of video for utterance level alignment

import re
import os
import pandas as pd

data_path = "/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"
user_path = "/afs/inf.ed.ac.uk/user/s19/s1940488/dissertation/IEMOCAP_words"

# headers = ["filename", "start_frame", "end_frame", "start_time_rel", "end_time_rel", "start_time_vid", "end_time_vid", "utterance"]
# utterance_df = pd.DataFrame(columns=headers)

def make_utterance_csv(data_path):
    lab_list = []
    for directory in os.listdir(data_path):
        if directory.startswith("Session"):
            tr_dir = os.path.join(data_path, directory, "dialog/transcriptions")
            for file in os.listdir(tr_dir):
                with open(os.path.join(tr_dir, file)) as tr_txt:
                    for line in tr_txt:
                        line = re.split('[\s-]', line)
                        line[1]=line[1].strip("[")
                        line[1] = float(line[1])
                        line[2]=line[2].strip("]:")
                        line[2] = float(line[2])
                        print(line)
                        exit()

make_utterance_csv(data_path)
    #                     lab_list.append(line)
    # lab_df = pd.DataFrame(lab_list, columns=["start_time", "end_time", "filename"])
    # lab_df["start_time"] = pd.to_numeric(lab_df["start_time"])
    # lab_df["end_time"] = pd.to_numeric(lab_df["end_time"])
    # return lab_df
    #print(lab_df.to_string(index=False))
#
# def create_utterance_csv(data_path, lab_df):
#     for directory in os.listdir(data_path):
#         if directory.startswith("Session"):
#             align_dir = os.path.join(data_path, directory, 'sentences/ForcedAlignment')
#             for folder in os.listdir(align_dir):
#                 for file in os.listdir(os.path.join(align_dir, folder)):
#                     if file.endswith(".wdseg"):
#                         with open(os.path.join(align_dir, folder, file) as align_file:
#                             align_file.readlines()
#                             align_list = align_file[1:-1]










