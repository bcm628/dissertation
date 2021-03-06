#process word-level data into a dataframe/csv
#aligns utterances with timing of video for utterance level alignment

#utterances still need to be processed of punctuation etc

import re
import os
import pandas as pd

data_path = "/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"
user_path = "/afs/inf.ed.ac.uk/user/s19/s1940488/dissertation/emotion_recognition/dissertation/Data"


def make_utterance_df(data_path):
    line_list = []
    for directory in os.listdir(data_path):
        if directory.startswith("Session"):
            tr_dir = os.path.join(data_path, directory, "dialog/transcriptions")
            for file in os.listdir(tr_dir):
                with open(os.path.join(tr_dir, file)) as tr_txt:
                    for line in tr_txt:
                        if line.startswith(('F', 'M')):
                            continue
                        else:
                            line = re.split('[\s-]', line)
                            line[1] = line[1].strip("[")
                            line[1] = float(line[1])
                            line[2] = line[2].strip("]:")
                            line[2] = float(line[2])
                            line[3:] = [line[3:]]
                            line_list.append(line)
    utterance_df = pd.DataFrame(line_list, columns=["filename", "start_time", "end_time", "utterance"])
    print(utterance_df.to_string(index=False))
    return utterance_df


utterance_df = make_utterance_df(data_path)
utterance_df.to_csv('{}/utterance_info.csv'.format(user_path), index=False)










