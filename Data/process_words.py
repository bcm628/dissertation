#process word-level data into a dataframe/csv
#aligns utterances with timing of video for utterance level alignment

import re
import os
import pandas as pd

data_path = "/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"
user_path = "/afs/inf.ed.ac.uk/user/s19/s1940488/dissertation/IEMOCAP_words"

# headers = ["filename", "start_frame", "end_frame", "start_time_rel", "end_time_rel", "start_time_vid", "end_time_vid", "utterance"]
# utterance_df = pd.DataFrame(columns=headers)

lab_list = []

for directory in os.listdir(data_path):
    if directory.startswith("Session"):
        lab_dir = os.path.join(data_path, directory, "/dialog/lab")
        for lab in lab_dir:
            for file in lab:
                with open(os.path.join(lab_dir, file)) as lab_file:
                    for line in lab_file:
                        line = line.readlines()
                        lab_list.append(line)

lab_df = pd.DataFrame(lab_list, columns=["start_time", "end_time", "filename"])
print(lab_df)



