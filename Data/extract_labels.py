#extracts emotion labels from IEMOCAP data
#based on code by Demfier

import re
import os
import pandas as pd


user_path="/afs/inf.ed.ac.uk/user/s19/s1940488/Desktop/IEMOCAP_labels"
data_path="/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
starts, ends, wav_files, emotions, vals, acts, doms = [], [], [], [], [], [], []

for directory in os.listdir(data_path):
    if directory.startswith('Session'):
        print("Looking through {}".format(directory))
        eval_dir = '{0}/{1}/dialog/EmoEvaluation'.format(data_path, directory)
        eval_files = [file for file in os.listdir(eval_dir) if os.path.isfile(os.path.join(eval_dir, file))]
        for f in eval_files:
            with open(os.path.join(eval_dir, f)) as txt:
                txt_file = txt.read()
            info_lines = re.findall(info_line, txt_file)
            for line in infolines[1:]:
                times, wav_file, emotion, dims = line.strip().split('\t')
                start, end = times[1:-1].split('-')
                start, end = float(start), float(end)
                valence, activation, dominance = dims[1:-1].split(',')
                valence, activation, dominance = float(valence), float(activation), float(dominance)
                starts.append(start)
                ends.append(end)
                wav_files.append(wav_file)
                emotions.append(emotion)
                vals.append(valence)
                acts.append(activation)
                doms.append(dominance)


df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'valence', 'activation', 'dominance'])

df_iemocap['start_time'] = starts
df_iemocap['end_time'] = ends
df_iemocap['wav_file'] = wav_files
df_iemocap['emotion'] = emotions
df_iemocap['val'] = vals
df_iemocap['act'] = acts
df_iemocap['dom'] = doms

df_iemocap.tail()


df_iemocap.to_csv('{}/IEMOCAP_labels.csv'.format(user_path), index=False)


# [18.6000 - 30.8976]     Ses01F_impro04_F002     fru     [2.5000, 3.5000, 3.0000]
# C-E1:   Frustration;    ()
# C-E2:   Frustration;    ()
# C-E4:   Neutral;        ()
# C-F1:   Neutral; Disgust;       ()
# A-E1:   val 3; act 3; dom  3;   ()
# A-E2:   val 2; act 4; dom  3;   ()
# A-F1:   val 2; act 3; dom  4;   (