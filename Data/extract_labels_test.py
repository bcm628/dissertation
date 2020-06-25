#extracts emotion labels from IEMOCAP data
#based on code by Demfier

import re
import os


data_path="/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)


for directory in os.listdir(data_path):
    if directory.startswith('Session'):
        eval_dir = '{0}/{1}/dialog/EmoEvaluation'.format(data_path, directory)
        eval_files = [file for file in os.listdir(eval_dir) if os.path.isfile(os.path.join(eval_dir, file))]
        for f in eval_files:
            with open(os.path.join(eval_dir, f)) as txt:
                txt_file = txt.read()
            info_lines = re.findall(info_line, txt_file)
            for line in infolines[1:10]:
                print(line.strip().split('\t'))
