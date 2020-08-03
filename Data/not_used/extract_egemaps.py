#change back to csv, add argument -instname with .wav file name

import os

user_path="/afs/inf.ed.ac.uk/user/s19/s1940488/dissertation"
data_path="/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"
output_folder_name = "IEMOCAP_egemaps"

print("Looking in {} for dataset...".format(data_path))

for directory in os.listdir(data_path):
    if directory.startswith('Session'):
        print("Looking through {}".format(directory))
        for path in os.listdir('{0}/{1}/sentences/wav/'.format(data_path, directory)):
            print("Now looking through {}".format(path))
            for filename in os.listdir('{0}/{1}/sentences/wav/{2}/'.format(data_path, directory, path)):
                output_file = '{0}/{1}/{2}_egemaps.txt'.format(user_path, output_folder_name, filename)
                try:
                    opensmile_cmd = "./SMILExtract -C config/gemaps/eGeMAPSv01a.conf -O {4} -I {0}/{1}/sentences/wav/{2}/{3}".format(data_path, directory, path, filename, output_file)
                    os.system(opensmile_cmd)
                    print("Moving on to next file")
                except:
                    print("oops... something went wrong. terminating.")
                    exit()
print('All done!')




