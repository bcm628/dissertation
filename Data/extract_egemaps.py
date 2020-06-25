import os
from pathlib import Path

user_path="/afs/inf.ed.ac.uk/user/s19/s1940488/Desktop"
data_path="/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"
output_folder_name = "IEMOCAP_egemaps"

# opensmile_file = "cd {}/opensmile-2.3.0/".format(user_path)
# print(opensmile_file)
# os.system(opensmile_file)
# os.system('pwd')
# exit()

print("Looking in {} for dataset...".format(data_path))

for directory in os.listdir(data_path):
    if directory.startswith('Session'):
        print("Looking through {}".format(directory))
        for path in os.listdir('{0}/{1}/sentences/wav/'.format(data_path, directory)):
            print("Now looking through {}".format(path))
            for filename in os.listdir('{0}/{1}/sentences/wav/{2}/'.format(data_path, directory, path)):
                if not os.path.exists('{0}/{1}/{2}_egemaps.csv'.format(user_path, output_folder_name, filename)):
                    os.mknod('{0}/{1}/{2}_egemaps.csv'.format(user_path, output_folder_name, filename))
                    print("Created a file. Now going to extract egemaps features...")
                else:
                    print('file already exists')
                output_file = '{0}/{1}/{2}_egemaps.csv'.format(user_path, output_folder_name, filename)
                try:
                    opensmile_cmd = "./SMILExtract -C config/gemaps/eGeMAPSv01a.conf -instname IEMOCAP_egemaps -I {0}/{1}/sentences/wav/{2}/{3} -csvoutput {4}".format(data_path, directory, path, filename, output_file)
                    os.system(opensmile_cmd)
                    print("Moving on to next file")
                except:
                    print("oops... something went wrong. terminating.)
                    exit()
print('All done!')

#(MSG) [2] in SMILExtract : openSMILE starting!
# (MSG) [2] in SMILExtract : config file is: config/gemaps/eGeMAPSv01a.conf
# (MSG) [2] in cComponentManager : successfully registered 96 component types.
# (MSG) [2] in instance 'gemapsv01a_logSpectral' : logSpecFloor = -140.00  (specFloor = 1.000000e-14)
# (MSG) [2] in instance 'egemapsv01a_logSpectral_flux' : logSpecFloor = -140.00  (specFloor = 1.000000e-14)
# (MSG) [2] in instance 'lldsink' : No filename given, disabling this sink component.
# (MSG) [2] in instance 'lldhtksink' : No filename given, disabling this sink component.
# (MSG) [2] in instance 'lldarffsink' : No filename given, disabling this sink component.
# (MSG) [2] in instance 'csvsink' : No filename given, disabling this sink component.
# (MSG) [2] in instance 'htksink' : No filename given, disabling this sink component.
# (WARN) [1] in instance 'gemapsv01a_formantVoiced.reader' : Mismatch in input level buffer sizes (levelconf.nT). Level #0 has size 5 which is smaller than the max. input size of all input levels (150). This might cause the processing to hang unpredictably or cause incomplete processing.
# (WARN) [1] in instance 'gemapsv01a_logSpectralVoiced.reader' : Mismatch in input level buffer sizes (levelconf.nT). Level #0 has size 5 which is smaller than the max. input size of all input levels (150). This might cause the processing to hang unpredictably or cause incomplete processing.
# (WARN) [1] in instance 'gemapsv01a_logSpectralUnvoiced.reader' : Mismatch in input level buffer sizes (levelconf.nT). Level #0 has size 5 which is smaller than the max. input size of all input levels (150). This might cause the processing to hang unpredictably or cause incomplete processing.
# (WARN) [1] in instance 'egemapsv01a_logSpectralVoiced.reader' : Mismatch in input level buffer sizes (levelconf.nT). Level #0 has size 5 which is smaller than the max. input size of all input levels (150). This might cause the processing to hang unpredictably or cause incomplete processing.
# (WARN) [1] in instance 'egemapsv01a_logSpectralUnvoiced.reader' : Mismatch in input level buffer sizes (levelconf.nT). Level #0 has size 5 which is smaller than the max. input size of all input levels (150). This might cause the processing to hang unpredictably or cause incomplete processing.
# (MSG) [2] in cComponentManager : successfully finished createInstances
#                                  (77 component instances were finalised, 1 data memories were finalised)
# (MSG) [2] in cComponentManager : starting single thread processing loop
# (MSG) [2] in cComponentManager : Processing finished! System ran for 163 ticks.




