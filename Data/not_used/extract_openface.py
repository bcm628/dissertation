import os

data_path = "/afs/inf.ed.ac.uk/group/corpora/large2/IEMOCAP_full_release"
user_path = "afs/inf.ed.ac.uk/user/s19/s1940488/dissertation"
output_folder = "IEMOCAP_openface"

for directory in os.listdir(data_path):
    if directory.startswith("Session"):
        print("Looking through {} directory for videos".format(directory))
        for vid in os.listdir(os.path.join(data_path, directory, "dialog/avi/DivX/")):
            if vid.startswith("Ses"):
                openface_cmd = "FeatureExtraction.exe -f \"{0}\" -out_dir {1}".format(vid, os.path.join(user_path, output_folder))
                os.system(openface_cmd)
                print("Moving on to next video")
