#create dict with words from glove

import numpy as np

#based off code by Karishma Malkan:
# https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python

def loadGloVeModel(file):
    with open(file, "r") as f:
        glovemodel = {}
        for line in f:
            line = line.split()
            word = line[0]
            embedding = np.array([float(value) for value in line[1:]])
            glovemodel[word] = embedding
    print(len(glovemodel), "words loaded")
    return glovemodel

#GloVemodel = loadGloVeModel(GloVefile)
#GloVemodel["hello"]

if __name__ == "__main__":
    GloVefile = "Downloads/glove.840B.300d/glove.840B.300d"
    GloVemodel = loadGloVeModel(GloVefile)





