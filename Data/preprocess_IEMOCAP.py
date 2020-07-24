"""Processes iemocap_data.pkl from CMU and removes neutral dimension in order to use with CMU-MOSEI"""

import pickle
import os




def proccess_iemocap(data_path):
    dataset = pickle.load(open(os.path.join(data_path, "iemocap_data.pkl"), 'rb'))
    for split in dataset:
        dataset[split]['labels'] = dataset[split]['labels'][:,1:,:]

    return dataset





if __name__ == "__main__":
    data_path = './IEMOCAP_aligned'
    proccess_iemocap(data_path)
