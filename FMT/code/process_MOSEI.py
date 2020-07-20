"""
Process tensors.pkl (CMU-MOSEI) from CMU Immortal server into a useable dictionary. Labels are made binary to compare performance
with IEMOCAP data. Only uses emotion labels from CMU-MOSEI and not sentiment labels.
This is imported as a module into MOSEI_new_dataset.py
"""



import numpy as np
import pickle


def format_mosei(data_path, pickle_out=False):
    """
    :arg data_path: path to tensors.pkl from  CMU immortal server
    :returns: nested dictionary with dataset folds as keys and a dictionary of labels and features as values;
    features are COVAREP acoustic features, glove word embeddings and Facet visual embeddings"""
    mosei = pickle.load(open(data_path, 'rb'), encoding='latin1')

    train_dict = {}
    valid_dict = {}
    test_dict = {}

    #access labels and make them binary predictors in order to use acc and F1 and compare with iemocap
    all_train = mosei[0][0]['All Labels']
    with np.nditer(all_train, op_flags=['readwrite']) as tr:
        for label in tr:
            if label > 0:
                label[...] = 1

    train_dict['language'] = mosei[0][0]['glove_vectors']
    train_dict['acoustic'] = mosei[0][0]['COAVAREP']
    train_dict['visual'] = mosei[0][0]['FACET 4.2']
    train_dict['labels'] = all_train[:,:,1:] #removes sentiment labels

    all_valid = mosei[0][1]['All Labels']
    with np.nditer(all_valid, op_flags=['readwrite']) as val:
        for label in val:
            if label > 0:
                label[...] = 1

    valid_dict['language'] = mosei[0][1]['glove_vectors']
    valid_dict['acoustic'] = mosei[0][1]['COAVAREP']
    valid_dict['visual'] = mosei[0][1]['FACET 4.2']
    valid_dict['labels'] = all_valid[:,:,1:]

    all_test = mosei[0][2]['All Labels']
    with np.nditer(all_test, op_flags=['readwrite']) as t:
        for label in t:
            if label > 0:
                label[...] = 1

    test_dict['language'] = mosei[0][2]['glove_vectors']
    test_dict['acoustic'] = mosei[0][2]['COAVAREP']
    test_dict['visual'] = mosei[0][2]['FACET 4.2']
    test_dict['labels'] = all_test[:,:,1:]

    mosei_new = {}

    mosei_new['train'] = train_dict
    mosei_new['valid'] = valid_dict
    mosei_new['test'] = test_dict

    if pickle_out == True:
        with open('mosei_dict.pickle', 'wb') as f:
            pickle.dump(mosei_new, f, protocol=pickle.HIGHEST_PROTOCOL)

    return mosei_new
