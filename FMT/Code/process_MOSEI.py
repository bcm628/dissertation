"""
Process tensors.pkl (CMU-MOSEI) from CMU Immortal server into a useable dictionary. Labels are made binary to compare performance
with IEMOCAP data. Only uses emotion labels from CMU-MOSEI and not sentiment labels.
This is imported as a module into MOSEI_new_dataset.py
"""



import numpy as np
import pickle


def format_mosei(data_path, pickle_out=False, three_dim=False):
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
    train_dict['labels'] = np.transpose(train_dict['labels'], (0,2,1))

    all_valid = mosei[0][1]['All Labels']
    with np.nditer(all_valid, op_flags=['readwrite']) as val:
        for label in val:
            if label > 0:
                label[...] = 1

    valid_dict['language'] = mosei[0][1]['glove_vectors']
    valid_dict['acoustic'] = mosei[0][1]['COAVAREP']
    valid_dict['visual'] = mosei[0][1]['FACET 4.2']
    valid_dict['labels'] = all_valid[:,:,1:]
    valid_dict['labels'] = np.transpose(valid_dict['labels'], (0,2,1))

    all_test = mosei[0][2]['All Labels']
    with np.nditer(all_test, op_flags=['readwrite']) as t:
        for label in t:
            if label > 0:
                label[...] = 1

    test_dict['language'] = mosei[0][2]['glove_vectors']
    test_dict['acoustic'] = mosei[0][2]['COAVAREP']
    test_dict['visual'] = mosei[0][2]['FACET 4.2']
    test_dict['labels'] = all_test[:,:,1:]
    test_dict['labels'] = np.transpose(test_dict['labels'], (0,2,1))

    mosei_new = {}

    mosei_new['train'] = train_dict
    mosei_new['valid'] = valid_dict
    mosei_new['test'] = test_dict

#TODO: remove samples with no labels
    if three_dim:
        for split in mosei_new:
            mosei_new[split]['labels'] = mosei_new[split]['labels'][:, 3:, :]

        final_mosei = {}
        splits = ['train', 'valid', 'test']
        feats = ['labels', 'language', 'acoustic', 'visual']

        for split in splits:
            final_mosei[split] = {}
            for feat in feats:
                final_mosei[split][feat] = []

        for split in mosei_new:
            for i, x in enumerate(mosei_new[split]['labels']):
                if np.sum(x) != 0:
                    for key in final_mosei[split]:
                        final_mosei[split][key].append(mosei_new[split][key][i])
            for key in final_mosei[split]:
                final_mosei[split][key] = np.array(final_mosei[split][key])
        mosei_new = final_mosei

    if pickle_out:
        with open('mosei_dict2.pickle', 'wb') as f:
            pickle.dump(mosei_new, f)

    return mosei_new


if __name__ == "__main__":
    format_mosei('C:/Users/bcmye/PycharmProjects/CMU-MultimodalSDK/data/MOSEI_aligned/tensors.pkl', pickle_out=True, three_dim=True)