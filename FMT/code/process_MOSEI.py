#code to process tensors.pkl from CMU Immortal server

#format of tensors.pkl:
#[[{feat:array, feat:array, ...}, {feat:array, feat: array, ...}, {feat:array, feat:array, ...}],["train","valid", "test"]]
#e.g. tensors[0][0]['glove_vectors'] = training glove_vectors

#expected format: {'test': {'labels_sent': ..., 'language': ...,, etc}, 'train': {'labels':..., 'language':..., etc}, 'valid': {etc}}
#I think...

#Iemocap features are Covarep, glove and Facet
import pickle

def format_mosei(data_path):
    mosei = pickle.load(open(data_path, 'rb'), encoding='latin1')

    train_dict = {}
    valid_dict = {}
    test_dict = {}

    train_dict['language'] = mosei[0][0]['glove_vectors']
    train_dict['acoustic'] = mosei[0][0]['COAVAREP']
    train_dict['visual'] = mosei[0][0]['FACET 4.2']
    train_dict['labels_sent'] = mosei[0][0]['All Labels']

    valid_dict['language'] = mosei[0][1]['glove_vectors']
    valid_dict['acoustic'] = mosei[0][1]['COAVAREP']
    valid_dict['visual'] = mosei[0][1]['FACET 4.2']
    valid_dict['labels_sent'] = mosei[0][1]['All Labels']

    test_dict['language'] = mosei[0][2]['glove_vectors']
    test_dict['acoustic'] = mosei[0][2]['COAVAREP']
    test_dict['visual'] = mosei[0][2]['FACET 4.2']
    test_dict['labels_sent'] = mosei[0][2]['All Labels']

    mosei_new = {}

    mosei_new['train'] = train_dict
    mosei_new['valid'] = valid_dict
    mosei_new['test'] = test_dict

    return mosei_new
