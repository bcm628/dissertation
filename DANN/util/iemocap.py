import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F

#from consts import global_consts as gc
from DANN import params

class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.feat = np.empty(0)
        self.y = np.empty(0)


class MultimodalDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, mod="acoustic", cls="train"):
        self.root = root
        self.cls = cls
        self.mod = mod
        # if len(MultimodalDataset.trainset.y) != 0 and cls != "train":
        #     print("Data has been previously loaded, fetching from previous lists.")
        # else:
        self.load_data(mod)

        if self.cls == "train":
            self.dataset = MultimodalDataset.trainset
        elif self.cls == "test":
            self.dataset = MultimodalDataset.testset
        elif self.cls == "valid":
            self.dataset = MultimodalDataset.validset

        self.feat = self.dataset.feat
        self.y = self.dataset.y


    def load_data(self, modality):
        dataset_path = os.path.join(params.iemocap_path, 'iemocap' + '_data.pkl')
        iemocap = pickle.load(open(dataset_path, 'rb'))
        #remove neutral labels since MOSEI does not have these
        for split in iemocap:
            iemocap[split]['labels'] = iemocap[split]['labels'][:, 1:, 1]


        #set up a new data dictionary to remove samples with only neutral label
        dataset = {}
        splits = ['train', 'valid', 'test']
        feats = ['labels', 'text', 'audio', 'vision']

        for split in splits:
            dataset[split] = {}
            for feat in feats:
                dataset[split][feat] = []

        #add samples to new dictionary if they have a label
        for split in iemocap:
            for i, x in enumerate(iemocap[split]['labels']):
                if np.sum(x) != 0:
                    for key in dataset[split]:
                        dataset[split][key].append(iemocap[split][key][i])
            for key in dataset[split]:
                dataset[split][key] = np.array(dataset[split][key])

        if modality == 'acoustic':
            params.mod_dim = dataset['test']['audio'].shape[2]

        elif modality == 'visual':
            params.mod_dim = dataset['test']['vision'].shape[2]

        for ds, split_type in [(MultimodalDataset.trainset, 'train'), (MultimodalDataset.validset, 'valid'),
                               (MultimodalDataset.testset, 'test')]:
            ds.y = torch.tensor(dataset[split_type]['labels'].astype(np.long)).cpu().detach()


            if modality == 'acoustic':
                ds.feat = torch.tensor(dataset[split_type]['audio'].astype(np.float32))
                #ds.feat = F.pad(ds.feat, (0,0,0,30))
                #print(ds.feat.shape)
                ds.feat[ds.feat == -np.inf] = 0
                ds.feat = ds.feat.clone().cpu().detach()

            elif modality == 'visual':
                ds.feat = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
                ds.feat = F.pad(ds.feat, (0, 0, 0, 30))

            #ds.feat = ds.feat


    def __getitem__(self, index):
        #inputLen = len(self.feat[index])
        return self.feat[index], self.y[index]

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MultimodalDataset(params.iemocap_path)
