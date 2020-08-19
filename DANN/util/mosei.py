import pickle
import numpy as np
import torch
import torch.utils.data as Data

from DANN.util.iemocap import MultimodalSubdata
from DANN import params


class MoseiNewDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train", mod='acoustic'):
        self.root = root
        self.cls = cls
        self.mod = mod
        # if len(MoseiNewDataset.trainset.y) != 0 and cls != "train":
        #     print("Data has been previously loaded, fetching from previous lists.")
        # else:
        self.load_data(mod)

        if self.cls == "train":
            self.dataset = MoseiNewDataset.trainset
        elif self.cls == "test":
            self.dataset = MoseiNewDataset.testset
        elif self.cls == "valid":
            self.dataset = MoseiNewDataset.validset

        self.feat = self.dataset.feat
        self.y = self.dataset.y


    def load_data(self, modality):
        #dataset = format_mosei(os.path.join(params.mosei_path, 'tensors.pkl'), three_dim=True)

        # dataset = pickle.load(open("C:/Users/bcmye/PycharmProjects/dissertation/FMT/Code/mosei_dict2.pickle", "rb"),
        #                       encoding='latin1')
        dataset = pickle.load(open('/content/drive/My Drive/Colab Notebooks/mosei_dict2.pickle', 'rb'),
                              encoding='latin1')

        if modality == 'acoustic':
            #params.padding_len = dataset['test']['language'].shape[1]
            params.mod_dim = dataset['test']['acoustic'].shape[2]

        elif modality == 'visual':
            #params.padding_len = dataset['test']['language'].shape[1]
            params.mod_dim = dataset['test']['visual'].shape[2]

        for ds, split_type in [(MoseiNewDataset.trainset, 'train'), (MoseiNewDataset.validset, 'valid'),
                               (MoseiNewDataset.testset, 'test')]:
            if modality == 'acoustic':
                ds.feat = torch.tensor(dataset[split_type]['acoustic'].astype(np.float32))
                ds.feat[ds.feat == -np.inf] = 0
                ds.feat = ds.feat.clone().cpu().detach()

            elif modality == 'visual':
                ds.feat = torch.tensor(dataset[split_type]['visual'].astype(np.float32)).cpu().detach()

            ds.y = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()


    def __getitem__(self, index):
        inputLen = len(self.feat[index])
        return self.feat[index], self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MoseiNewDataset(params.mosei_path)
