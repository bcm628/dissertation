import torch.nn as nn
import torch.nn.functional as F

from consts import global_consts as gc
from transformer import Models

#TODO: pass visual and acoustic input into Embedding Layer then input that output into Net()

class AcousticEmbedding(nn.Module):
    """Feedforward DNN feature extractor"""
    def __init__(self, input_dims, output_dims):
        super(AcousticEmbedding, self).__init__()
        # self.batch_size = batch_size
        # self.num_layers = num_layers
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        #self.fc4 = nn.Linear(54, 64)
        self.fc3 = nn.Linear(256, output_dims)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        #x = self.dropout(x)
        x = F.relu(self.dropout(self.fc2(x)))
        #x = F.relu(self.fc4(x))
        x = self.fc3(x)

        return x

class VisualEmbedding(nn.Module):
    """Feedforward DNN feature extractor"""
    def __init__(self, input_dims, output_dims):
        super(VisualEmbedding, self).__init__()
        # self.batch_size = batch_size
        # self.num_layers = num_layers
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        #self.fc4 = nn.Linear(54, 64)
        self.fc3 = nn.Linear(256, output_dims)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc4(x))
        x = self.fc3(x)

        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conf = gc.config
        proj_dim_a = conf['proj_dim_a']
        proj_dim_v = conf['proj_dim_v']
        #self.proj_a = nn.Linear(gc.dim_a, proj_dim_a)
        #self.proj_v = nn.Linear(gc.dim_v, proj_dim_v)
        self.transformer_encoder = Models.TransformerEncoder((gc.dim_l, proj_dim_a, proj_dim_v),
                                                             conf['n_layers'], conf['dropout'])
        dim_total_proj = conf['dim_total_proj']
        dim_total = gc.dim_l + proj_dim_a + proj_dim_v
        self.gru = nn.GRU(input_size=dim_total, hidden_size=dim_total_proj)
        if gc.dataset == 'iemocap':
            final_out_dim = 2 * len(gc.best.iemocap_emos)
        #TODO: check that this is right
        elif gc.dataset == 'mosei_new':
            final_out_dim = 2 * len(gc.best.mosei_emos)
        elif gc.dataset == 'pom':
            final_out_dim = len(gc.best.pom_cls)
        else:
            final_out_dim = 1
        self.finalW = nn.Linear(dim_total_proj, final_out_dim)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, words, covarep, facet, inputLens):
        state = self.transformer_encoder((words, covarep, facet))
        #state = self.transformer_encoder((words, self.proj_a(covarep), self.proj_v(facet)))
        # convert input to GRU from shape [batch_size, seq_len, input_size] to [seq_len, batch_size, input_size]
        _, gru_last_h = self.gru(state.transpose(0, 1))
        return self.finalW(gru_last_h.squeeze()).squeeze()
