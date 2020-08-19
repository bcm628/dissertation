import torch
import torch.nn as nn
import torch.nn.functional as F

import params


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    from @CuthbertCai
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

#TODO: try using hidden states
#TODO: add linear layer?
class FeatureExtractor_RNN(nn.Module):
    def __init__(self, batch_size, hidden_dim, num_layers):
        super(FeatureExtractor, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(params.mod_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          nonlinearity='relu')
        self.hidden = torch.zeros(num_layers, batch_size, hidden_dim).cuda()
        self.fc = nn.Linear(hidden_dim, params.mod_dim)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rnn_out, _ = self.rnn(x, self.hidden)
        out = self.fc(rnn_out)
        print(out)
        #out = self.sigmoid(preds)
        return out

class FeatureExtractor(nn.Module):
    """Feedforward DNN feature extractor"""
    def __init__(self, proj_dim):
        super(FeatureExtractor, self).__init__()
        # self.batch_size = batch_size
        # self.num_layers = num_layers
        self.proj_dim = proj_dim
        self.fc1 = nn.Linear(params.mod_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        #self.fc4 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, proj_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.dropout(self.fc4(x)))
        x = self.fc3(x)

        return x


class TaskClassifier(nn.Module):
    """LSTM emotion classifier"""
    def __init__(self, batch_size, hidden_dim, num_layers, bidirectional):
        super(TaskClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(params.proj_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, 3)
        else:
            self.fc = nn.Linear(hidden_dim, 3)

        self.sigmoid = nn.Sigmoid()

        # if bidirectional:
        #     self.hidden = (torch.zeros(num_layers*2, batch_size, hidden_dim).cuda(),
        #                    torch.zeros(num_layers*2, batch_size, hidden_dim).cuda())
        # else:
        #     self.hidden = (torch.zeros(num_layers, batch_size, hidden_dim).cuda(),
        #                    torch.zeros(num_layers, batch_size, hidden_dim).cuda())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        out = self.sigmoid(out)
        return out


class DomainClassifier_RNN(nn.Module):
    def __init__(self, batch_size, hidden_dim, num_layers):
        super(DomainClassifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(params.mod_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          nonlinearity='relu',
                          dropout=0.2)
        self.hidden = torch.zeros(num_layers, batch_size, hidden_dim).cuda()
        self.fc = nn.Linear(hidden_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        rnn_out, _ = self.rnn(x, self.hidden)
        preds = self.fc(rnn_out[:, -1, :])
        out = self.sigmoid(preds)
        return out



class DomainClassifier(nn.Module):
    """Classifies the domain of input samples. Trained to increase loss"""
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(params.proj_dim * params.seq_len, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout()
        self.sig = nn.Sigmoid()

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = x.view(params.batch_size, -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return self.sig(x)
