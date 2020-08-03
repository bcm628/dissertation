import torch
import torch.nn as nn
import torch.nn.functional as F

from DANN import params


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


class FeatureExtractor(nn.Module):
    """Feedforward DNN feature extractor"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # self.batch_size = batch_size
        # self.num_layers = num_layers

        self.fc1 = nn.Linear(params.mod_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, params.mod_dim)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.relu(x)


class TaskClassifier(nn.Module):
    """LSTM emotion classifier"""
    def __init__(self, batch_size, hidden_dim, num_layers, bidirectional):
        super(TaskClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(params.mod_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim*num_layers, 3)
        self.sigmoid = nn.Sigmoid()

        if bidirectional:
            self.hidden = (torch.zeros(num_layers*2, batch_size, hidden_dim).cuda(),
                           torch.zeros(num_layers*2, batch_size, hidden_dim).cuda())
        else:
            self.hidden = (torch.zeros(num_layers, batch_size, hidden_dim).cuda(),
                           torch.zeros(num_layers, batch_size, hidden_dim).cuda())

    def forward(self, x):
        lstm_out, _ = self.lstm(x, self.hidden)
        out = self.fc(lstm_out[:, -1, :])
        out = self.sigmoid(out)
        return out

class DomainClassifier(nn.Module):
    """Classifies the domain of input samples. Trained to increase loss"""
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(params.mod_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout()
        self.sig = nn.Sigmoid()

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return self.sig(x)