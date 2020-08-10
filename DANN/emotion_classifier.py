import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

import numpy as np

from DANN.util import dataloaders

train_loader = dataloaders.get_train_loader('mosei')
test_loader = dataloaders.get_test_loader('mosei')
valid_loader = dataloaders.get_valid_loader('mosei')

criterion = nn.BCELoss()

epochs = 200


class Task_Classifier(nn.Module):
    def __init__(self):
        super(Task_Classifier, self).__init__()
        self.lstm = nn.LSTM(74,
                            128,
                            num_layers=3,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(128*2, 3)
        self.sigmoid = nn.Sigmoid()
        # self.hidden = (torch.zeros(2*2, 20, 128).cuda(),
        #                torch.zeros(2*2, 20, 128).cuda())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        preds = self.fc(lstm_out[:, -1, :])
        preds = self.dropout(preds)
        out = self.sigmoid(preds)
        return out

task_classifier = Task_Classifier()
task_classifier = task_classifier.cuda()

optimizer = optim.Adam(task_classifier.parameters(), lr=0.001)

def train(classifier, optimizer, criterion, data_loader):
    for i, data in enumerate(data_loader):

        input, labels = data
        input = input.float().cuda()
        labels = labels.float().cuda()
        #input = input.float()
        #labels = labels.float()

        optimizer.zero_grad()

        #classifier.hidden = (torch.zeros(2*2, 20, 128).cuda(), torch.zeros(2*2, 20, 128).cuda())

        output = classifier(input)
        #print(output)

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

        if (i + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                i * len(input), len(data_loader.dataset),
                100. * i / len(data_loader), loss.item()
            ))

def test(classifier, optimizer, criterion, data_loader, split):
    classifier.eval()
    output_all = []
    label_all = []

    for i, data in enumerate(data_loader):
        input, labels = data
        input = input.float().cuda()
        labels = labels.float().cuda()
        #input = input.float()
        #labels = labels.float()

        output = classifier(input)
        output_all.extend(output.tolist())
        label_all.extend(labels.tolist())

        loss = criterion(output, labels)

        if (i + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\t{} Loss: {:.6f}'.format(
                i * len(input), len(data_loader.dataset),
                100. * i / len(data_loader), split, loss.item()
             ))

    f1, acc = eval_data(label_all, output_all, split)


def eval_data(label_all, output_all, split):
    truths = np.array(label_all)
    results = np.array(output_all)
    ones_idx = results > 0.5
    zeros_idx = results <= 0.5
    results[ones_idx] = 1
    results[zeros_idx] = 0

    f1_total = {}
    acc_total = {}
    print("{}\n".format(split))
    emo_labels = ['happy', 'sad', 'angry']

    for i, emo in enumerate(emo_labels):
        truths_i = truths[:, i]
        results_i = results[:, i]
        f1 = f1_score(results_i, truths_i, average='weighted')
        acc = accuracy_score(results_i, truths_i)
        f1_total[emo] = f1
        acc_total[emo] = acc
        print("\t%sF1 Score: %f" % (emo_labels[i], f1))
        print("\t%s Accuracy Score: %f" % (emo_labels[i], acc))

    return f1_total, acc_total


for epoch in range(epochs):
    print("epoch: {}".format(epoch))
    task_classifier.train()
    train(task_classifier, optimizer, criterion, train_loader)

    task_classifier.eval()
    test(task_classifier, optimizer, criterion, valid_loader, "Valid")
    test(task_classifier, optimizer, criterion, test_loader, "Test")