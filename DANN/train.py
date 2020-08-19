import torch
import numpy as np

from DANN import params


def train(extractor, task_classifier, domain_classifier, optimizer, task_criterion, domain_criterion,
          src_data_loader, tgt_data_loader, epoch):

    #set up the models for training
    extractor.train()
    task_classifier.train()
    domain_classifier.train()

    start_steps = epoch * len(src_data_loader)
    total_steps = params.epochs * len(src_data_loader)

    for i, (src_data, tgt_data) in enumerate(zip(src_data_loader, tgt_data_loader)):

        #get the data and labels
        src_input, src_labels = src_data
        src_input = src_input.float()
        src_labels = src_labels.float()

        tgt_input, tgt_labels = tgt_data
        tgt_input = tgt_input.float()
        tgt_labels = tgt_labels.float()

        #TODO: add optimizer scheduler?
        optimizer.zero_grad()

        #TODO: check this
        #hyperparameter for domain classifier
        #constant is lambda in Ganin paper
        p = float(i + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-params.gamma * p)) - 1
        #print(constant)

        #set up LSTM classifier
        # if params.bidirectional == True:
        #     task_classifier.hidden = (torch.zeros(params.lstm_layers*2, params.batch_size, params.lstm_dim).cuda(),
        #                               torch.zeros(params.lstm_layers*2, params.batch_size, params.lstm_dim).cuda())
        # else:
        #     task_classifier.hidden = (torch.zeros(params.lstm_layers, params.batch_size, params.lstm_dim).cuda(),
        #                               torch.zeros(params.lstm_layers, params.batch_size, params.lstm_dim).cuda())

        #set up labels for domain classifier
        src_labels1 = torch.zeros(src_input.size()[0])
        src_labels2 = torch.ones(src_input.size()[0])
        src_domain_labels = torch.stack((src_labels1, src_labels2), dim=1)

        tgt_labels1 = torch.ones(tgt_input.size()[0])
        tgt_labels2 = torch.zeros(tgt_input.size()[0])
        tgt_domain_labels = torch.stack((tgt_labels1, tgt_labels2), dim=1)


        #get features from feature extractor
        src_feature = extractor(src_input)
        tgt_feature = extractor(tgt_input)

        #pass extracted source features to emotion classifier
        src_output = task_classifier(src_feature)
        #print(src_output)
        task_loss = task_criterion(src_output, src_labels)

        #pass extracted features to domain classifier
        src_preds = domain_classifier(src_feature, constant=constant)
        tgt_preds = domain_classifier(tgt_feature, constant=constant)
        src_loss = domain_criterion(src_preds, src_domain_labels)
        tgt_loss = domain_criterion(tgt_preds, tgt_domain_labels)
        domain_loss = tgt_loss + src_loss

        #theta defaults to one but can be changed
        loss = task_loss + params.theta * domain_loss
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                i * len(src_input), len(src_data_loader.dataset),
                100. * i / len(src_data_loader), loss.item(), task_loss.item(),
                domain_loss.item()
            ))
