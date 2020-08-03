import torch
from DANN import params


def train(extractor, task_classifier, domain_classifier, optimizer, task_criterion, domain_criterion,
          src_data_loader, tgt_data_loader, epoch):
    extractor.train()
    task_classifier.train()
    domain_classifier.train()

    start_steps = epoch * len(src_data_loader)
    total_steps = params.epochs * len(src_data_loader)

    for i, (src_data, tgt_data) in enumerate(zip(src_data_loader, tgt_data_loader)):
        input, labels = src_data
        input = input.float().cuda()
        labels = labels.float().cuda()

        #TODO: check this
        p = float(i + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-params.gamma * p)) - 1

        optimizer.zero_grad()

        if params.bidirectional == True:
            task_classifier.hidden = (torch.zeros(params.lstm_layers*2, params.batch_size, params.lstm_dim).cuda(),
                                 torch.zeros(params.lstm_layers*2, params.batch_size, params.lstm_dim).cuda())
        else:
            task_classifier.hidden = (torch.zeros(params.lstm_layers, params.batch_size, params.lstm_dim).cuda(),
                                 torch.zeros(params.lstm_layers, params.batch_size, params.lstm_dim).cuda())

        src_feature = extractor(input)

        output = task_classifier(src_feature)
        loss = task_criterion(output, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                i * len(input), len(data_loader.dataset),
                100. * i / len(data_loader), loss.item()
            ))
