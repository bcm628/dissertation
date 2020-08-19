import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import params


def test(extractor, task_classifier, domain_classifier, task_criterion, domain_criterion, data_loader, mode, split):
    extractor.eval()
    task_classifier.eval()
    domain_classifier.eval()

    src_output_all = []
    src_label_all = []

    src_domain_preds = []

    embeddings = []

    for i, data in enumerate(data_loader):
        input, labels = data
        input = input.float()
        labels = labels.float()

        p = float(i) / len(data_loader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        src_feature = extractor(input)
        embeddings.extend(src_feature.cpu().detach().numpy())

        output = task_classifier(src_feature)

        src_output_all.extend(output.tolist())
        src_label_all.extend(labels.tolist())

        src_preds = domain_classifier(src_feature, constant)
        #print(src_preds.shape) [20, 2]

        src_domain_preds.extend(src_preds.tolist())

        if mode == 'valid':
            labels1 = torch.zeros(input.size()[0])
            labels2 = torch.ones(input.size()[0])
            domain_labels = torch.stack((labels1, labels2), dim=1)
            domain_loss =  domain_criterion(src_preds, domain_labels)

        elif mode =='test':
            labels1 = torch.ones(input.size()[0])
            labels2 = torch.zeros(input.size()[0])
            domain_labels = torch.stack((labels1, labels2), dim=1)
            domain_loss =  domain_criterion(src_preds, domain_labels)

        task_loss = task_criterion(output, labels)

        loss = task_loss + params.theta * domain_loss

        if (i + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tValid Loss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                i * len(input), len(data_loader.dataset),
                100. * i / len(data_loader), loss.item(), task_loss.item(),
                domain_loss.item()
            ))

    src_f1, src_acc = eval_data(src_label_all, src_output_all, split)

    if mode == 'valid':
        eval_src_domain(src_domain_preds, split)
    elif mode == 'test':
        eval_tgt_domain(src_domain_preds, split)
    return embeddings

#TODO: check this
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

    for i, emo in enumerate(params.emo_labels):
        truths_i = truths[:, i]
        results_i = results[:, i]
        f1 = f1_score(results_i, truths_i, average='weighted')
        acc = accuracy_score(results_i, truths_i)
        f1_total[emo] = f1
        acc_total[emo] = acc
        print("\t%sF1 Score: %f" % (params.emo_labels[i], f1))
        print("\t%s Accuracy Score: %f" % (params.emo_labels[i], acc))

    return f1_total, acc_total

#TODO: fix this
#for source, second row is ones; for target, first row is ones

def eval_src_domain(domain_preds, split):
    preds = np.array(domain_preds)
    ones_idx = preds > 0.5
    zeros_idx = preds <= 0.5
    preds[ones_idx] = 1
    preds[zeros_idx] = 0
    correct = np.count_nonzero(preds.transpose()[1])
    total = np.size(preds.transpose()[1])
    print("\t{} Domain Accuracy: {:.6f}".format(split, correct/total))

def eval_tgt_domain(domain_preds, split):
    preds = np.array(domain_preds)
    ones_idx = preds > 0.5
    zeros_idx = preds <= 0.5
    preds[ones_idx] = 1
    preds[zeros_idx] = 0
    correct = np.count_nonzero(preds.transpose()[0])
    total = np.size(preds.transpose()[0])
    print("\t{} Domain Accuracy: {:.6f}".format(split, correct/total))