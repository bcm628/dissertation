import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from DANN import params


def test(extractor, task_classifier, domain_classifier, criterion, data_loader, mode, split):
    extractor.eval()
    task_classifier.eval()
    domain_classifier.eval()

    src_output_all = []
    src_label_all = []

    src_domain_preds = []

    embeddings = []

    for i, data in enumerate(data_loader):
        input, labels = data
        input = input.float().cuda()
        labels = labels.float().cuda()

        p = float(i) / len(data_loader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        src_feature = extractor(input)
        embeddings.extend(src_feature.cpu().detach().numpy())

        output = task_classifier(src_feature)

        src_output_all.extend(output.tolist())
        src_label_all.extend(labels.tolist())

        src_preds = domain_classifier(src_feature, constant)
        src_domain_preds.extend(src_preds.tolist())

        if mode == 'valid':
            loss = criterion(output, labels)

            if (i + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\t{} Loss: {:.6f}'.format(
                    i * len(input), len(data_loader.dataset),
                    100. * i / len(data_loader), split, loss.item()
                ))

    src_f1, src_acc = eval_data(src_label_all, src_output_all, split)
    eval_domain(src_domain_preds, split)
    return embeddings

#TODO: check this
def eval_data(label_all, output_all, split):
    truths = np.array(label_all)
    results = np.array(output_all)
    print(results)
    ones_idx = results > 0.5
    zeros_idx = results <= 0.5
    results[ones_idx] = 1
    results[zeros_idx] = 0
    print(results)

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


def eval_domain(domain_preds, split):
    preds = np.array(domain_preds)
    ones_idx = preds > 0.5
    zeros_idx = preds <= 0.5
    preds[ones_idx] = 1
    preds[zeros_idx] = 0
    correct = np.count_nonzero(preds)
    total = np.size(preds)
    print("\t{} Domain Accuracy: {:.6f}".format(split, correct/total))
