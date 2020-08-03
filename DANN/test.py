import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from DANN import params


def test(extractor, classifier, optimizer, criterion, data_loader, split):
    extractor.eval()
    classifier.eval()

    src_output_all = []
    src_label_all = []

    for i, data in enumerate(data_loader):
        input, labels = data
        input = input.float().cuda()
        labels = labels.float().cuda()

        src_feature = extractor(input)
        output = classifier(src_feature)

        src_output_all.extend(output.tolist())
        src_label_all.extend(labels.tolist())

        loss = criterion(output, labels)

        if (i + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\t{} Loss: {:.6f}'.format(
                i * len(input), len(data_loader.dataset),
                100. * i / len(data_loader), split, loss.item()
            ))

    src_f1, src_acc = eval_data(src_label_all, src_output_all, split)

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

