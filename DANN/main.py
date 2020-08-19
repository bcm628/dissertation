import torch
import torch.optim as optim
import torch.nn as nn

import time
import argparse
import sys

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from util import dataloaders
from DANN import params, models, train, test
from DANN import pretrain


def tsne_embeddings(src_embeddings, tgt_embeddings, save_name, epoch):
    source = np.array(src_embeddings)
    target = np.array(tgt_embeddings)
    min_dim = min(np.shape(source)[0], np.shape(target)[0])
    source = source[:min_dim, :, :]
    target = target[:min_dim, :, :]
    source = np.reshape(source, (min_dim, -1))
    target = np.reshape(target, (min_dim, -1))
    #tsne = TSNE(n_components=2)
    pca = PCA(n_components=2)
    src_tsne = pca.fit_transform(source)
    tgt_tsne = pca.fit_transform(target)

    src_x = [value[0] for value in src_tsne]
    src_y = [value[1] for value in src_tsne]
    tgt_x = [value[0] for value in tgt_tsne]
    tgt_y = [value[1] for value in tgt_tsne]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(src_x, src_y, c='#648fff', label='source')
    ax.scatter(tgt_x, tgt_y, c='#dc267f', label='target')
    ax.legend()

    plt.savefig("{}_{}.pdf".format(save_name, epoch))
    plt.clf()

def main(args):
    params.epochs = args.epochs
    source = args.source_domain
    target = args.target_domain
    print("training on source domain ", source)
    print("testing on target domain ", target)
    params.modality = args.modality
    print("modality is: ", params.modality)
    lr = args.learning_rate
    params.lstm_layers = args.lstm_layers
    params.bidirectional = args.bidirectional


    #load in source and target data
    src_train_loader = dataloaders.get_train_loader(source)
    src_test_loader = dataloaders.get_test_loader(source)
    src_valid_loader = dataloaders.get_valid_loader(source)

    tgt_train_loader = dataloaders.get_train_loader(target)
    tgt_valid_loader = dataloaders.get_valid_loader(target)
    tgt_test_loader = dataloaders.get_test_loader(target)

    if params.modality == 'acoustic':
        params.proj_dim = 74
    elif params.modality == 'visual':
        params.proj_dim = 35

    feature_extractor = models.FeatureExtractor(params.proj_dim)

    # #feature_extractor = models.FeatureExtractor(batch_size=params.batch_size,
    #                                             hidden_dim=64,
    #                                             num_layers=2).cuda()
    #
    # init_hidden = emotion_classifier.main()
    # init_hidden = list(init_hidden)
    # for tensor in init_hidden:
    #     tensor.detach_()
    # init_hidden = tuple(init_hidden)

    task_classifier = models.TaskClassifier(batch_size=params.batch_size,
                                            hidden_dim=params.lstm_dim,
                                            num_layers=params.lstm_layers,
                                            bidirectional=args.bidirectional)
    # #domain_classifier = models.DomainClassifier(batch_size=params.batch_size,
    #                                             hidden_dim=params.rnn_dim,
    #                                             num_layers=params.rnn_layers).cuda()
    domain_classifier = models.DomainClassifier()


    task_criterion = nn.BCELoss()
    domain_criterion = nn.BCELoss()

    optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                            {'params': task_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr=lr)
    # for epoch in range(10):
    #     print('Epoch:', epoch)
    #     pretrain.train(task_classifier, src_train_loader, epoch=epoch, lr=lr)

    for epoch in range(params.epochs):
        print('Epoch: ', epoch)
        train.train(feature_extractor, task_classifier, domain_classifier, optimizer, task_criterion, domain_criterion,
                    src_train_loader, tgt_train_loader, epoch)
        src_embeddings = test.test(feature_extractor, task_classifier, domain_classifier, task_criterion, domain_criterion, src_valid_loader,
                                   mode='valid', split="Source Valid")
        # tgt_embeddings = test.test(feature_extractor, task_classifier, domain_classifier, task_criterion, domain_criterion, tgt_test_loader,
        #                            mode='test', split="Target Test")
        tgt_embeddings = test.test(feature_extractor, task_classifier, domain_classifier, task_criterion,
                                   domain_criterion, tgt_valid_loader,
                                   mode='test', split="Target Valid")

        if args.save_visuals:
            if epoch == 0 or (epoch+1)% 50 == 0:
                tsne_embeddings(src_embeddings, tgt_embeddings, args.visual_name, epoch)

        if args.save_model:
            if epoch == params.epochs-1:
                torch.save(feature_extractor.state_dict(), 'extractor.pt')



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', '-source', type=str, default='iemocap', help='Choose source domain: iemocap or mosei')
    parser.add_argument('--target_domain', '-target', type=str, default='mosei', help='Choose target domain: iemocap or mosei')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--modality', '-mod', type=str, default='visual', help='specify modality: acoustic or visual.')
    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--bidirectional', type=bool, default=True, help='Use a unidirectional or bidirectional LSTM classifier')
    parser.add_argument('--save_visuals', type=bool, default=False, help='Save TSNE visualizations of embeddings')
    parser.add_argument('--visual_name', type=str, default='pca_1', help='Choose save name for visualizations. Name will be appended with epoch')
    parser.add_argument('--save_model', type=bool, default=False, help='save final feature extractor state dict to disk')

    return parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()
    print('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))
    main(parse_arguments(sys.argv[1:]))
    time_passed = time.time() - start_time
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(time_passed)))