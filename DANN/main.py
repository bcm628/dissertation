import torch
import torch.optim as optim
import torch.nn as nn

import time
import argparse
import sys

from DANN.util import dataloaders
from DANN import params, models, train, test

def main(args):
    params.epochs = args.epochs
    source = args.source_domain
    target = args.target_domain
    print("training on source domain ", source)
    print("testing on target domain ", target)
    params.modality = args.modality
    print("modality is: ", params.modality)
    lr = args.learning_rate
    params.save_name = args.visual_name
    params.lstm_layers = args.lstm_layers
    params.bidirectional = args.bidirectional


    #load in source and target data
    src_train_loader = dataloaders.get_train_loader(source)
    src_test_loader = dataloaders.get_test_loader(source)
    src_valid_loader = dataloaders.get_valid_loader(source)

    tgt_train_loader = dataloaders.get_train_loader(target)
    tgt_valid_loader = dataloaders.get_valid_loader(target)
    tgt_test_loader = dataloaders.get_test_loader(target)

    feature_extractor = models.FeatureExtractor().cuda()
    task_classifier = models.TaskClassifier(batch_size=params.batch_size,
                                            hidden_dim=params.lstm_dim,
                                            num_layers=params.lstm_layers,
                                            bidirectional=args.bidirectional).cuda()
    domain_classifier = models.DomainClassifier().cuda()

    task_criterion = nn.BCELoss()
    domain_criterion = nn.BCELoss()

    optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                            {'params': task_classifier.parameters()}], lr=lr)

    for epoch in range(params.epochs):
        print('Epoch: ', epoch)
        train.train(feature_extractor, task_classifier, domain_classifier, optimizer, task_criterion, domain_criterion, src_train_loader, tgt_train_loader, epoch)
        test.test(feature_extractor, task_classifier, optimizer, task_criterion, src_valid_loader, split="Source Valid")
        test.test(feature_extractor, task_classifier, optimizer, task_criterion, tgt_test_loader, split="Target Test")



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', '-source', type=str, default='iemocap', help='Choose source domain: iemocap or mosei')
    parser.add_argument('--target_domain', '-target', type=str, default= 'mosei', help='Choose target domain: iemocap or mosei')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--modality', '-mod', type=str, default='acoustic', help='specify modality: acoustic or visual.')
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--bidirectional', type=bool, default=True, help= 'Use a unidirectional or bidirectional LSTM classifier')
    parser.add_argument('--save_visuals', type=bool, default=False, help='Save TSNE visualizations of embeddings')
    parser.add_argument('--visual_name', type=str, default=None, help='Choose save name for visualizations. Name will be appended with epoch')

    return parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()
    print('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))
    main(parse_arguments(sys.argv[1:]))
    time_passed = time.time() - start_time
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(time_passed)))