import argparse
import torch
import torch.nn.functional as F
import copy
import os
import sys
sys.path.append('../')
sys.path.append('../..')
from tqdm import tqdm
from torch.utils.data import DataLoader
from multi_class.config import load_config
from multi_class.datasets.dataset_loader import load_dataset
from multi_class.models.resnet20 import cifar100_resnet20 as resnet20
from multi_class.models.resnet50 import resnet50


def retrain(model, optim, train_loader, test_loader, n_epochs, early_stop=-1):
    best_epoch = 0
    best_acc = 0.0
    best_model = None
    early_stop_epochs = 0

    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch}')
        print('-' * 80)
        model.train()
        predicts, labels = [], []
        for batch_input, batch_labels in tqdm(train_loader, ncols=80, desc='Train'):
            batch_input, batch_labels = batch_input.to('cuda'), batch_labels.to('cuda')
            optim.zero_grad()

            batch_output = model(batch_input)
            loss = F.cross_entropy(batch_output, batch_labels)
            loss.backward()
            optim.step()

            predicts.append(torch.argmax(batch_output, dim=1).detach())
            labels.append(batch_labels.detach())

        predicts = torch.cat(predicts, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = torch.sum(predicts == labels) / labels.shape[0]
        print(f'Train Acc: {acc:.2%}')

        model.eval()
        with torch.no_grad():
            predicts, labels = [], []
            for batch_input, batch_labels in tqdm(test_loader, ncols=80, desc='Val  '):
                batch_input, batch_labels = batch_input.to('cuda'), batch_labels.to('cuda')
                batch_output = model(batch_input)
                predicts.append(torch.argmax(batch_output, dim=1).detach())
                labels.append(batch_labels.detach())

            predicts = torch.cat(predicts, dim=0)
            labels = torch.cat(labels, dim=0)
            acc = torch.sum(predicts == labels) / labels.shape[0]
            print(f'Val   Acc: {acc:.2%}')

        if acc >= best_acc:
            best_acc = acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            early_stop_epochs = 0
        else:
            early_stop_epochs += 1
            if early_stop_epochs == early_stop:
                print(f'Early Stop.\n\n')
                break
    return best_model, best_acc, best_epoch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet20', 'resnet50'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet'], required=True)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--superclass_type', type=str, choices=['predefined', 'random'], default='predefined')
    parser.add_argument('--target_superclass_idx', type=int, default=-1)
    parser.add_argument('--n_classes', type=int, default=-1,
                        help='When randomly construct superclasses, how many classes dose a reengineered_model recognize.')
    parser.add_argument('--shots', default=-1, type=int, help='how many samples for each classes.')
    parser.add_argument('--seed', type=int, default=1,
                        help='the random seed for sampling ``num_superclasses'' classes as target classes.')
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--early_stop', type=int, default=30)
    args = parser.parse_args()
    return args


def main():
    if args.model == 'resnet20':
        assert args.dataset == 'cifar100'
    elif args.model == 'resnet50':
        assert args.dataset == 'imagenet'
    else:
        raise ValueError

    save_dir = f'{configs.project_data_save_dir}/{args.model}_{args.dataset}/type_{args.superclass_type}_retrain'

    if args.superclass_type == 'predefined':
        save_dir = f'{save_dir}/tc_{args.target_superclass_idx}'
    elif args.superclass_type == 'random':
        save_dir = f'{save_dir}/n_classes_{args.n_classes}'
    else:
        raise ValueError
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/retrain_seed_{args.seed}_shots_{args.shots}.pth'

    dataset_train = load_dataset(dataset_name=args.dataset, is_train=True, shots=args.shots,
                                 superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                                 n_classes=args.n_classes, seed=args.seed, reorganize=True)
    dataset_test = load_dataset(dataset_name=args.dataset, is_train=False, shots=args.shots,
                                superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                                n_classes=args.n_classes, seed=args.seed, reorganize=True)
    assert dataset_train.classes == dataset_test.classes
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    print(f'INFO: Superclass {args.target_superclass_idx} contains {len(dataset_train.classes)} classes.\n')
    model = eval(args.model)(pretrained=False,
                             num_class_in_super=len(dataset_train.classes),
                             is_reengineering=False).to('cuda')

    optim = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=0.0001)
    retrained_model, best_acc, best_epoch = retrain(model, optim, train_loader, test_loader,
                                                    n_epochs=args.n_epochs, early_stop=args.early_stop)



    retrained_model_params = retrained_model.state_dict()
    torch.save(retrained_model_params, save_path)
    print()
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Acc  : {best_acc:.2%}')


if __name__ == '__main__':
    args = get_args()
    print(args)
    configs = load_config()

    num_workers = 16
    pin_memory = True

    model_name = args.model
    main()
