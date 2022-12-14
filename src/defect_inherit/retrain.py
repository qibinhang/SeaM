import argparse
import os.path
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append('../')
from defect_inherit.models.resnet import resnet18
from utils.dataset_loader import load_dataset
from defect_inherit.finetuner import finetune
from defect_inherit.config import load_config


def retrain(model, train_loader, test_loader, n_epochs, lr, momentum, weight_decay, save_path):
    optim = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model_rt, best_acc, best_epoch = finetune(model, optim, train_loader, test_loader, n_epochs=n_epochs)
    model_rt_params = model_rt.state_dict()
    torch.save(model_rt_params, save_path)
    print()
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Acc  : {best_acc:.2%}')
    print(f'Finish Fine-tuning.\n\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet18',], required=True)
    parser.add_argument('--dataset', type=str,
                        choices=['cub200', 'dog120', 'flower102', 'mit67', 'action40'], required=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    configs = load_config()
    num_workers = 16
    pin_memory = True
    save_path = f'{configs.retrain_dir}/{args.model}_{args.dataset}/model_rt.pth'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    dataset_train = load_dataset(args.dataset, is_train=True)
    dataset_test = load_dataset(args.dataset, is_train=False)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    num_classes = dataset_train.num_classes
    model = eval(args.model)(pretrained=False, dropout=args.dropout, num_classes=num_classes).to('cuda')

    retrain(model, train_loader, test_loader,
            args.n_epochs, args.lr, args.momentum, args.weight_decay, save_path)


if __name__ == '__main__':
    main()
