import argparse
import os.path
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append('../')
sys.path.append('../..')
from binary_class.reengineer import Reengineer
from binary_class.datasets.dataset_loader import load_dataset
from binary_class.config import load_config
from tqdm import tqdm
from binary_class.models.vgg import cifar10_vgg16_bn as cifar10_vgg16
from binary_class.models.vgg import cifar100_vgg16_bn as cifar100_vgg16
from binary_class.models.resnet import cifar10_resnet20, cifar100_resnet20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vgg16', 'resnet20', ], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--shots', default=-1, type=int, help='how many samples for each classes.')
    parser.add_argument('--seed', type=int, default=0, help='the random seed for sampling ``num_classes'' classes as target classes.')
    parser.add_argument('--n_epochs', type=int, default=300)

    parser.add_argument('--lr_head', type=float, default=0.1)
    parser.add_argument('--lr_mask', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1,
                        help='the weight for the weighted sum of two losses in re-engineering.')
    parser.add_argument('--early_stop', type=int, default=-1)
    parser.add_argument('--tuning_param', action='store_true')
    args = parser.parse_args()
    return args


def reengineering(model, train_loader, test_loader, lr_mask, lr_head, n_epochs, alpha, early_stop, acc_pre_model):
    save_dir = f'{config.project_data_save_dir}/{args.model}_{args.dataset}/tc_{args.target_class}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/lr_head_mask_{args.lr_head}_{args.lr_mask}_alpha_{args.alpha}.pth'

    reengineer = Reengineer(model, train_loader, test_loader, acc_pre_model=acc_pre_model)
    reengineered_model = reengineer.alter(lr_mask=lr_mask, lr_head=lr_head,
                              n_epochs=n_epochs, alpha=alpha, early_stop=early_stop)


    masks = reengineered_model.get_masks()
    module_head = reengineered_model.get_module_head()
    masks.update(module_head)
    torch.save(masks, save_path)

    # check
    model_static = model.state_dict()
    reengineered_model_static = reengineered_model.state_dict()
    for k in model_static:
        if 'mask' not in k and 'module_head' not in k:
            model_weight = model_static[k]
            reengineered_model_weight = reengineered_model_static[k]
            assert (model_weight == reengineered_model_weight).all()


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    n_corrects = 0
    n_samples = 0

    for batch_inputs, batch_labels in tqdm(test_loader, ncols=80, desc=f'Eval '):
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')
        batch_outputs = model(batch_inputs)
        n_samples += batch_labels.shape[0]
        batch_preds = torch.argmax(batch_outputs, dim=1)

        if args.target_class >= 0:
            # Transfer the multiclass classification into the binary classification.
            batch_preds = batch_preds == args.target_class
            batch_labels = batch_labels == args.target_class

        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def eval_pretrained_model():
    model = eval(f'{args.dataset}_{args.model}')(pretrained=True).to('cuda')
    dataset_test = load_dataset(args.dataset, is_train=False, target_class=args.target_class, reorganize=False)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    acc = evaluate(model, test_loader)
    return acc


def main():
    acc_pre_model = eval_pretrained_model()
    print(f'\nPretrained Model Test Acc: {acc_pre_model:.2%}\n\n')

    model = eval(f'{args.dataset}_{args.model}')(pretrained=True, is_reengineering=True).to('cuda')
    dataset_train = load_dataset(args.dataset, is_train=True, shots=args.shots,
                                 target_class=args.target_class, reorganize=True)
    dataset_test = load_dataset(args.dataset, is_train=False, target_class=args.target_class, reorganize=True)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    reengineering(model, train_loader, test_loader, args.lr_mask, args.lr_head,
                  args.n_epochs, args.alpha, args.early_stop, acc_pre_model)

    print(f'\nPretrained Model Test Acc: {acc_pre_model:.2%}\n\n')

if __name__ == '__main__':
    args = get_args()
    print(args)
    config = load_config()

    num_workers = 8
    pin_memory = True

    model_name = args.model
    main()