import argparse
import os.path
import sys
import torch
import time
from torch.utils.data import DataLoader
sys.path.append('../')
sys.path.append('../..')
from multi_class.reengineer import Reengineer
from multi_class.datasets.dataset_loader import load_dataset
from multi_class.config import load_config
from multi_class.models.resnet20 import cifar100_resnet20 as resnet20
from multi_class.models.resnet50 import resnet50
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet20', 'resnet50'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet'], required=True)
    parser.add_argument('--superclass_type', type=str, choices=['predefined', 'random'], default='predefined')
    parser.add_argument('--target_superclass_idx', type=int, default=-1)
    parser.add_argument('--n_classes', type=int, default=-1,
                        help='When randomly construct superclasses, how many classes dose a reengineered model recognize.')
    parser.add_argument('--shots', default=-1, type=int, help='how many samples for each classes.')
    parser.add_argument('--seed', type=int, default=0,
                        help='the random seed for sampling ``num_superclasses'' classes as target classes.')
    parser.add_argument('--n_epochs', type=int, default=300)

    parser.add_argument('--lr_head', type=float, default=0.1)
    parser.add_argument('--lr_mask', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1,
                        help='the weight for the weighted sum of two losses in re-engineering.')
    parser.add_argument('--early_stop', type=int, default=-1)
    args = parser.parse_args()
    return args


def reengineering(model, train_loader, test_loader, lr_mask, lr_head, n_epochs, alpha, early_stop, save_path, acc_pre_model):

    reengineer = Reengineer(model, train_loader, test_loader, acc_pre_model)
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
        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def eval_pretrained_model():
    model = eval(args.model)(pretrained=True).to('cuda')
    dataset_test = load_dataset(dataset_name=args.dataset, is_train=False, shots=args.shots,
                 superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                 n_classes=args.n_classes, seed=args.seed, reorganize=False)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    acc = evaluate(model, test_loader)
    return acc


def main():
    if args.model == 'resnet20':
        assert args.dataset == 'cifar100'
    elif args.model == 'resnet50':
        assert args.dataset == 'imagenet'
    else:
        raise ValueError

    dataset_train = load_dataset(dataset_name=args.dataset, is_train=True, shots=args.shots,
                                 superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                                 n_classes=args.n_classes, seed=args.seed, reorganize=True)
    dataset_test = load_dataset(dataset_name=args.dataset, is_train=False, shots=args.shots,
                                superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                                n_classes=args.n_classes, seed=args.seed, reorganize=True)
    assert dataset_train.classes == dataset_test.classes
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f'INFO: Superclass {args.target_superclass_idx} contains {len(dataset_train.classes)} classes: {dataset_train.classes}.\n')
    if len(dataset_train.classes) == 1:
        sys.exit(0)

    # prepare reengineered model saved path.
    save_dir = f'{config.project_data_save_dir}/{args.model}_{args.dataset}/{args.superclass_type}/tsc_{args.target_superclass_idx}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.superclass_type == 'predefined':
        save_path = f'{save_dir}/lr_head_mask_{args.lr_head}_{args.lr_mask}_alpha_{args.alpha}.pth'
    elif args.superclass_type == 'random':
        # save_path = f'{save_dir}/n_classes_{args.n_classes}.pth'
        raise ValueError('Not support for now.')
    else:
        raise ValueError

    model = eval(args.model)(pretrained=True,
                             num_classes_in_super=len(dataset_train.classes),
                             is_reengineering=True).to('cuda')

    acc_pre_model = eval_pretrained_model()
    print(f'Pretrained Model Test Acc: {acc_pre_model:.2%}\n\n')

    s_time = time.time()
    reengineering(model, train_loader, test_loader,
                  args.lr_mask, args.lr_head, args.n_epochs, args.alpha, args.early_stop,
                  save_path, acc_pre_model)
    e_time = time.time()
    print(f'Time Elapse: {(e_time - s_time)/60:.1f} min\n')

    print(f'Pretrained Model Test Acc: {acc_pre_model:.2%}\n\n')


if __name__ == '__main__':
    args = get_args()
    print(args)
    config = load_config()

    num_workers = 16
    pin_memory = True

    model_name = args.model
    main()