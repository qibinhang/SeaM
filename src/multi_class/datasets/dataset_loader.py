import sys
sys.path.append('..')
from multi_class.datasets.load_imagenet import load_imagenet_superclass
from multi_class.datasets.load_cifar100 import load_cifar100_superclass

def load_dataset(dataset_name, is_train: bool, shots=-1,
                 superclass_type='predefined', target_superclass_idx=0,
                 n_classes=10, seed=0, reorganize=False):
    if dataset_name == 'imagenet':
        dataset = load_imagenet_superclass(is_train=is_train, shots=shots,
                                           superclass_type=superclass_type, target_superclass_idx=target_superclass_idx,
                                           n_classes=n_classes, seed=seed, reorganize=reorganize)
    elif dataset_name == 'cifar100':
        dataset = load_cifar100_superclass(is_train=is_train, shots=shots,
                                           superclass_type=superclass_type, target_superclass_idx=target_superclass_idx,
                                           n_classes=n_classes, seed=seed, reorganize=reorganize)
    else:
        raise ValueError
    return dataset


if __name__ == '__main__':
    load_dataset('cifar100', is_train=False, superclass_type='predefined', target_superclass_idx=10, reorganize=False)

    print()