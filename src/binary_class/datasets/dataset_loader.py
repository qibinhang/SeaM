import sys
sys.path.append('..')
from binary_class.datasets.load_cifar10 import load_cifar10
from binary_class.datasets.load_cifar100 import load_cifar100



def load_dataset(dataset_name, is_train, shots=-1, target_class: int=-1, reorganize=False):
    if dataset_name == 'cifar10':
        dataset = load_cifar10(is_train, shots=shots, target_class=target_class, reorganize=reorganize)
    elif dataset_name == 'cifar100':
        dataset = load_cifar100(is_train, shots=shots, target_class=target_class, reorganize=reorganize)
    else:
        raise ValueError
    return dataset
