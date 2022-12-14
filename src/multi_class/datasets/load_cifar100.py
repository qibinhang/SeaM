import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import numpy as np
import random
from multi_class.config import load_config


def load_cifar100_superclass(is_train, shots=-1, superclass_type='predefined', target_superclass_idx=0,
                             n_classes=10, seed=0, reorganize=True):
    assert superclass_type in ('predefined', 'random', 'no_superclass')
    config = load_config()
    # The mean and std could be different in different developers;
    # however, this will not influence the test accuracy much.
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    if is_train:
        transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        normalize])

    dataset =  datasets.CIFAR100(f'{config.dataset_dir}', train=is_train, transform=transform)

    if superclass_type == 'predefined':
        assert 0 <= target_superclass_idx <= 19
        sc2c = get_superclass2class_dict()
        dataset = _load_superclass_predefined(dataset, target_superclass_idx, sc2c, reorganize)
    elif superclass_type == 'random':
        dataset = _load_superclass_randomly(dataset, n_classes, seed, reorganize)
    elif superclass_type == 'no_superclass':  # Just for evaluating pretrained models on total test data.
        pass
    else:
        raise ValueError

    return dataset



def _load_superclass_predefined(dataset, target_superclass_idx, superclass2class, reorganize):
    classes = superclass2class[target_superclass_idx]
    dataset = extract_part_classes(dataset, classes, reorganize)
    return dataset

def _load_superclass_randomly(dataset, n_classes, seed, reorganize):
    random.seed(seed)
    classes = list(range(100))
    random.shuffle(classes)
    target_classes = classes[:n_classes]
    print(f'\nrandomly sampled classes: {target_classes}\n')

    dataset = extract_part_classes(dataset, target_classes, reorganize)
    return dataset


def get_superclass2class_dict():
    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c


def extract_part_classes(dataset, target_classes, reorganize):
    tc_data_list = []
    tc_targets_list = []
    for i, tc in enumerate(target_classes):
        tc_data_idx = np.where(np.array(dataset.targets) == tc)[0]
        tc_data = dataset.data[tc_data_idx]
        if reorganize:
            tc_targets = [i] * len(tc_data)
        else:
            tc_targets = [tc] * len(tc_data)
        tc_data_list.append(tc_data)
        tc_targets_list.append(tc_targets)
    tc_data = np.concatenate(tc_data_list, axis=0)
    tc_targets = np.concatenate(tc_targets_list, axis=0)

    dataset.data = tc_data
    dataset.targets = tc_targets

    idx2class = dict([(v, k) for k, v in  dataset.class_to_idx.items()])
    dataset.classes = [idx2class[idx] for idx in target_classes]
    return dataset


if __name__ == '__main__':
    load_cifar100_superclass(is_train=True, superclass_type='random', n_classes=10, reorganize=True)