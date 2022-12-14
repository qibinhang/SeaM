import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import numpy as np
from binary_class.config import load_config


def load_cifar100(is_train, shots=-1, target_class: int=-1, reorganize=False):
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

    if target_class >= 0:
        assert 0 <= target_class <= 99
        dataset = extract_target_class(dataset, target_class, reorganize)

    return dataset


def extract_target_class(dataset, target_class, reorganize):
    tc_data_idx = np.where(np.array(dataset.targets) == target_class)[0]
    tc_data = dataset.data[tc_data_idx]
    if reorganize:
        tc_targets = [0] * len(tc_data)
    else:
        tc_targets = [target_class] * len(tc_data)

    non_tc_data_idx = np.where(np.array(dataset.targets) != target_class)[0]
    np.random.seed(0)
    np.random.shuffle(non_tc_data_idx)
    non_tc_data_idx = non_tc_data_idx[:len(tc_data)]
    non_tc_data = dataset.data[non_tc_data_idx]
    if reorganize:
        non_tc_targets = [1] * len(tc_data)
    else:
        non_tc_targets = [dataset.targets[i] for i in non_tc_data_idx]

    dataset.data = np.concatenate((tc_data, non_tc_data), axis=0)
    dataset.targets = tc_targets + non_tc_targets
    return dataset


if __name__ == '__main__':
    load_cifar100(is_train=True, target_class=1, reorganize=True)