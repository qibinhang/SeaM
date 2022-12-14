import torchvision.transforms as transforms
import torchvision.datasets as datasets
from multi_class.config import load_config
import random
import numpy as np

import copy
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def load_imagenet_superclass(is_train, shots=-1, superclass_type='predefined', target_superclass_idx=0,
                             n_classes=10, seed=0, reorganize=True):
    """
    superclasses are referred to https://github.com/noameshed/novelty-detection
    superclass_type = 'predefined' means that superclasses are constructed according to the similarity of classes.
    For instance, the superclass in CIFAR-100 is predefined by the authors,
    and the superclass in imagenet is predefined by a public GitHub repository.

    superclass_type = 'random' means that superclasses are constructed randomly.

    target_superclass is meaningful just for superclass_type = 'predefined'.
    n_classes and seed are meaningful just for superclass_type = 'random'
    """
    assert superclass_type in ('predefined', 'random', 'no_superclass')
    config = load_config()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        dataset_dir = f'{config.dataset_dir}/ILSVRC2012/train'
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        dataset_dir = f'{config.dataset_dir}/ILSVRC2012/val'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = MyImageFolder(
        root=dataset_dir, transform=transform, shots=shots
    )

    if superclass_type == 'predefined':
        assert 0 <= target_superclass_idx <= 66
        sc_idx2sc_path = f"{os.path.dirname(__file__)}/imagenet_superclass_idx2sc.csv"
        scn2c_path = f"{os.path.dirname(__file__)}/imagenet_superclass_scn2c.csv"
        dataset = _load_superclass_predefined(dataset, target_superclass_idx, sc_idx2sc_path, scn2c_path, reorganize)
    elif superclass_type == 'random':
        dataset = _load_superclass_randomly(dataset, n_classes, seed, reorganize)
    elif superclass_type == 'no_superclass':  # Just for evaluating pretrained models on total test data.
        pass
    else:
        raise ValueError

    return dataset


def _load_superclass_predefined(dataset, target_superclass_idx, superclass_idx2sc_path, superclass_scn2c_path, reorganize):
    with open(superclass_idx2sc_path, 'r') as f:
        for each_line in f.readlines():
            sc_idx = int(each_line.split(',')[0])
            if target_superclass_idx == sc_idx:
                tsc_name = each_line.strip().split(',')[1]
                break

    with open(superclass_scn2c_path, 'r') as f:
        for each_line in f.readlines():
            sc_name = each_line.split(',')[0]
            if tsc_name == sc_name:
                classes = each_line.strip().split(',')[1].split(' ')
                break

    dataset = extract_part_classes(dataset, classes, reorganize)
    return dataset

def _load_superclass_randomly(dataset, n_classes, seed, reorganize):
    random.seed(seed)
    classes = copy.deepcopy(dataset.classes)
    random.shuffle(classes)
    target_classes = classes[:n_classes]
    print(f'\nrandomly sampled classes: {target_classes}\n')

    dataset = extract_part_classes(dataset, target_classes, reorganize)
    return dataset


def extract_part_classes(dataset, target_classes, reorganize):
    new_samples = []
    new_imgs = []
    new_targets = []
    new_class_to_idx = dict()
    dataset_target = np.array(dataset.targets)
    for new_class_idx, each_class_name in enumerate(target_classes):
        class_idx = dataset.class_to_idx[each_class_name]
        target_sample_idx = np.where(dataset_target == class_idx)[0]

        for each_ts_idx in target_sample_idx:
            each_sample = dataset.samples[each_ts_idx]
            assert each_sample[1] == class_idx
            if reorganize:
                new_each_sample = (each_sample[0], new_class_idx)
                new_targets.append(new_class_idx)
                new_class_to_idx[each_class_name] = new_class_idx
            else:
                new_each_sample = each_sample
                new_targets.append(class_idx)
                new_class_to_idx[each_class_name] = class_idx

            new_samples.append(new_each_sample)
            new_imgs.append(new_each_sample)
    dataset.samples = new_samples
    dataset.imgs = new_imgs
    dataset.classes = target_classes
    dataset.targets = new_targets
    if reorganize:
        dataset.class_to_idx = new_class_to_idx
    return dataset


class MyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, shots=-1):
        self.shots = shots
        super().__init__(root, transform)

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions,
                            is_valid_file=is_valid_file, shots=self.shots)


def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        shots: int = -1
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue

        n_samples_each_class = 0
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    n_samples_each_class += 1
                    if n_samples_each_class == shots:
                        break
    return instances


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)