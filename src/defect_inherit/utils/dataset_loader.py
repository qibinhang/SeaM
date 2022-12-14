import torch.utils.data as data
import torchvision.transforms as transforms
import sys
sys.path.append('..')
from defect_inherit.dataset.cub200 import CUB200Data
from defect_inherit.dataset.stanford_dog import SDog120Data
from defect_inherit.dataset.flower102 import Flower102Data
from defect_inherit.dataset.mit67 import MIT67Data
from defect_inherit.dataset.stanford_40 import Stanford40Data
from defect_inherit.config import load_config


def load_dataset(dataset_name, is_train, shots=-1):
    assert dataset_name in ('cub200', 'dog120', 'flower102', 'mit67', 'action40')

    config = load_config()
    dataset_dir = config.dataset_dir

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_train:
        transform = transforms.Compose([
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize,
             ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if dataset_name == 'cub200':
        dataset = CUB200Data(dataset_dir=f'{dataset_dir}/CUB_200_2011',
                             is_train=is_train,
                             transform=transform,
                             shots=shots)
    elif dataset_name == 'dog120':
        dataset = SDog120Data(root=f'{dataset_dir}/stanford_dog',
                              is_train=is_train,
                              transform=transform,
                              shots=shots)
    elif dataset_name == 'flower102':
        dataset = Flower102Data(root=f'{dataset_dir}/Flower_102',
                                is_train=is_train,
                                transform=transform,
                                shots=shots)
    elif dataset_name == 'mit67':
        dataset = MIT67Data(root=f'{dataset_dir}/MIT_67',
                            is_train=is_train,
                            transform=transform,
                            shots=shots)
    elif dataset_name == 'action40':
        dataset = Stanford40Data(root=f'{dataset_dir}/stanford_40',
                                 is_train=is_train,
                                 transform=transform,
                                 shots=shots)
    else:
        raise ValueError
    return dataset


if __name__ == '__main__':
    dataset_name = 'flower102'
    dataset_train = load_dataset(dataset_name, is_train=True)
    dataset_test = load_dataset(dataset_name, is_train=False)
    for i in dataset_train.img_ids:
        if i in dataset_test.img_ids:
            print('Test in training...')
    print('Test PASS!')
    print('Train', dataset_train.img_ids[:5])
    print('Test', dataset_test.img_ids[:5])