import torch.utils.data as data
from PIL import Image
import glob
import time
import numpy as np
import random
import os
import scipy.io as sio


class SDog120Data(data.Dataset):
    def __init__(self, root, is_train=True, transform=None, shots=5, seed=0):
        self.num_classes = 120
        self.transform = transform

        if is_train:
            mapfile = os.path.join(root, 'train_list.mat')
        else:
            mapfile = os.path.join(root, 'test_list.mat')
        assert os.path.exists(mapfile), 'Mapping txt is missing ({})'.format(mapfile)
        dset_list = sio.loadmat(mapfile)

        self.labels = []
        self.image_path = []

        for idx, f in enumerate(dset_list['file_list']):
            self.image_path.append(os.path.join(root, 'Images', f[0][0]))
            # Stanford Dog starts 1
            self.labels.append(dset_list['labels'][idx][0]-1)

        if is_train:
            self.image_path = np.array(self.image_path)
            self.labels = np.array(self.labels)

            if shots > 0:
                new_img_path = []
                new_labels = []
                for c in range(self.num_classes):
                    ids = np.where(self.labels == c)[0]
                    random.seed(seed)
                    random.shuffle(ids)
                    count = 0
                    for i in ids:
                        new_img_path.append(self.image_path[i])
                        new_labels.append(c)
                        count += 1
                        if count == shots:
                            break
                self.image_path = np.array(new_img_path)
                self.labels = np.array(new_labels)

        self.imgs = {}

    def __getitem__(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.labels[index]

    def __len__(self):
        return len(self.labels)
