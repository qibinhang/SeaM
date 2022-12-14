import torch.utils.data as data
from PIL import Image
import glob
import time
import numpy as np
import random
import os
from pdb import set_trace as st


class MIT67Data(data.Dataset):
    def __init__(self, root, is_train=False, transform=None, shots=-1, seed=0):
        self.num_classes = 67
        self.transform = transform
        cls = glob.glob(os.path.join(root, 'Images', '*'))
        self.cls_names = [name.split('/')[-1] for name in cls]

        if is_train:
            mapfile = os.path.join(root, 'TrainImages.txt') 
        else:
            mapfile = os.path.join(root, 'TestImages.txt') 

        assert os.path.exists(mapfile), 'Mapping txt is missing ({})'.format(mapfile)

        self.labels = []
        self.image_path = []

        with open(mapfile) as f:
            for line in f:
                self.image_path.append(os.path.join(root, 'Images', line.strip()))
                cls = line.split('/')[-2]
                self.labels.append(self.cls_names.index(cls))
        
        if is_train:
            indices = np.arange(0, len(self.image_path))
            random.seed(seed)
            random.shuffle(indices)
            self.image_path = np.array(self.image_path)[indices]
            self.labels = np.array(self.labels)[indices]

            if shots > 0:
                new_img_path = []
                new_labels = []
                for c in range(self.num_classes):
                    ids = np.where(self.labels == c)[0]
                    count = 0
                    for i in ids:
                        new_img_path.append(self.image_path[i])
                        new_labels.append(c)
                        count += 1
                        if count == shots:
                            break
                self.image_path = np.array(new_img_path)
                self.labels = np.array(new_labels)

        self.imgs = []

    def __getitem__(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.labels[index]

    def __len__(self):
        return len(self.labels)
