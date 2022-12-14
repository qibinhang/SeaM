import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import random
import os
import glob
import numpy as np


class Flower102Data(data.Dataset):
    def __init__(self, root, is_train=True, transform=None, shots=-1, seed=0):
        self.num_classes = 102
        self.transform = transform
        imglabel_map = os.path.join(root, 'imagelabels.mat')
        setid_map = os.path.join(root, 'setid.mat')
        assert os.path.exists(imglabel_map), 'Mapping txt is missing ({})'.format(imglabel_map)
        assert os.path.exists(setid_map), 'Mapping txt is missing ({})'.format(setid_map)

        imagelabels = sio.loadmat(imglabel_map)['labels'][0]
        setids = sio.loadmat(setid_map)

        if is_train:
            ids = np.concatenate([setids['trnid'][0], setids['valid'][0]])
        else:
            ids = setids['tstid'][0]

        self.labels = []
        self.image_path = []

        for i in ids:
            # Original label start from 1, we shift it to 0
            self.labels.append(int(imagelabels[i-1])-1)
            self.image_path.append( os.path.join(root, 'jpg', 'image_{:05d}.jpg'.format(i)) )


        self.labels = np.array(self.labels)

        new_img_path = []
        new_img_labels = []
        if is_train:
            if shots != -1:
                self.image_path = np.array(self.image_path)
                for c in range(self.num_classes):
                    ids = np.where(self.labels == c)[0]
                    random.seed(seed)
                    random.shuffle(ids)
                    count = 0
                    new_img_path.extend(self.image_path[ids[:shots]])
                    new_img_labels.extend([c for i in range(shots)])
                self.image_path = new_img_path
                self.labels = new_img_labels

    def __getitem__(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.labels[index]

    def __len__(self):
        return len(self.labels)
