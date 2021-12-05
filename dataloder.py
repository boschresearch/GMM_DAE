import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from skimage import transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_type='float32', nch=1, transform=[]):
        self.data_dir = data_dir
        self.transform = transform
        self.nch = nch
        self.data_type = data_type

        lst_data = os.listdir(data_dir)

        self.names = lst_data

    def __getitem__(self, index):
        if self.nch == 1:
            data = plt.imread(os.path.join(self.data_dir, self.names[index]))[:, :, np.newaxis]  #:self.nch]
            data = data[:, :, :1]
        else:
            data = plt.imread(os.path.join(self.data_dir, self.names[index]))[:, :, :self.nch]

        if data.dtype == np.uint8:
            data = data / 255.0

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.names)


class Normalize(object):
    def __call__(self, data):
        data = 2 * data - 1
        return data


class ToTensor(object):
    def __call__(self, data):
        data = data.transpose((2, 0, 1)).astype(np.float32)
        return torch.from_numpy(data)


class ToNumpy(object):
    def __call__(self, data):
        if data.ndim == 3:
            data = data.to('cpu').detach().numpy().transpose((1, 2, 0))
        elif data.ndim == 4:
            data = data.to('cpu').detach().numpy().transpose((0, 2, 3, 1))
            print(np.shape(data))
        return data


class Denormalize(object):
    def __call__(self, data):
        return (data + 1) / 2


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        data = transform.resize(data, (new_h, new_w))
        return data
