import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .utils import noisify_sym


class CIFAR10(Dataset):
    
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    
    def __init__(self, train=True, transform = transforms.ToTensor(), target_transform=None, noise_rate = 0.2):
        self.train = train
        self.transform = transform
        self.train_data = np.zeros((0,3,32,32))
        self.train_labels = np.zeros((0,))
        self.noise_rate = noise_rate    
        if self.train:
            for fentry in self.train_list:
                f = fentry[0]
                file = './data/cifar-10-batches-py/' + f
                fo = open(file, 'rb')
                entry = pickle.load(fo, encoding='latin1')
                self.train_data = np.concatenate((self.train_data, entry['data'].reshape((10000,3,32,32))))
                self.train_labels = np.concatenate((self.train_labels, np.array(entry['labels'])))
            self.train_labels = torch.from_numpy(noisify_sym(self.train_labels.astype('uint8'), noise_rate = self.noise_rate, random_state = 0)[0])
            
        
        else:
            f = self.test_list[0][0]
            file = './data/cifar-10-batches-py/' + f
            fo = open(file, 'rb')
            entry = pickle.load(fo, encoding='latin1')
            self.train_data = np.concatenate((self.train_data, entry['data'].reshape((10000,3,32,32))))
            self.train_labels = torch.from_numpy(np.concatenate((self.train_labels, np.array(entry['labels']))))

        self.train_data = self.train_data.transpose(0,2,3,1)
        

    def __getitem__(self,index):
        
        img, label = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img.astype('uint8'))
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index
    
    def __len__(self):
        
        return self.train_data.shape[0]
