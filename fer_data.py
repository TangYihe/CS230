# Data loading & pre-processing related utils 

from logging.config import valid_ident
from random import shuffle
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import os
from PIL import Image
from deepface import DeepFace

from IPython import embed

class fer_dataset(Dataset):
    def __init__(self, csv_path, usage=None, transforms=None, one_hot=False) -> None:
        '''
        Creates pytorch dataset from fer2013 csv file specified in input. 
        Optional input param usage to select training/validation/testing data in csv file (must have a header of 'Usage')
        '''
        super().__init__()
        print(usage)

        self.raw_data = pd.read_csv(csv_path)
        self.pixels = self.raw_data['pixels'].to_list()

        # in FER13 each image in csv is a string of pixel values, we process here to get the images
        self.images_arr = np.array([self.pixel_to_image(pixel_str) for pixel_str in self.pixels]).reshape(-1, 48, 48)
        self.images = np.array([Image.fromarray(np.uint8(self.images_arr[i])) for i in range(len(self.images_arr))], dtype=object)
        self.labels = np.array(self.raw_data['emotion'])

        self.usage = usage
        if usage:
            self.usage_sel = list(self.raw_data['Usage'] == usage)
            self.images = self.images[self.usage_sel]
            self.labels = self.labels[self.usage_sel]

        self.transforms = transforms
        
        self.one_hot = one_hot


    def pixel_to_image(self, pixel_str):
        return [int(i) for i in pixel_str.split(' ')]

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(img)

        return (np.array(img), label)

    def __len__(self):
        print(self.labels.shape)
        return len(self.labels)


def create_fer_dataloaders(csv_path, batch_size, train_transforms=None, val_transforms=None):
    print('-------Collecting Training Dataset--------')
    train_dataset = fer_dataset(csv_path, usage='Training', transforms=train_transforms, one_hot=True)
    print('-------Collecting Validation Dataset--------')
    val_dataset = fer_dataset(csv_path, usage='PublicTest', transforms=val_transforms)
    print('-------Collecting Test Dataset--------')
    test_dataset = fer_dataset(csv_path, usage='PrivateTest', transforms=val_transforms)

    print('-------Generating Dataloader--------')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=os.cpu_count() - 1)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=os.cpu_count() - 1)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=os.cpu_count() - 1)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    train_transforms = torchvision.transforms.Compose([
        T.Resize(48 + int(.1*48)),  
        T.RandomResizedCrop(48, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(48),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    train_loader, val_loader, test_loader = create_fer_dataloaders('./dataset/fer2013.csv', 32, train_transforms, val_transforms)

    torch.save(train_loader, './dataset/fer_train_32.pt')
    torch.save(val_loader, './dataset/fer_val_32.pt')
    torch.save(test_loader, './dataset/fer_test_32.pt')


