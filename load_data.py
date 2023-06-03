# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torchvision import transforms
#from scipy import misc
from PIL import Image
import numpy as np
import imageio
import torch
import os

to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)
    print(RAW_norm.shape)
    
    return RAW_norm


class LoadData(Dataset):

    def __init__(self, dataset_dir, dataset_size, dslr_scale, test=True):

        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'jpg_img')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'pynet_fullres_cropped_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'pynet_fullres_cropped_jpg')

        self.dataset_size = dataset_size
        self.scale = dslr_scale
        self.test = test

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))
        
        dslr_image = imageio.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg")) #jpg changed to png
        dslr_image = np.asarray(dslr_image)
        
        im = Image.fromarray(dslr_image)
        size = tuple((np.array(im.size) *self.scale / 2.0).astype(int))
        dslr_image = np.float32(np.array(im.resize(size, Image.BICUBIC))/255.0)
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        
        #dslr_image = misc.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg"))
        #dslr_image = np.asarray(dslr_image)
        #dslr_image = np.float32(misc.imresize(dslr_image, self.scale / 2.0)) / 255.0
        
        #print('dslr shape: ',dslr_image.shape)
        #print('raw_image shape: ',raw_image.shape)
        
        return raw_image, dslr_image


class LoadVisualData(Dataset):

    def __init__(self, data_dir, size, scale, full_resolution=True):

        self.raw_dir = os.path.join(data_dir,'test','full_resolution')
        #self.raw_dir = os.path.join(data_dir)
        self.dataset_size = size
        self.scale = scale
        self.full_resolution = full_resolution
        self.test_images = os.listdir(self.raw_dir)
        self.image_height = 1440
        self.image_width = 1984
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, self.test_images[idx])))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = raw_image[0:self.image_height, 0:self.image_width, :]
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image
