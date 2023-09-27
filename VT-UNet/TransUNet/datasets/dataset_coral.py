import os
import random

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def vis_test(image, label, before_im, before_label):
    im_out = np.concatenate((before_im[0], image[0]), axis=0)
    for i in range(image.shape[0]):
        if i == 0:
            continue
        im_out = np.concatenate((im_out, np.concatenate((before_im[i, :, :], image[i, :, :]), axis=0)), axis=1)

    im_out = np.concatenate((im_out, np.concatenate(((before_label[0, :, :] > 0) * 255, (label[0, :, :] > 0) * 255), axis=0)), axis=1)
    cv2.imshow("test", im_out.astype(np.uint8))
    cv2.waitKey()


def random_rot_flip(image, label):
    # before_im = image.copy()
    # before_label = label.copy()

    k = np.random.randint(0, 4)
    for i in range(image.shape[0]):
        image[i, :, :] = np.rot90(image[i, :, :], k).copy()
    label[0, :, :] = np.rot90(label[0, :, :], k)

    axis = np.random.randint(0, 2)
    for i in range(image.shape[0]):
        image[i, :, :] = np.flip(image[i, :, :], axis=axis).copy()
    label[0, :, :] = np.flip(label[0, :, :], axis=axis).copy()

    # vis_test(image, label, before_im, before_label)

    return image, label


def random_rotate(image, label):
    # before_im = image.copy()
    # before_label = label.copy()

    angle = np.random.randint(-20, 20)
    for i in range(image.shape[0]):
        image[i, :, :] = ndimage.rotate(image[i, :, :], angle, order=0, reshape=False)
    label[0, :, :] = ndimage.rotate(label[0, :, :], angle, order=0, reshape=False)
    
    # vis_test(image, label, before_im, before_label)

    return image, label


def add_gaussian_noise(image, label):
    mean = 0
    std = random.uniform(0, 0.5)
    rand_grid = np.random.rand(image[0].shape[0], image[0].shape[1]) * 255
    noise = (rand_grid * std + mean).astype(np.uint8)
    old_ims = image.copy()
    for i in range(image.shape[0]):
        image[i] += noise
    return image, label



def resize_inputs(output_size, sample):
    image = sample['image']
    label = sample['label']
    _, x, y = image.shape

    if x != output_size[0] or y != output_size[1]:
        new_image = np.zeros((image.shape[0], output_size[0], output_size[1]))
        for i in range(image.shape[0]):
            new_image[i, :, :] = zoom(image[i, :, :], (output_size[0] / x, output_size[1] / y),
                                      order=0).copy()
        new_label = zoom(label[0], (output_size[0] / x, output_size[1] / y), order=0)
        image = torch.from_numpy(new_image.astype(np.float32))
        label = torch.from_numpy(new_label.astype(np.float32)).unsqueeze(0)

        sample = {'image': image, 'label': label.long()}
    return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if random.random() > 0.5:
            add_gaussian_noise(image, label)

        sample = {'image': image, 'label': label}
        return sample


class Coral_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "val":
            name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + f"/train/{name}"
            data = np.load(data_path, allow_pickle=True)
            image = data['image']
            label = (data['label'] > 0) * 1

        else:
            name = self.sample_list[idx].strip('\n')
            filepath = f"{self.data_dir}/{name}"
            data = np.load(filepath, allow_pickle=True)
            image = data['image']
            label = (data['label'] > 0) * 1

        sample = {'image': image, 'label': label}
        old_sample = sample['image'].copy()
        if self.transform:
            sample = self.transform(sample)

        # Used for visualising augs.
        # path = "/Users/rob/University/University/Year 3/Coral/CoralGrowth/report_stuff/images/augmentations"
        # cv2.imwrite(f"{path}/{name}.png", np.concatenate((old_sample[2], sample['image'][2]), axis=1))
        output_size = [224, 224]
        sample = resize_inputs(output_size, sample)
        sample['coral_name'] = self.sample_list[idx].strip('\n')
        return sample
