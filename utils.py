import os
from glob import glob
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder


'''
Original : https://github.com/pytorch/examples/tree/master/super_resolution

'''

cropsize = 256
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def get_training_set(upscale_factor,folder):
    root_dir = join("dataset", folder)
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(cropsize, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor,folder):
    root_dir = join("dataset", folder)
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(cropsize, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/model_epoch_*.pth".format(checkpoint_dir))]
    idxes = [int(os.path.basename(path).split('_')[2].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    latest_checkpoint = os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(max_idx))
    print(" [*] Found latest checkpoint: {}".format(latest_checkpoint))
    return latest_checkpoint, max_idx