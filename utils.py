import os
import torch, torchvision
import urllib.request
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def download_images():
    """ Download players' face images. """
    # Read the csv file:
    csv_file = pd.read_csv("players_20.csv")
    # Extract urls to the players' pages:
    ids = csv_file['sofifa_id']
    urls = ['https://cdn.sofifa.org/players/10/20/' + \
            str(i) + '.png' for i in ids]

    # Change the request name to avoid error:
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'whatever')
    # Extract players' name:
    names = csv_file['short_name']
    # Download players images and save to 'images' folder:
    for i in range(len(urls)):
        if i % 1000 == 0:
            print(f'Downloaded {i} images.')
        try:
            opener.retrieve(urls[i], 'images/' + names[i] + '.png')
        except:
            continue

def remove_error_images():
    """ Remove all images that are empty. """
    img_names = os.listdir('images/')
    for name in img_names:
        path = 'images/' + name
        try:
            img = plt.imread(path)
        except:
            os.remove(path)

class EarlyStopping():
    """ Early stopping criterion for training. """
    def __init__(self, tolerance, patience):
        self.tolerance = tolerance
        self.patience = patience

    def stop_criterion(self, losses):
        if (len(losses) <= self.patience):
            return False
        min_loss = min(losses)
        recent_losses = np.array(losses[-self.patience:])
        return all(recent_losses > min_loss + self.tolerance)

class FaceDataset(Dataset):
    """ Customize dataset class for player images. """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_names = os.listdir(root_dir)
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = torch.tolist()
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        img = plt.imread(img_path)[:, :, :3]
        if self.transform:
            img = self.transform(img)
        return img
