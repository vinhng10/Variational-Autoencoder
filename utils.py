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

def vae_loss(x, recon_x, mu, logsigma, beta):
    """ Variational autoencoder loss. """
    # Binary cross entropy loss for reconstructed image:
    BCE = F.binary_cross_entropy(recon_x.view(recon_x.shape[0], -1), \
            x.view(x.shape[0], -1), reduction='sum')
    # KL-divergence between trained latent space and
    # standard normal distribution:
    DKL = -0.5 * torch.sum(1 + 2*logsigma - torch.pow(mu, 2) - \
            torch.pow(torch.exp(logsigma), 2))
    return BCE + beta * DKL

def train_model(data_loader, device, model=None, epochs=20, lr=0.001,
        weight_decay=0.01, beta=1, step_size=5, gamma=0.1):
    """ Training procedure. """
    # Check if a model is given for further training:
    if not model:
        model = DVAE().to(device)
    # Prepare optimizer and learning rate scheduler:
    optimizer = optim.Adam(model.parameters(), \
            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, \
            step_size=step_size, gamma = gamma)
    # Loop over the dataset multiple times:
    losses = []
    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        for i, x in enumerate(data_loader):
            # Main training phase:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logsigma, z, eps = model(x)
            loss = vae_loss(x, recon_x, mu, logsigma, 1)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # Print progress:
            if (i + 1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item()))
        scheduler.step()
    # Done training:
    return model, losses

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
        img = plt.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img

class Conv2Linear(nn.Module):
    """ Flatten convolution layer to linear layer. """
    def __init(self):
        super(Conv2Linear, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Linear2Conv(nn.Module):
    """ Reshape linear layer to convolution layer. """
    def __init__(self, channels, size):
        super(Linear2Conv, self).__init__()
        self.channels = channels
        self.size = size

    def forward(self, x):
        return x.view(x.shape[0], self.channels, self.size, -1)

class DVAE(nn.Module):
    """ Disintangled Variational Autoencoder. """
    def __init__(self):
        """ Initialize model. """
        super(DVAE, self).__init__()
        # Encoder network:
        self.encoder = nn.Sequential(
                nn.Conv2d(4, 16, 3, 1, 0),          # 4*120*120     -> 16*118*118
                nn.ReLU(),                          # ReLU
                nn.Conv2d(16, 32, 4, 3, 0),         # 16*118*118     -> 32*39*39
                nn.ReLU(),                          # ReLU
                nn.Conv2d(32, 64, 7, 4, 0),         # 32*39*39      -> 64*9*9
                nn.ReLU(),                          # ReLU
                Conv2Linear(),                      # 9*20*20       -> 3600
                nn.Linear(5184, 512),               # 3600          -> 512
                nn.ReLU(),                           # ReLU
                )

        # Latent space:
        self.mu = nn.Linear(512, 20)                # 512       -> 20
        self.logsigma = nn.Linear(512, 20)          # 512       -> 20

        # Decoder network:
        self.decoder = nn.Sequential(
                nn.Linear(20, 512),                 # 20        -> 512
                nn.ReLU(),                          # ReLU
                nn.Linear(512, 5184),
                nn.ReLU(),
                Linear2Conv(64, 9),                 # 3600      -> 9*20*20
                nn.ConvTranspose2d(64, 32, 7, 4, 0),  # 9*20*20   -> 6*58*58
                nn.ReLU(),                          # ReLU
                nn.ConvTranspose2d(32, 16, 4, 3, 0),  # 6*58*58   -> 3*118*118
                nn.ReLU(),                          # ReLU
                nn.ConvTranspose2d(16, 4, 3, 1, 0),  # 3*118*118 -> 4*120*120
                nn.Sigmoid()                        # Sigmoid
                )

    def encode(self, x):
        """ Encode data to latent space. """
        h = self.encoder(x)
        return self.mu(h), self.logsigma(h)

    def reparametrize(self, mu, logsigma):
        """ Reparametrize latent variable. """
        eps = torch.randn(mu.shape, device=mu.device)
        z = mu + torch.exp(logsigma)*eps
        return z, eps

    def decode(self, mu, logsigma):
        """ Reconstruct latent variable to original data. """
        z, eps = self.reparametrize(mu, logsigma)
        recon_x = self.decoder(z)
        return recon_x, z, eps

    def forward(self, x):
        """ Forward pass. """
        mu, logsigma = self.encode(x)
        recon_x, z, eps = self.decode(mu, logsigma)
        return recon_x, mu, logsigma, z, eps

