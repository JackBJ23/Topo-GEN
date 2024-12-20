import sys
import os
import math
import random
import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import make_grid, save_image
from plotly import graph_objects as go
from IPython.display import Image, display

# TDA libraries
import ripser
import persim
import tadasets
from gph import ripser_parallel
from gtda.homology._utils import _postprocess_diagrams
from gtda.plotting import plot_diagram, plot_point_cloud

from topo_functions import get_dgm, d_bottleneck0, d_bottleneck1, dsigma0, dsigma1, loss_density, loss_persentropy0, loss_persentropy1
from models import VAE

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

######

def plotdgm(dgm):
  dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,1), np.inf, True)[0]
  fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,1)))
  fig.show()

#######

seed = 1
batch_size = 128
epochs = 30
log_interval = 50
torch.manual_seed(seed)
img_size = 28*28

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the VAE with a desired latent dimension
latent_dim = 32  # You can adjust this as needed
vae = VAE(latent_dim)

"""Standard loss fctn and regularizers:"""

def loss_vae0(recon_x, x, mu, logvar, t):
    global losses, losses2, lossestopo, lossestopo2
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') #recon_x: fake batch of imgs, x: real batch of imgs

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if t==0:
      losses.append(BCE.item())
      losses2.append(KLD.item())
    else:
      lossestopo.append(BCE.item())
      lossestopo2.append(KLD.item())

    return BCE + KLD

fids = []

def loss_fctn1(recon_x, x, mu, logvar, dgm, dgm2, w_topo0): #bottleneck0
    w_vae = 1.
    ## normal loss:
    loss = w_vae * loss_vae0(recon_x, x, mu, logvar, 1)
    ## compute topological loss:
    l_topo0, got_loss0 = d_bottleneck0(recon_x, dgm, dgm2)

    if got_loss0==1: loss += l_topo0 * w_topo0

    return loss

def loss_fctn2(recon_x, x, mu, logvar, dgm, dgm2, w_topo0, w_topo1): #bottleneck0+1
    w_vae = 1.
    ## normal loss:
    loss = w_vae * loss_vae0(recon_x, x, mu, logvar, 1)
    ## compute topological loss:
    l_topo0, got_loss0 = d_bottleneck0(recon_x, dgm, dgm2) #loss of degree 0
    l_topo1, got_loss1 = d_bottleneck1(recon_x, dgm, dgm2) #loss of degree 1

    if got_loss0==1: loss += l_topo0 * w_topo0
    if got_loss1==1: loss += l_topo1 * w_topo1

    return loss

def loss_fctn3(recon_x, x, mu, logvar, dgm, dgm2, w_topo0, w_topo1, delta): #pers entropy0+1. weights: 0.5, 0.5: does not learn. 0.3, 0.1: good
    w_vae = 1.
    ## normal loss:
    loss = w_vae * loss_vae0(recon_x, x, mu, logvar, 1)
    ## compute topological loss:
    l_topo0, got_loss0 = loss_persentropy0(recon_x, dgm, dgm2, delta) #loss of degree 0
    l_topo1, got_loss1 = loss_persentropy1(recon_x, dgm, dgm2, delta) #loss of degree 1

    if got_loss0==1: loss += l_topo0 * w_topo0
    if got_loss1==1: loss += l_topo1 * w_topo1

    return loss

def loss_fctn4(recon_x, x, mu, logvar, dgm, dgm2, w_topo0, w_topo1, delta): #pers entropy0+dsigma1. 0.5, 0.5: does not learn. 0.1, 0.3: relatively good.
    w_vae = 1.
    ## normal loss:
    loss = w_vae * loss_vae0(recon_x, x, mu, logvar, 1)
    ## compute topological loss:
    l_topo0, got_loss0 = loss_persentropy0(recon_x, dgm, dgm2, delta) #loss of degree 0
    l_topo1, got_loss1 = dsigma1(recon_x, x.view(-1, img_size), dgm, dgm2) #loss of degree 1

    if got_loss0==1: loss += l_topo0 * w_topo0
    if got_loss1==1: loss += l_topo1 * w_topo1

    return loss

def loss_fctn5(recon_x, x, mu, logvar, dgm, dgm2, w_topo0, w_topo1): # density0+dsigma1
    w_vae = 1.
    ## normal loss:
    loss = w_vae * loss_vae0(recon_x, x, mu, logvar, 1)
    ## compute topological loss:
    l_topo0 = loss_density(recon_x, x.view(-1, img_size), dgm, dgm2, 0.1, 0.0005, 15., 30, False) #loss of degree 0. loss_density(point_cloud, point_cloud2, dgm, dgm2, sigma, scale, maxrange, npoints, plot)
    l_topo1, got_loss1 = dsigma1(recon_x, x.view(-1, img_size), dgm, dgm2) #loss of degree 1

    loss += l_topo0 * w_topo0
    if got_loss1==1: loss += l_topo1 * w_topo1

    return loss

def loss_fctn6(recon_x, x, mu, logvar, dgm, dgm2, w_topo0, w_topo1): # dsigma0+dsigma1
    w_vae = 1.
    ## normal loss:
    loss = w_vae * loss_vae0(recon_x, x, mu, logvar, 1)
    ## compute topological loss:
    l_topo0 = dsigma0(recon_x, x.view(-1, img_size), dgm, dgm2) #loss of degree 0
    l_topo1, got_loss1 = dsigma1(recon_x, x.view(-1, img_size), dgm, dgm2) #loss of degree 1

    loss += l_topo0 * w_topo0
    if got_loss1==1: loss += l_topo1 * w_topo1

    return loss

def plot_imgs(data, recon_batch_0, recon_batch_t, epoch, step):
    # Reshape tensors for visualization
    data = data.reshape(-1, 1, 28, 28)
    recon_batch_0 = recon_batch_0.reshape(-1, 1, 28, 28)
    recon_batch_t = recon_batch_t.reshape(-1, 1, 28, 28)

    # Create grids for each dataset
    grid_data = torchvision.utils.make_grid(data[:32], nrow=8, normalize=True)
    grid_recon_0 = torchvision.utils.make_grid(recon_batch_0[:32], nrow=8, normalize=True)
    grid_recon_t = torchvision.utils.make_grid(recon_batch_t[:32], nrow=8, normalize=True)

    # Convert tensors to numpy arrays for plotting
    grid_data = grid_data.cpu().numpy().transpose((1, 2, 0))
    grid_recon_0 = grid_recon_0.cpu().numpy().transpose((1, 2, 0))
    grid_recon_t = grid_recon_t.cpu().numpy().transpose((1, 2, 0))

    # Plot the three grids next to each other
    plt.figure(figsize=(15, 5))

    # Left: Data
    plt.subplot(1, 3, 1)
    plt.imshow(grid_data)
    plt.axis('off')
    plt.title("True")

    # Middle: Reconstructed Batch 0 (standard VAE)
    plt.subplot(1, 3, 2)
    plt.imshow(grid_recon_0)
    plt.axis('off')
    plt.title("VAE")

    # Right: Reconstructed Batch from TopoVAE
    plt.subplot(1, 3, 3)
    plt.imshow(grid_recon_t)
    plt.axis('off')
    plt.title("TopoVAE")

    plt.tight_layout()
    plt.savefig(f'figures_epoch_{epoch}_step_{step}.png')
    plt.show()

# Download dataset

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Sizes: Training set: {len(train_dataset)}; Validation set: {len(val_dataset)}; Test set: {len(test_dataset)}")

# Pre-compute persistence diagrams:
dgms_batches = []
for step, (data, _) in enumerate(train_loader):
  data = data.view(data.size(0), -1)
  points_np = data.view(-1, img_size).numpy()
  if step==0: print("shape:", data.shape, "pts", points_np)
  dgm2 = ripser_parallel(points_np, maxdim=1, return_generators=True)
  dgms_batches.append(dgm2)

# Train and compare model0 (normal VAE) and model2 (some TopoVAE):

## hyperparameters:
n_epochs = 10
n_showplots = 25
n_latent = 10

## create 2 models with same parameters:
seed = 515
torch.manual_seed(seed)
model0 = VAE(n_latent)
torch.manual_seed(seed)
model2 = VAE(n_latent)

optimizer0 = optim.Adam(model0.parameters(), lr=1e-3)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
model0.train()
model2.train()
losses = []
losses2 = []
lossestopo = []
lossestopo2 = []

for epoch in range(1):
    for batch_idx, (data, _) in enumerate(train_loader):

        if batch_idx%n_showplots==0 and batch_idx>0:
          plt.plot(np.arange(len(losses)), losses, label='Conv-VAE')
          plt.plot(np.arange(len(lossestopo)), lossestopo, label='Conv-TopoVAE')
          plt.xlabel("Iteration")
          plt.ylabel("BCE loss")
          plt.legend(loc='upper right')
          plt.show()
          plt.plot(np.arange(len(losses2)), losses2, label='Conv-VAE')
          plt.plot(np.arange(len(lossestopo2)), lossestopo2, label='Conv-TopoVAE')
          plt.xlabel("Iteration")
          plt.ylabel("KLD loss")
          plt.legend(loc='upper right')
          plt.show()

        if batch_idx % n_showplots != 0 or batch_idx==0:
          if batch_idx%50==0: print(batch_idx)
          #get dgm2:
          dgm2 = dgms_batches[batch_idx]

          #update the 2 models:
          optimizer0.zero_grad()
          optimizer2.zero_grad()

          #model0
          recon_batch, mean, log_var = model0(data)
          loss = loss_vae0(recon_batch, data, mean, log_var, 0)
          loss.backward()
          optimizer0.step()
          #model2
          recon_batch, mean, log_var = model2(data)
          dgm = get_dgm(recon_batch.view(data.size(0), -1), 1)
          ## replace the next line by the topo-loss of choice (or combination of topo-losses):
          loss = loss_fctn2(recon_batch, data, mean, log_var, dgm, dgm2, 15., 15.)
          # loss_fctn2(recon_batch, data, mean, log_var, dgm, dgm2, 5., 5.)
          # loss_fctn3(recon_batch, data, mean, log_var, dgm, dgm2, 5., 5., 0.1)
          # loss_fctn4(recon_batch, data, mean, log_var, dgm, dgm2, 10., 10., 0.1)
          # loss_fctn5(recon_batch, data, mean, log_var, dgm, dgm2, 5., 5.)
          loss.backward()
          optimizer2.step()

          if batch_idx % n_showplots == 1 and batch_idx > 100:
            print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)}")
            print("Input: real data (trained on)")
            with torch.no_grad():
                print("Real batch, VAE0, TopoVAE:")
                recon_batch_0, _, _ = model0(data)
                recon_batch_t, _, _ = model2(data)
                plot_imgs(data, recon_batch_0, recon_batch_t, epoch, batch_idx)

        else: #ie batch_idx % n_showplots == 0 and >0:
            print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)}")
            print("Input: new data (not trained on) and input random latent vectors:")

            with torch.no_grad():
                print("Real batch, VAE0, TopoVAE:")
                recon_batch_0, _, _ = model0(data)
                recon_batch_t, _, _ = model2(data)
                plot_imgs(data, recon_batch_0, recon_batch_t, epoch, batch_idx)
