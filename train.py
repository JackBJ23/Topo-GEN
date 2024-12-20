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

def plotdgm(dgm):
  dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,1), np.inf, True)[0]
  fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,1)))
  fig.show()

def plot_imgs(data, recon_batch_0, recon_batch_t, epoch, type):
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
    plt.savefig(f'figures_epoch_{epoch}_step_{type}.png')
    plt.show()

def loss_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') #recon_x: reconstructed batch of imgs, x: real batch of imgs
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014. https://arxiv.org/abs/1312.6114
    return BCE, KLD

# loss of TopoVAEs (Standard + Topoloss of degree 0 + Topoloss of degree 1)
def loss_topovae(recon_x, x, mu, logvar, dgms, dgms_true, w_topo0, w_topo1):
    # Standard loss:
    BCE, KLD = loss_vae(recon_x, x, mu, logvar)
    loss = BCE + KLD
    # Topological loss:
    l_topo0, got_loss0 = d_bottleneck0(recon_x, dgms, dgms_true) #loss of degree 0
    l_topo1, got_loss1 = d_bottleneck1(recon_x, dgms, dgms_true) #loss of degree 1

    if got_loss0==1: loss += l_topo0 * w_topo0
    if got_loss1==1: loss += l_topo1 * w_topo1
    return BCE, KLD, l_topo0, l_topo1, loss

# Train and compare model0 (normal VAE) and model2 (some TopoVAE):

def evaluate(model0, model1, val_loader, dgms_batches, epoch, type_eval, w_topo0, w_topo1):
  model0.eval()
  model1.eval()
  running_loss0 = 0.
  running_loss1 = 0.
  with torch.no_grad():
      for batch_idx, (data, _) in enumerate(val_loader):
        dgms_true = dgms_batches[batch_idx]
        #model0
        recon_batch0, mean, log_var = model0(data)
        BCE, _ = loss_vae(recon_batch0, data, mean, log_var)
        running_loss0 += BCE.item()
        #model1
        recon_batch1, mean, log_var = model1(data)
        dgm = get_dgm(recon_batch1.view(data.size(0), -1), 1)
        BCE, _, _, _, _ = loss_topovae(recon_batch1, data, mean, log_var, dgm, dgms_true, w_topo0, w_topo1)
        running_loss1 += BCE.item()
        if batch_idx == 0: plot_imgs(data, recon_batch0, recon_batch1, epoch, type_eval) # batch_idx set as -1: means it is validation

  return running_loss0/len(val_loader), running_loss1/len(val_loader)

def train(model0, model1, optimizer0, optimizer1, n_epochs, train_loader, val_loader, dgms_batches, w_topo0, w_topo1):
  # Losses saved once per epoch:
  train_losses0 = []
  train_losses1 = []
  val_losses0 = []
  val_losses1 = []

  # Losses saved in all training steps to view a more detailed evolution:
  train_losses0_all = []
  train_losses1_all = []

  for epoch in range(n_epochs):
      model0.train()
      model1.train()
      running_loss0 = 0.
      running_loss1 = 0.
      for batch_idx, (data, _) in enumerate(train_loader):
          dgms_true = dgms_batches[batch_idx]
          optimizer0.zero_grad()
          optimizer1.zero_grad()

          # model0: VAE
          recon_batch0, mean, log_var = model0(data)
          BCE, KLD = loss_vae(recon_batch0, data, mean, log_var)
          loss0 = BCE + KLD
          loss0.backward()
          optimizer0.step()
          running_loss0 += BCE.item()
          train_losses0_all.append(BCE.item())

          # model1: TopoVAE
          recon_batch1, mean, log_var = model1(data)
          dgm = get_dgm(recon_batch1.view(data.size(0), -1), 1)
          BCE, _, _, _, loss1 = loss_topovae(recon_batch1, data, mean, log_var, dgm, dgms_true, w_topo0, w_topo1)
          loss1.backward()
          optimizer1.step()
          running_loss1 += BCE.item()
          train_losses1_all.append(BCE.item())

          if batch_idx == 0: plot_imgs(data, recon_batch0, recon_batch1, epoch, 'train')

      print("End of epoch", epoch)
      # Average of losses over one epoch:
      train_losses0.append(running_loss0 / len(train_loader))
      train_losses1.append(running_loss1 / len(train_loader))
      # Evaluate on the evaluation set:
      val_loss0, val_loss1 = evaluate(model0, model1, val_loader, dgms_batches, epoch, 'eval', w_topo0, w_topo1)
      val_losses0.append(val_loss0)
      val_losses1.append(val_loss1)

  # Training ended
  # Save losses over all iterations: (for the purposes of this work, we only focus on BCE loss, but KLD loss can also be added)
  plt.plot(np.arange(len(train_losses0_all)), train_losses0_all, label='VAE0')
  plt.plot(np.arange(len(train_losses1_all)), train_losses1_all, label='TopoVAE')
  plt.xlabel("Iteration")
  plt.ylabel("BCE loss")
  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig('BCElosses_train_all.png')

  # Save losses and validation losses over epochs:
  plt.plot(np.arange(len(train_losses0)), train_losses0, label='VAE0, train')
  plt.plot(np.arange(len(train_losses1)), train_losses1, label='TopoVAE, train')
  plt.plot(np.arange(len(val_losses0)), val_losses0, label='VAE0, val')
  plt.plot(np.arange(len(val_losses1)), val_losses1, label='TopoVAE, val')
  plt.xlabel("Epoch")
  plt.ylabel("BCE loss")
  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig('BCElosses_train_val.png')

  return model0, model1

if __name__ == "__main__":
  ## hyperparameters:
  w_topo0 = 15.
  w_topo1 = 15.
  n_epochs = 1
  n_showplots = 25
  n_latent = 10
  seed = 123
  batch_size = 64
  img_size = 28 * 28
  torch.manual_seed(seed)

  model0 = VAE(n_latent)
  model1 = VAE(n_latent)
  model1.load_state_dict(model0.state_dict())

  optimizer0 = optim.Adam(model0.parameters(), lr=5e-4)
  optimizer1 = optim.Adam(model1.parameters(), lr=5e-4)
  model0.train()
  model1.train()

  # Download datasets:
  transform = transforms.ToTensor()
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
    if step==0: print("shape:", data.shape, "pts", points_np.shape)
    dgm2 = ripser_parallel(points_np, maxdim=1, return_generators=True)
    dgms_batches.append(dgm2)

  print("Training...")
  model0, model1 = train(model0, model1, optimizer0, optimizer1, n_epochs, train_loader, val_loader, dgms_batches, w_topo0, w_topo1)
  print("Testing...")
  test_loss0, test_loss1 = evaluate(model0, model1, test_loader, dgms_batches, n_epochs, 'test', w_topo0, w_topo1)
  print("Test losses:", test_loss0, test_loss1)
  print("Done.")
