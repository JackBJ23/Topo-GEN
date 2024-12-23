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

from topo_functions import get_dgm, topo_losses
from models import VAE
from utils import plot_dgm, plot_gen_imgs

def loss_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') #recon_x: reconstructed batch of imgs, x: real batch of imgs
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014. https://arxiv.org/abs/1312.6114
    return BCE, KLD

# loss of TopoVAEs (Standard + Topoloss of degree 0 + Topoloss of degree 1)
def loss_topovae(recon_x, x, mu, logvar, dgm, dgm_true, args):
    # Standard loss:
    BCE, KLD = loss_vae(recon_x, x, mu, logvar)
    # Topological loss:
    topo_loss = topo_losses(recon_x, x, dgm, dgm_true, args)
    return BCE, KLD, BCE + KLD + topo_loss

# Train and compare model0 (normal VAE) and model2 (some TopoVAE):
def evaluate(model0, model1, val_loader, epoch, type_eval):
  model0.eval()
  model1.eval()
  running_loss0 = 0.
  running_loss1 = 0.
  with torch.no_grad():
      for batch_idx, (data, _) in enumerate(val_loader):
        #model0
        recon_batch0, mean, log_var = model0(data)
        BCE, _ = loss_vae(recon_batch0, data, mean, log_var)
        running_loss0 += BCE.item()
        #model1
        recon_batch1, mean, log_var = model1(data)
        BCE, _ = loss_vae(recon_batch1, data, mean, log_var)
        running_loss1 += BCE.item()
        if batch_idx == 0: plot_gen_imgs(data, recon_batch0, recon_batch1, epoch, type_eval) # batch_idx set as -1: means it is validation

  return running_loss0/len(val_loader), running_loss1/len(val_loader)

def train(model0, model1, optimizer0, optimizer1, train_loader, val_loader, dgms_batches, args, device):
  # Losses saved once per epoch:
  train_losses0 = []
  train_losses1 = []
  val_losses0 = []
  val_losses1 = []

  # Losses saved in all training steps to view a more detailed evolution:
  train_losses0_all = []
  train_losses1_all = []

  for epoch in range(args.n_epochs):
      model0.train()
      model1.train()
      running_loss0 = 0.
      running_loss1 = 0.
      for batch_idx, (data, _) in enumerate(train_loader):
          dgm_true = dgms_batches[batch_idx]
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
          BCE, _, loss1 = loss_topovae(recon_batch1, data, mean, log_var, dgm, dgm_true, args)
          loss1.backward()
          optimizer1.step()
          running_loss1 += BCE.item()
          train_losses1_all.append(BCE.item())

          if batch_idx % args.n_plot == 0: plot_gen_imgs(data, recon_batch0, recon_batch1, epoch, 'train', batch_idx)

      print("End of epoch", epoch)
      # Average of losses over one epoch:
      train_losses0.append(running_loss0 / len(train_loader))
      train_losses1.append(running_loss1 / len(train_loader))
      # Evaluate on the evaluation set:
      val_loss0, val_loss1 = evaluate(model0, model1, val_loader, epoch, 'eval')
      val_losses0.append(val_loss0)
      val_losses1.append(val_loss1)

  # Training ended
  # Plot losses over all iterations: (for the purposes of this work, we only focus on BCE loss, but KLD loss can also be added)
  plt.figure() 
  plt.plot(np.arange(len(train_losses0_all)), train_losses0_all, label='VAE0')
  plt.plot(np.arange(len(train_losses1_all)), train_losses1_all, label='TopoVAE')
  plt.xlabel("Iteration")
  plt.ylabel("BCE loss")
  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig('BCElosses_train_all.png')

  # Plot losses and validation losses over epochs:
  plt.figure()
  plt.plot(np.arange(len(train_losses0)), train_losses0, label='VAE0, train')
  plt.plot(np.arange(len(train_losses1)), train_losses1, label='TopoVAE, train')
  plt.plot(np.arange(len(val_losses0)), val_losses0, label='VAE0, val')
  plt.plot(np.arange(len(val_losses1)), val_losses1, label='TopoVAE, val')
  plt.xticks(ticks=np.arange(0, len(train_losses0)), labels=np.arange(0, len(train_losses0)))
  plt.xlabel("Epoch")
  plt.ylabel("BCE loss")
  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig('BCElosses_train_val.png')

  return model0, model1

def parse_topo_weights(value):
    try:
        weights = [float(x) for x in value.split(",")]
        if len(weights) != 7:
            raise argparse.ArgumentTypeError("topo_weights must be a 7-element vector of floats.")
        return weights
    except ValueError:
        raise argparse.ArgumentTypeError("topo_weights must contain valid floats.")

def load_config():
    parser = argparse.ArgumentParser(description='Train and evaluate a generative model with topological regularizers.')
    parser.add_argument('--n_latent', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_plot', type=int, default=50)
    parser.add_argument('--topo_weights', type=parse_topo_weights, default=[10., 10., 10., 10., 0., 0., 0.], help="7-element vector of floats for topology weights (e.g., '0.1,0.2,0.3,0.4,0.5,0.6,0.7')")
    # Hyperparameters for some topological functions (default values already given):
    parser.add_argument('--pers0_delta', type=float, default=0.001)
    parser.add_argument('--pers1_delta', type=float, default=0.001)
    parser.add_argument('--dsigma0_scale', type=float, default=0.05)
    parser.add_argument('--dsigma1_scale', type=float, default=0.05)
    parser.add_argument('--density_sigma', type=float, default=0.2)
    parser.add_argument('--density_scale', type=float, default=0.002)
    parser.add_argument('--density_maxrange', type=float, default=35.)
    parser.add_argument('--density_npoints', type=int, default=30)
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint file to load model weights')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
  # Hyperparameters:
  args = load_config()
  torch.manual_seed(args.seed)
  print("Weights", args.topo_weights)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model0 = VAE(args.n_latent)
  model1 = VAE(args.n_latent)
  model1.load_state_dict(model0.state_dict())
  optimizer0 = optim.Adam(model0.parameters(), lr=args.learning_rate)
  optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
  model0.train()
  model1.train()

  # Download datasets:
  transform = transforms.ToTensor()
  full_train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
  train_size = int(0.8 * len(full_train_dataset))
  val_size = len(full_train_dataset) - train_size
  train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
  test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  print(f"Sizes: Training set: {len(train_dataset)}; Validation set: {len(val_dataset)}; Test set: {len(test_dataset)}")

  # Pre-compute persistence diagrams:
  print("Pre-computing persistence diagrams...")
  dgms_batches = []
  for step, (data, _) in enumerate(train_loader):
    dgms_batches.append(get_dgm(data.view(data.size(0), -1), 1))

  print("Training...")
  model0, model1 = train(model0, model1, optimizer0, optimizer1, train_loader, val_loader, dgms_batches, args, device)
  print("Testing...")
  test_loss0, test_loss1 = evaluate(model0, model1, test_loader, args.n_epochs, 'test')
  print("Test losses:", test_loss0, test_loss1)
  print("Done.")
