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

# Import topological functions and model
from topogen import get_dgm, topo_losses, plot_gen_imgs
from models import VAE

# Standard loss of VAE
def loss_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') #recon_x: reconstructed batch of imgs, x: real batch of imgs
    KLD = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
    # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014. https://arxiv.org/abs/1312.6114
    return BCE, KLD

# Loss of TopoVAEs (Standard + Topolosses)
def loss_topovae(recon_x, x, mu, logvar, dgm_true, topo_weights, deg=1, pers0_delta=0.001, pers1_delta=0.001, dsigma0_scale=0.05, dsigma1_scale=0.05,
                density_sigma=0.2, density_scale=0.002, density_maxrange=35., density_npoints=30, device="cpu"):
    # Standard loss:
    BCE, KLD = loss_vae(recon_x, x, mu, logvar)
    # Topological loss:
    topo_loss = topo_losses(recon_x, x, topo_weights, deg, dgm_true, device, pers0_delta, pers1_delta, dsigma0_scale, dsigma1_scale,
                density_sigma, density_scale, density_maxrange, density_npoints)
    
    return BCE, KLD, BCE + KLD + topo_loss

# Evaluate model0 (normal VAE) and model1 (TopoVAE):
def evaluate(model0, model1, val_loader, epoch, type_eval, device):
  model0.eval()
  model1.eval()
  running_loss0 = 0.
  running_loss1 = 0.
  with torch.no_grad():
      for batch_idx, (data, _) in enumerate(val_loader):
        data = data.to(device)
        #model0
        recon_batch0, mean, log_var = model0(data)
        BCE, _ = loss_vae(recon_batch0, data, mean, log_var)
        running_loss0 += BCE.item()
        #model1
        recon_batch1, mean, log_var = model1(data)
        # No need to compute topoloss here, only need BCE for comparison:
        BCE, _ = loss_vae(recon_batch1, data, mean, log_var)
        running_loss1 += BCE.item()
        if batch_idx == 0: plot_gen_imgs(data.cpu(), recon_batch0.cpu(), recon_batch1.cpu(), epoch, type_eval)

  return running_loss0/len(val_loader), running_loss1/len(val_loader)

# Train and compare model0 (normal VAE) and model1 (TopoVAE):
def train(model0, model1, optimizer0, optimizer1, train_loader, val_loader, dgms_batches, device, args):
  # Losses saved once per epoch:
  train_losses0 = []
  train_losses1 = []
  val_losses0 = []
  val_losses1 = []

  # Losses saved at all training steps, for plotting the loss evolution with more detail:
  train_losses0_all = []
  train_losses1_all = []

  for epoch in range(args.n_epochs):
      model0.train()
      model1.train()
      running_loss0 = 0.
      running_loss1 = 0.
      for batch_idx, (data, _) in enumerate(train_loader):
          data = data.to(device)
          dgm_true = dgms_batches[batch_idx] # Get the pre-computed persistence diagram of the true batch, to avoid computation time
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
          BCE, _, loss1 = loss_topovae(recon_batch1, data, mean, log_var, dgm_true, args.topo_weights, args.deg, args.pers0_delta, args.pers1_delta, args.dsigma0_scale, args.dsigma1_scale, args.density_sigma, args.density_scale, args.density_maxrange, args.density_npoints, device)
          loss1.backward()
          optimizer1.step()
          running_loss1 += BCE.item()
          train_losses1_all.append(BCE.item())
          print("step", batch_idx)

          if batch_idx % args.n_plot == 0: plot_gen_imgs(data.cpu(), recon_batch0.cpu(), recon_batch1.cpu(), epoch, 'train', batch_idx)

      print("End of epoch", epoch)
      # Average of losses over one epoch:
      train_losses0.append(running_loss0 / len(train_loader))
      train_losses1.append(running_loss1 / len(train_loader))
      # Evaluate on the evaluation set:
      val_loss0, val_loss1 = evaluate(model0, model1, val_loader, epoch, 'eval', device)
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
  plt.savefig('BCElosses_train_all_steps.png')

  # Plot losses and validation losses over epochs:
  if args.n_epochs > 1:
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
      plt.savefig('BCElosses_train_val_epochs.png')
  else:
      print(f"Average training BCE loss over 1 epoch for VAE: {train_losses0}; for TopoVAE: {train_losses1}".)
      print(f"Average validation BCE loss after 1 epoch for VAE: {val_losses0}; for TopoVAE: {val_losses1}".)

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
    parser = argparse.ArgumentParser(description="Train and evaluate a generative model with topological regularizers.")
    parser.add_argument('--n_latent', type=int, default=10, help="Latent dimension of the VAE.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training the model.")
    parser.add_argument('--n_epochs', type=int, default=2, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_plot', type=int, default=50, help="Interval (in training steps) at which generated images are saved.")
    parser.add_argument('--deg', type=int, default=1, choices=[0, 1], help="Homology degree used. 1 is the more general option.")
    parser.add_argument('--topo_weights', type=parse_topo_weights, default=[10., 10., 10., 10., 0., 0., 0.], help="7-element vector of floats for topology weights (e.g., '0.1,0.2,0.3,0.4,0.5,0.6,0.7')")
    parser.add_argument('--save_models', type=str, default="n", choices=["y", "n"], help="Select y for saving the models after training, and n for not saving them.")
    # Hyperparameters for some topological functions (reference values by default):
    parser.add_argument('--pers0_delta', type=float, default=0.001)
    parser.add_argument('--pers1_delta', type=float, default=0.001)
    parser.add_argument('--dsigma0_scale', type=float, default=0.05)
    parser.add_argument('--dsigma1_scale', type=float, default=0.05)
    parser.add_argument('--density_sigma', type=float, default=0.2)
    parser.add_argument('--density_scale', type=float, default=0.002)
    parser.add_argument('--density_maxrange', type=float, default=35.)
    parser.add_argument('--density_npoints', type=int, default=30)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
  args = load_config()
  torch.manual_seed(args.seed)
  print("Topological weights:", args.topo_weights)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device:", device)
  model0 = VAE(args.n_latent).to(device)
  model1 = VAE(args.n_latent).to(device)
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

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False) # Set shuffle=False to pre-compute persistence diagrams for all batches only once before training
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  print(f"Sizes: Training set: {len(train_dataset)}; Validation set: {len(val_dataset)}; Test set: {len(test_dataset)}")

  # Pre-compute persistence diagrams:
  print("Pre-computing persistence diagrams...")
  dgms_batches = []
  for step, (data, _) in enumerate(train_loader):
    dgms_batches.append(get_dgm(data.view(data.size(0), -1), 1))

  print("Training...")
  model0, model1 = train(model0, model1, optimizer0, optimizer1, train_loader, val_loader, dgms_batches, device, args)
  if args.save_models == "y":
      torch.save(model0.state_dict(), "model0_weights.pth")
      torch.save(model1.state_dict(), "model1_weights.pth")
      print("Weights of VAE and TopoVAE saved.")
  print("Testing...")
  test_loss0, test_loss1 = evaluate(model0, model1, test_loader, args.n_epochs, 'test', device)
  print("Test losses:", test_loss0, test_loss1)
  print("Done!")