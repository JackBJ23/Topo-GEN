import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import logging

# Import topological functions and model
from topogen import get_dgm, TopologicalLoss, plot_gen_imgs, plot_iter_losses, plot_epoch_losses
from model import VAE

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

# Standard loss of VAE
def loss_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') #recon_x: reconstructed batch of images, x: true batch of images
    KLD = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
    # See Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014. https://arxiv.org/abs/1312.6114
    return BCE, KLD

# Evaluate model0 (normal VAE) and model1 (TopoVAE):
def evaluate(model0, model1, val_loader, epoch, eval_type, device):
  model0.eval()
  model1.eval()
  running_loss0 = 0.
  running_loss1 = 0.
  with torch.no_grad():
      for batch_idx, (data, _) in enumerate(val_loader):
        data = data.to(device)
        # model0
        recon_batch0, mean0, log_var0 = model0(data)
        BCE0, _ = loss_vae(recon_batch0, data, mean0, log_var0)
        running_loss0 += BCE0.item()
        # model1
        recon_batch1, mean1, log_var1 = model1(data)
        # No need to compute the topological loss here, only need BCE for comparison (KLD could also be included):
        BCE1, _ = loss_vae(recon_batch1, data, mean1, log_var1)
        running_loss1 += BCE1.item()
        if batch_idx == 0: plot_gen_imgs(data.cpu(), recon_batch0.cpu(), recon_batch1.cpu(), epoch, eval_type, filename=f'imgs_{eval_type}_after_{epoch}_epoch{"s" if epoch!=1 else ""}')

  return running_loss0 / len(val_loader), running_loss1 / len(val_loader)

# Train and compare model0 (normal VAE) and model1 (TopoVAE):
def train(model0, model1, optimizer0, optimizer1, train_loader, val_loader, dgms_batches, device, args):
  # Losses to save once per epoch:
  train_losses0 = []
  train_losses1 = []
  val_losses0 = []
  val_losses1 = []

  # Losses to save at all training steps, for plotting the loss evolution with more detail:
  train_losses0_all = []
  train_losses1_all = []

  # Create the topological loss:
  topo_loss = TopologicalLoss(args.topo_weights, args.deg, args.pers0_delta, args.pers1_delta, args.dsigma0_sigma, args.dsigma1_sigma, args.density_sigma, args.density_scale, args.density_maxrange, args.density_npoints)

  for epoch in range(args.n_epochs):
      model0.train()
      model1.train()
      running_loss0 = 0.
      running_loss1 = 0.
      for batch_idx, (data, _) in enumerate(train_loader):
          data = data.to(device)
          dgm_true = dgms_batches[batch_idx] # Get the pre-computed persistence diagram of the true batch to avoid computation time
          optimizer0.zero_grad()
          optimizer1.zero_grad()

          # model0 (VAE)
          recon_batch0, mean0, log_var0 = model0(data)
          BCE0, KLD0 = loss_vae(recon_batch0, data, mean0, log_var0)
          loss0 = BCE0 + KLD0
          loss0.backward()
          optimizer0.step()
          running_loss0 += BCE0.item()
          train_losses0_all.append(BCE0.item())

          # model1 (TopoVAE)
          recon_batch1, mean1, log_var1 = model1(data)
          BCE1, KLD1 = loss_vae(recon_batch1, data, mean1, log_var1)
          loss1 = BCE1 + KLD1
          topoloss, gotloss = topo_loss.compute_loss(recon_batch1, data, None, dgm_true)
          if gotloss: loss1 = loss1 + topoloss
          loss1.backward()
          optimizer1.step()
          running_loss1 += BCE1.item()
          train_losses1_all.append(BCE1.item())

          if batch_idx % args.n_plot == 0: plot_gen_imgs(data.cpu(), recon_batch0.cpu(), recon_batch1.cpu(), epoch, 'train', batch_idx, filename=f'imgs_train_epoch_{epoch}_step_{batch_idx}')

      logging.info(f"End of epoch {epoch+1}/{args.n_epochs}")
      # Save average of losses over the epoch:
      train_losses0.append(running_loss0 / len(train_loader))
      train_losses1.append(running_loss1 / len(train_loader))
      # Evaluate both models:
      val_loss0, val_loss1 = evaluate(model0, model1, val_loader, epoch+1, 'val', device)
      val_losses0.append(val_loss0)
      val_losses1.append(val_loss1)

  # Training ended
  # Plot and save losses over all training steps: (for the purposes of this work, we only focus on BCE loss, but KLD loss can also be added)
  plot_iter_losses(train_losses0_all, train_losses1_all, filename='BCElosses_train.png')
  # Plot training losses and validation losses over epochs:
  if args.n_epochs > 1:
      plot_epoch_losses(train_losses0, train_losses1, val_losses0, val_losses1, filename='BCElosses_train_val_epochs.png')
  else:
      logging.info(f"Average training BCE loss over 1 epoch for VAE: {train_losses0}; for TopoVAE: {train_losses1}")
      logging.info(f"Average validation BCE loss after 1 epoch for VAE: {val_losses0}; for TopoVAE: {val_losses1}")
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
    parser.add_argument('--n_epochs', type=int, default=2, help="Number of training epochs. 1 or 2 are sufficient for training the VAE on FashionMNIST.")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_plot', type=int, default=50, help="Interval (in training steps) at which generated images are plotted/saved.")
    parser.add_argument('--deg', type=int, default=1, choices=[0, 1], help="Homology degree used. 1 is the more general option.")
    parser.add_argument('--topo_weights', type=parse_topo_weights, default=[10., 10., 10., 10., 0., 0., 0.], help="7-element vector of floats for topology weights (e.g., '0.1,0.2,0.3,0.4,0.5,0.6,0.7')")
    parser.add_argument('--save_models', type=bool, default=False, help="Select True for saving the models after training, False otherwise.")
    # Hyperparameters for topological functions (reference values by default):
    parser.add_argument('--pers0_delta', type=float, default=0.001, help="Controls loss_persentropy0.")
    parser.add_argument('--pers1_delta', type=float, default=0.001, help="Controls loss_persentropy1.")
    parser.add_argument('--dsigma0_sigma', type=float, default=0.05, help="Controls loss_dsigma0.")
    parser.add_argument('--dsigma1_sigma', type=float, default=0.05, help="Controls loss_dsigma1.")
    parser.add_argument('--density_sigma', type=float, default=0.2, help="Controls loss_density.")
    parser.add_argument('--density_scale', type=float, default=0.002, help="Controls loss_density.")
    parser.add_argument('--density_maxrange', type=float, default=35., help="Controls loss_density.")
    parser.add_argument('--density_npoints', type=int, default=30, help="Controls loss_density.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
  args = load_config()
  torch.manual_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using topological weights: {args.topo_weights}. Device: {device}.")
  model0 = VAE(args.n_latent).to(device)
  model1 = VAE(args.n_latent).to(device)
  model1.load_state_dict(model0.state_dict())
  optimizer0 = optim.Adam(model0.parameters(), lr=args.learning_rate)
  optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)

  # Download datasets:
  transform = transforms.ToTensor()
  full_train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
  train_size = int(0.8 * len(full_train_dataset))
  val_size = len(full_train_dataset) - train_size
  train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
  test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

  # Set shuffle=False to pre-compute persistence diagrams for all batches only one time before training
  # If shuffle=True, ground truth persistence diagrams have to be pre-computed before starting each epoch
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
  logging.info(f"Sizes: Training set: {len(train_dataset)}; Validation set: {len(val_dataset)}; Test set: {len(test_dataset)}")

  # Pre-compute persistence diagrams:
  logging.info("Pre-computing persistence diagrams...")
  dgms_batches = []
  for step, (data, _) in enumerate(train_loader):
    dgms_batches.append(get_dgm(data.view(data.size(0), -1), 1))

  logging.info("Training...")
  model0, model1 = train(model0, model1, optimizer0, optimizer1, train_loader, val_loader, dgms_batches, device, args)
  if args.save_models:
      torch.save(model0.state_dict(), "vae_weights.pth")
      torch.save(model1.state_dict(), "topovae_weights.pth")
      logging.info("Weights of VAE and TopoVAE saved in vae_weights.pth and topovae_weights.pth, respectively.")
  logging.info("Testing...")
  test_loss0, test_loss1 = evaluate(model0, model1, test_loader, args.n_epochs, 'test', device)
  logging.info(f"Test losses: {test_loss0}, {test_loss1}.\nDone!")
