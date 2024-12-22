import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt
from gtda.homology._utils import _postprocess_diagrams
from gtda.plotting import plot_diagram, plot_point_cloud
from plotly import graph_objects as go

def plot_dgm(dgm):
  dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,1), np.inf, True)[0]
  fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,1)))
  fig.show()

def plot_gen_imgs(data, recon_batch_0, recon_batch_t, epoch, eval_type, step=None):
    if step==None: filename = f'figures_epoch_{epoch}_{eval_type}.png'
    else: filename = f'figures_epoch_{epoch}_step_{step}_{eval_type}.png'

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
    plt.savefig(filename)
