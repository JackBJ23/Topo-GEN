"""
A set of functions for visualization that can be helpful when working with topological regularizers (either with generative
models, or in experiments with learnable point clouds in 2D). Includes functions for visualizing:
- Persistence diagrams: plot_fig_dgm
- The effect of topological regularizers on 2D point clouds: plot_fig_pc, generate_animation
- The performance of topology-informed models compared to their non-regularized counterparts: plot_gen_imgs, plot_iter_losses, plot_epoch_losses.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from gtda.homology._utils import _postprocess_diagrams
from gtda.plotting import plot_diagram, plot_point_cloud
from plotly import graph_objects as go

from PIL import Image
from IPython.display import Image as IPImage

def plot_fig_dgm(dgm, filename):
    """
    Plots a persistence diagram and saves it in a file.
    Args:
        dgm (dict): Persistence diagram. Can be obtained using the function get_dgm(), see topo_functions.py.
        filename: File name to save the plot.
    """
    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,1), np.inf, True)[0]
    fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,1)))
    fig.write_image(filename)
 
def plot_fig_pc(pointcloud, filename):
    """
    Plots a point cloud and saves it in a file.
    Args:
        pointcloud (np.ndarray): Point cloud, shape (number of points, dimension of each point).
        filename: File name to save the plot.
    """
    fig = go.Figure(plot_point_cloud(pointcloud))
    fig.write_image(filename)

def _plot_pc_gif(point_cloud, x1, x2, y1, y2):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='b')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.close(fig)
    return fig

def generate_animation(point_clouds, test_name, x1, x2, y1, y2):
    """
    Generates and saves an animation of the evolution of a point cloud. Helpful for visualizing the impact of topological 
    regularizers on 2D point clouds.
    Args:
        point_clouds (list): List of the point_clouds, where point_clouds[i] is the i-th point cloud, expected to be
        a np.ndarray of shape (number of points, dimension of each point).
        test_name (str): Name of the test.
        x1, x2, y1, y2 (float): Limits of the figures (min-x, max-x, min-y, max-y, respectively).
    """
    # Create a list of figures for each point cloud
    figures = [_plot_pc_gif(point_cloud, x1, x2, y1, y2) for point_cloud in point_clouds]

    gif_path = f'{test_name}_point_clouds_evolution.gif'
    # Save each figure as an image and store them in a list
    images = []
    file_paths = []
    for idx, fig in enumerate(figures):
        file_path = f'point_cloud_{idx}.png'
        fig.savefig(file_path, dpi=80)
        images.append(Image.open(file_path))
        file_paths.append(file_path)

    # Save the images as a GIF
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=50, loop=0)
    for file_path in file_paths: os.remove(file_path)
    IPImage(gif_path)

def plot_gen_imgs(data, recon_batch_0, recon_batch_t, epoch, eval_type, step=None, img_size=28, n_imgs=32, modelname="VAE", filename=None, show=False):
    """
    Plots and saves ground truth images (grayscale or RGB), images reconstructed by a standard generative model and images reconstructed by a topology-informed model.
    Args:
        data (torch.Tensor): Original data batch. Shape (batch size, num channels, height, width).
        recon_batch_0 (torch.Tensor): Reconstructed batch from the standard model. Shape (batch size, num channels, height, width).
        recon_batch_t (torch.Tensor): Reconstructed batch from topology-informed model. Shape (batch size, num channels, height, width).
        epoch (int): Current epoch number, used for generating titles.
            Note: If plotting during training (evaluation type is 'train'), use epoch = current epoch (0, 1, ...). 
            If plotting during validation/test, use epoch = number of epochs completed (1, 2, ...).
        eval_type (str): Evaluation type ('train', 'val', 'test').
        step (int): Training step index. Ony needed if eval_type='train'.
        img_size (int): Image height in pixels (assuming square image). Defaults to 28 for FashionMNIST.
        n_imgs (int): Number of images to display in the grid.
        modelname (str): Model name (e.g., VAE, GAN, DiffusionModel, etc.)
        filename (str): File name to save the plot. If None, the figure is not saved.
        show (bool): Whether to display the plot.
    """
    if eval_type == 'train':
        suptitle = f'True and generated images at epoch {epoch}{", step " + str(step) + " " if step is not None else ""}({eval_type})'
    else: 
        suptitle = f'True and generated images after {epoch} training epochs ({eval_type})'

    # Create grids for each dataset
    grid_data = torchvision.utils.make_grid(data[:n_imgs], nrow=8, normalize=True)
    grid_recon_0 = torchvision.utils.make_grid(recon_batch_0[:n_imgs], nrow=8, normalize=True)
    grid_recon_t = torchvision.utils.make_grid(recon_batch_t[:n_imgs], nrow=8, normalize=True)
    # Convert tensors to numpy arrays for plotting
    grid_data = grid_data.cpu().numpy().transpose((1, 2, 0))
    grid_recon_0 = grid_recon_0.cpu().numpy().transpose((1, 2, 0))
    grid_recon_t = grid_recon_t.cpu().numpy().transpose((1, 2, 0))
    # Plot the three grids next to each other
    plt.figure(figsize=(15, 5))
    plt.suptitle(suptitle, fontsize=16)
    # Left: Ground truth data
    plt.subplot(1, 3, 1)
    plt.imshow(grid_data)
    plt.axis('off')
    plt.title("True")
    # Middle: Reconstructed batch from the standard model
    plt.subplot(1, 3, 2)
    plt.imshow(grid_recon_0)
    plt.axis('off')
    plt.title(modelname)
    # Right: Reconstructed batch from the topology-informed model
    plt.subplot(1, 3, 3)
    plt.imshow(grid_recon_t)
    plt.axis('off')
    plt.title(f"Topo{modelname}")
    plt.tight_layout()
    if filename is not None: plt.savefig(filename)
    if show: plt.show()
    plt.close()

def plot_iter_losses(train_losses0_all, train_losses1_all, steps_per_item=1, modelname="VAE", metric="BCE", filename=None, show=False):
    """
    Plots and saves training losses for a standard generative model and a topology-informed model across iterations.
    Args:
        train_losses0_all (list): Losses for the standard model.
        train_losses1_all (list): Losses for the topology-informed model.
        steps_per_item (int): interval (in training steps) at which every measure of the loss is saved.
        modelname (str): Model name (e.g., VAE, GAN, DiffusionModel, etc.).
        metric (str): Metric used (e.g., BCE, KLD, MSE, etc.).
        filename (str): Filename for saving the plot. If None, the plot is not saved.
        show (bool): Whether to display the plot.
    """
    plt.figure()
    iterations = np.arange(len(train_losses0_all)) * steps_per_item
    plt.plot(iterations, train_losses0_all, label=modelname, color='blue')
    plt.plot(iterations, train_losses1_all, label=f"Topo{modelname}", color='orange')
    plt.xlabel("Iteration")
    plt.ylabel(f"{metric} loss")
    plt.title(f"Training {metric} Losses Over Iterations")
    plt.legend(loc='upper right')
    plt.tight_layout()
    if filename is not None: plt.savefig(filename)
    if show: plt.show()
    plt.close()

def plot_epoch_losses(train_losses0, train_losses1, val_losses0, val_losses1, modelname="VAE", metric="BCE", filename=None, show=False):
    """
    Plots training and validation losses for a standard generative model and a topology-informed model over epochs (i.e., one value per epoch).
    Args:
        train_losses0 (list): Training losses for the standard model.
        train_losses1 (list): Training losses for the topology-informed model.
        val_losses0 (list): Validation losses for the standard model.
        val_losses1 (list): Validation losses for the topology-informed model.
        modelname (str): Model name (e.g., VAE, GAN, DiffusionModel, etc.).
        metric (str): Metric used (e.g., BCE, KLD, MSE, etc.).
        filename (str): File path to save the plot. If None, the plot is not saved.
        show (bool): Whether to display the plot.
    """
    epochs = np.arange(len(train_losses0))
    plt.figure()
    plt.plot(epochs, train_losses0, label=f'{modelname} (Train)', marker='o', color='blue')
    plt.plot(epochs, train_losses1, label=f'Topo{modelname} (Train)', marker='o', color='orange')
    plt.plot(epochs, val_losses0, label=f'{modelname} (Val)', marker='x', linestyle='dashed', color='blue')
    plt.plot(epochs, val_losses1, label=f'Topo{modelname} (Val)', marker='x', linestyle='dashed', color='orange')
    plt.xticks(ticks=epochs, labels=epochs)
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric} loss")
    plt.title(f"Training and Validation {metric} Losses Over Epochs")
    plt.legend(loc='upper right')
    plt.tight_layout()
    if filename is not None: plt.savefig(filename)
    if show: plt.show()
    plt.close()
