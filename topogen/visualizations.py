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
    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,1), np.inf, True)[0]
    fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,1)))
    fig.write_image(filename)

def plot_gen_imgs(data, recon_batch_0, recon_batch_t, epoch, eval_type, step=None, img_size=28, n_imgs=32, modelname="VAE", filename=None, show=False):
    """
    Plots and saves ground truth images, images reconstructed by the standard generative model, and by the topology-informed model.
    Args:
        data (torch.Tensor): Original data batch.
        recon_batch_0 (torch.Tensor): Reconstructed batch from VAE.
        recon_batch_t (torch.Tensor): Reconstructed batch from TopoVAE.
        epoch (int): Current epoch number, used for generating titles.
            Note: If plotting during training (evaluation type is 'train'), use the current epoch (0, 1, ...). 
            If plotting during validation/test, use epoch = number of epochs completed (1, 2, ...).
        eval_type (str): Evaluation type ('train', 'val', 'test').
        step (int, optional): Training step index. If eval_type='train', provide the step.
        img_size (int, optional): Image size (height and width). Defaults to 28 for FashionMNIST.
        n_imgs (int): Number of images to display in the grid.
        modelname (str): Model name (e.g., VAE, GAN, DiffusionModel, etc.)
        filename (str, optional): Filename to save the plot. If None, the figure is not saved.
        show (bool): Whether to display the plot.
    """
    if eval_type == 'train':
        suptitle = f'True and generated images at epoch {epoch}{", step " + str(step) + " " if step is not None else ""}({eval_type})'
    else: 
        suptitle = f'True and generated images after {epoch} training epochs ({eval_type})'

    # Reshape tensors for visualization
    data = data.reshape(-1, 1, img_size, img_size)
    recon_batch_0 = recon_batch_0.reshape(-1, 1, img_size, img_size)
    recon_batch_t = recon_batch_t.reshape(-1, 1, img_size, img_size)

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

    # Middle: Reconstructed Batch 0 (standard Model)
    plt.subplot(1, 3, 2)
    plt.imshow(grid_recon_0)
    plt.axis('off')
    plt.title(modelname)

    # Right: Reconstructed Batch from TopoModel
    plt.subplot(1, 3, 3)
    plt.imshow(grid_recon_t)
    plt.axis('off')
    plt.title(f"Topo{modelname}")
    plt.tight_layout()
    if filename is not None: plt.savefig(filename)
    if show: plt.show()
    plt.close()

def plot_training_losses(train_losses0_all, train_losses1_all, filename=None, show=False):
    """
    Plots training losses for VAE and TopoVAE across all iterations.
    Args:
        train_losses0_all (list): Losses for the normal VAE.
        train_losses1_all (list): Losses for the TopoVAE.
        filename (str, optional): Filename for saving the plot. If None, the plot will not be saved.
        show (bool, optional): Whether to display the plot. 
    """
    plt.figure()
    plt.plot(np.arange(len(train_losses0_all)), train_losses0_all, label='VAE0')
    plt.plot(np.arange(len(train_losses1_all)), train_losses1_all, label='TopoVAE')
    plt.xlabel("Iteration")
    plt.ylabel("BCE loss")
    plt.title("Training Losses Over Iterations")
    plt.legend(loc='upper right')
    plt.tight_layout()
    if filename is not None: plt.savefig(filename)
    if show: plt.show()
    plt.close()

def plot_losses_avg_epoch(train_losses0, train_losses1, val_losses0, val_losses1, filename=None, show=False):
    """
    Plots the average training and validation losses for VAE and TopoVAE over epochs.
    Args:
        train_losses0 (list): Average training losses for VAE0 per epoch.
        train_losses1 (list): Average training losses for TopoVAE per epoch.
        val_losses0 (list): Average validation losses for VAE0 per epoch.
        val_losses1 (list): Average validation losses for TopoVAE per epoch.
        filename (str, optional): File path to save the plot. If None, the plot is not saved.
        show (bool, optional): Whether to display the plot.
    """
    epochs = np.arange(len(train_losses0))
    plt.figure()
    plt.plot(epochs, train_losses0, label='VAE0, Train', marker='o')
    plt.plot(epochs, train_losses1, label='TopoVAE, Train', marker='o')
    plt.plot(epochs, val_losses0, label='VAE0, Val', marker='x', linestyle='dashed')
    plt.plot(epochs, val_losses1, label='TopoVAE, Val', marker='x', linestyle='dashed')
    plt.xticks(ticks=epochs, labels=epochs)
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Training and Validation Losses Over Epochs")
    plt.legend(loc='upper right')
    plt.tight_layout()
    if filename is not None: plt.savefig(filename)
    if show: plt.show()
    plt.close()

def plot_fig_pc(pointcloud, filename):
    fig = go.Figure(plot_point_cloud(pointcloud))
    fig.write_image(filename)

def _plot_pc_gif(point_cloud, x1, x2, y1, y2):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='b')
    #plt.xlabel('X')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.close(fig)
    return fig

def generate_animation(point_clouds, test_name, x1, x2, y1, y2):
    """
    Generates and saves an animation of the evolution of a point cloud. Helpful for visualizing the impact of topological 
    regularizers on 2D point clouds.
    Args:
        point_clouds (list): list of the point_clouds, where point_clouds[i] is the i-th point cloud, expected to be
        a np.ndarray of shape (number of points, dimension of each point).
        test_name (str): Name of the test.
        x1, x2, y1, y2: Limits of the figures (min-x, max-x, min-y, max-y, respectively).
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
