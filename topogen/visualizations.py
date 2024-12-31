import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from gtda.homology._utils import _postprocess_diagrams
from gtda.plotting import plot_diagram, plot_point_cloud
from plotly import graph_objects as go

from PIL import Image
from IPython.display import Image as IPImage

def save_fig_dgm(dgm, filename):
    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,1), np.inf, True)[0]
    fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,1)))
    fig.write_image(filename)

def save_fig_pc(pointcloud, filename):
    fig = go.Figure(plot_point_cloud(point_cloud_true))
    fig.write_image(filename)

def save_gen_imgs(data, recon_batch_0, recon_batch_t, epoch, eval_type, step=None, img_size=28, n_imgs=32):
    if step is None: filename = f'figures_epoch_{epoch}_{eval_type}.png'
    else: filename = f'figures_epoch_{epoch}_step_{step}_{eval_type}.png'

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

def plot_pc_gif(point_cloud, x1, x2, y1, y2):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='b')
    #plt.xlabel('X')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.close(fig)
    return fig

def save_animation(point_clouds, test_name, x1, x2, y1, y2):
    # Create a list of figures for each point cloud
    figures = [plot_pc_gif(point_cloud, x1, x2, y1, y2) for point_cloud in point_clouds]

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
