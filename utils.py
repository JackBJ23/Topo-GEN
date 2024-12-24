import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from gtda.homology._utils import _postprocess_diagrams
from gtda.plotting import plot_diagram, plot_point_cloud
from plotly import graph_objects as go

from PIL import Image
from IPython.display import Image as IPImage

def plot_dgm(dgm, filename):
  dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,1), np.inf, True)[0]
  fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,1)))
  fig.write_image(filename)

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

def plot_pc_gif(point_cloud):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='b')
    #plt.xlabel('X')
    plt.xlim(-10, 20)  # Adjust the limits as per your point cloud data
    plt.ylim(-5, 30)  # Adjust the limits as per your point cloud data
    plt.close(fig)
    return fig

def generate_gif(point_clouds):
    # Create a list of figures for each point cloud
    figures = [plot_pc_gif(point_cloud) for point_cloud in point_clouds]

    # Save each figure as an image and store them in a list
    images = []
    for idx, fig in enumerate(figures):
        fig.savefig(f'point_cloud_{idx}.png', dpi=80)
        images.append(Image.open(f'point_cloud_{idx}.png'))

    # Save the images as a GIF
    images[0].save('point_clouds_evolution.gif', save_all=True, append_images=images[1:], duration=50, loop=0) # 70 for test3

    # Display the GIF
    IPImage('point_clouds_evolution.gif')
