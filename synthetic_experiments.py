import numpy as np
from gph import ripser_parallel
import sys

# Import utils
import numpy as np
from gtda.homology._utils import _postprocess_diagrams

# To generate dataset
from sklearn import datasets

# Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotly import graph_objects as go
import plotly.io as pio
from gtda.plotting import plot_diagram, plot_point_cloud
from PIL import Image
from IPython.display import Image as IPImage

#used this: https://persim.scikit-tda.org/en/latest/notebooks/distances.html
import persim
import tadasets
import ripser
#other packages:
import math
import scipy
import torch
import random

from topo_functions import *
from utils import plot_dgm, generate_gif

def loss_bottleneck01(point_cloud, dgm_true):
  dgm = get_dgm(point_cloud, 1)
  l_topo0, got_loss0 = loss_bottleneck0(point_cloud, dgm, dgm_true)
  l_topo1, got_loss1 = loss_bottleneck1(point_cloud, dgm, dgm_true)
  if got_loss0==1 or got_loss1==1: return l_topo0 + l_topo1, l_topo0.item() + l_topo1.item()
  # only if did not get losses from the previous functions:
  return loss_push0(point_cloud, dgm), l_topo0.item() + l_topo1.item()

# saves:
# i) initial true point cloud, initial true persistence diagram, initial learnable point cloud
# f) final point cloud, final PD of point cloud, loss evolution, video of the point cloud evolution
def synthetic_test(point_cloud, point_cloud_true, num_steps, num_save, lr, test_name):
  # plot initial true point cloud:
  fig = go.Figure(plot_point_cloud(point_cloud_true))
  fig.write_image(f'{test_name}_ini_true_pointcloud.png')
  # plot initial true PD:
  dgm_true = get_dgm(point_cloud_true, 1)
  plot_dgm(dgm_true, f'{test_name}_ini_true_diagram.png')
  # plot initial learnable point cloud:
  fig = go.Figure(plot_point_cloud(point_cloud))
  fig.write_image(f'{test_name}_ini_pointcloud.png')
  # plot initial PD of the learnable point cloud:
  dgm = get_dgm(point_cloud, 1)
  plot_dgm(dgm, f'{test_name}_ini_diagram.png')

  point_cloud_true = torch.tensor(point_cloud_true, dtype=torch.float32)
  point_cloud = torch.tensor(point_cloud, dtype=torch.float32, requires_grad = True)

  point_clouds = [point_cloud.detach().numpy()]
  losses = []
  xs = []
  optimizer = torch.optim.Adam([point_cloud], lr=lr)

  print("Training...")
  for i in range(num_steps):
      optimizer.zero_grad()
      loss, lossitem = loss_bottleneck01(point_cloud, dgm_true)
      loss.backward()
      optimizer.step()

      if i % 5 == 0 or i == num_steps - 1:
        losses.append(lossitem)
        xs.append(i)

      if i % num_save == 0 or i == num_steps - 1: 
        point_clouds.append(np.copy(point_cloud.detach().cpu().numpy()))
        print(f"Iteration {i}/{num_steps}, Loss: {loss.item()}")

  print("Training ended")
  # save PD of final point cloud:
  plot_dgm(get_dgm(point_clouds[-1], 1), f'{test_name}_final_diagram.png')
  # save final point cloud:
  fig = go.Figure(plot_point_cloud(point_clouds[-1]))
  fig.write_image(f'{test_name}_final_pointcloud.png')
  # save loss evolution:
  plt.plot(xs, losses)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.savefig(f'{test_name}_loss_evolution.png')
  # save video of evolution of the point cloud:
  generate_gif(point_clouds, test_name)
  print(f"Test {test_name} done!")

def test1():
  # First, generate a snythetic ground truth point cloud:
  point_cloud_true = np.array([[5.,5.], [10., 10.], [20.0, 6.0]])
  # Second, manually create the initial point cloud:
  point_cloud = np.zeros((64,2))
  r1 = 0.5
  for i in range(10):
    point_cloud[i][0] = random.uniform(-r1, r1)
    point_cloud[i][1] = random.uniform(-r1, r1)
    point_cloud[i+10][0] = random.uniform(-r1, r1)+10.
    point_cloud[i+10][1] = random.uniform(-r1, r1)
    point_cloud[i+20][0] = random.uniform(-r1, r1)
    point_cloud[i+20][1] = random.uniform(-r1, r1)+20
    point_cloud[i+30][0] = random.uniform(-r1, r1)+30
    point_cloud[i+30][1] = random.uniform(-r1, r1)+30
  for i in range(24):
    point_cloud[i+40][0] = random.uniform(-r1, r1)+10
    point_cloud[i+40][1] = random.uniform(-r1, r1)-25
  synthetic_test(point_cloud, point_cloud_true, 15000, 50, 0.01, 'test_1', loss_bottleneck01)

def test2():
  point_cloud_true = np.zeros((128,2))
  r1 = 0.3
  for i in range(30):
    point_cloud_true[i][0] = random.uniform(-r1, r1)
    point_cloud_true[i][1] = random.uniform(-r1, r1)
  for i in range(30, 50):
    point_cloud_true[i][0] = random.uniform(-r1, r1)+10.
    point_cloud_true[i][1] = random.uniform(-r1, r1)
  for i in range(50,80):
    point_cloud_true[i][0] = random.uniform(-r1, r1)-5.
    point_cloud_true[i][1] = random.uniform(-r1, r1)+4.
  for i in range(80,128):
    point_cloud_true[i][0] = random.uniform(-r1, r1)+8.
    point_cloud_true[i][1] = random.uniform(-r1, r1)+13.

  point_cloud = np.zeros((64,2))
  r1 = 0.4
  for i in range(30):
    point_cloud[i][0] = random.uniform(-r1, r1)
    point_cloud[i][1] = random.uniform(-r1, r1)
  for i in range(34):
    point_cloud[i+10][0] = random.uniform(-r1, r1)+10.
    point_cloud[i+10][1] = random.uniform(-r1, r1)+5.
  synthetic_test(point_cloud, point_cloud_true, 2500, 25, 0.05, 'test_2', loss_bottleneck01)

def test3():
  point_cloud_true = tadasets.dsphere(d=1, n=100, noise=0.0) * 5.
  #initial point cloud: 2 lines with added noise
  point_cloud = np.zeros((64,2))
  r1 = 0.1
  for i in range(32):
    point_cloud[i][0] = random.uniform(-r1, r1)
    point_cloud[i][1] = float(i)*0.7 + random.uniform(-r1, r1)
    point_cloud[i+32][0] = random.uniform(-r1, r1) + 5. + float(i) * 0.2
    point_cloud[i+32][1] = float(i)*0.9 + random.uniform(-r1, r1)
  synthetic_test(point_cloud, point_cloud_true, 7500, 50, 0.1, 'test_3', loss_bottleneck01)

if __name__ == "__main__":
  # Test 1: The learnable point cloud starts with 5 clusters, and the reference point cloud has 3 clusters
  test1()
  print("Test 1 done.")
  # Test 2: The learnable point cloud starts with 2 clusters, and the reference point cloud has 4 clusters
  test2()
  print("Test 2 done.")
  # Test 3: The learnable point cloud starts as 2 lines, and the reference point cloud is a circle
  test3()
  print("Test 3 done.")
