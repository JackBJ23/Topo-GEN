import sys
import torch
import numpy as np
import random
import tadasets
import matplotlib.pyplot as plt
from gtda.plotting import plot_point_cloud
from plotly import graph_objects as go

from topogen import get_dgm, loss_bottleneck0, loss_bottleneck1, loss_push0, plot_dgm, generate_gif

# Loss function for the point cloud:
def loss_bottleneck01(point_cloud, dgm_true, device):
  dgm = get_dgm(point_cloud, 1)
  l_topo0, got_loss0 = loss_bottleneck0(point_cloud, dgm, dgm_true, device)
  l_topo1, got_loss1 = loss_bottleneck1(point_cloud, dgm, dgm_true, device)
  if got_loss0==1 or got_loss1==1: return l_topo0 + l_topo1, l_topo0.item() + l_topo1.item()
  # If did not get losses from the previous functions: use loss_push0, which adds a small perturbation to the point cloud that "pushes" points away from each other
  # Empirically, this leads, in the following iterations, to obtain losses from the bottleneck functions 
  return loss_push0(point_cloud, dgm), l_topo0.item() + l_topo1.item()

# Function for running a synthetic test with the bottleneck functions. This function saves images of:
# i) initial true point cloud, initial true persistence diagram, initial learnable point cloud
# f) final point cloud, final persistence diagram of point cloud, loss evolution, and a video of the point cloud evolution during training
def synthetic_test(point_cloud, point_cloud_true, device, num_steps=2000, num_save=50, lr=0.001, test_name="test", x1=-10., x2=40., y1=-40., y2=40.):
  # Plot initial true point cloud:
  fig = go.Figure(plot_point_cloud(point_cloud_true))
  fig.write_image(f'{test_name}_ini_true_pointcloud.png')
  # Plot its persistence diagram:
  dgm_true = get_dgm(point_cloud_true, 1)
  plot_dgm(dgm_true, f'{test_name}_ini_true_diagram.png')
  # Plot initial learnable point cloud:
  fig = go.Figure(plot_point_cloud(point_cloud))
  fig.write_image(f'{test_name}_ini_pointcloud.png')
  # Plot its persistence diagram:
  dgm = get_dgm(point_cloud, 1)
  plot_dgm(dgm, f'{test_name}_ini_diagram.png')

  point_cloud_true = torch.tensor(point_cloud_true, dtype=torch.float32, device=device)
  point_cloud = torch.tensor(point_cloud, dtype=torch.float32, requires_grad = True, device=device)

  point_clouds = [point_cloud.detach().cpu().numpy()]
  losses = []
  xs = []
  optimizer = torch.optim.Adam([point_cloud], lr=lr)

  print("Training...")
  for i in range(num_steps):
      optimizer.zero_grad()
      loss, lossitem = loss_bottleneck01(point_cloud, dgm_true, device)
      loss.backward()
      optimizer.step()

      if i % 5 == 0 or i == num_steps - 1:
        losses.append(lossitem)
        xs.append(i)

      if i % num_save == 0 or i == num_steps - 1: 
        point_clouds.append(np.copy(point_cloud.detach().cpu().numpy()))
      if i % 100 == 0 or i == num_steps - 1:
        print(f"Iteration {i}/{num_steps}, Loss: {lossitem}")

  print("Training ended")
  # save persistence diagram of final point cloud:
  plot_dgm(get_dgm(point_clouds[-1], 1), f'{test_name}_final_diagram.png')
  # Save final point cloud:
  fig = go.Figure(plot_point_cloud(point_clouds[-1]))
  fig.write_image(f'{test_name}_final_pointcloud.png')
  # Save loss evolution:
  plt.figure()
  plt.plot(xs, losses)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.savefig(f'{test_name}_loss_evolution.png')
  # Save video of evolution of the point cloud:
  generate_gif(point_clouds, test_name, x1, x2, y1, y2)
  print(f"Test {test_name} done!")

def test1(device):
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
  synthetic_test(point_cloud, point_cloud_true, device, 15000, 50, 0.01, 'test_1')

def test2(device):
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
  synthetic_test(point_cloud, point_cloud_true, device, 2500, 25, 0.05, 'test_2')

def test3(device):
  point_cloud_true = tadasets.dsphere(d=1, n=100, noise=0.0) * 5.
  # Initial point cloud: 2 lines with added noise
  point_cloud = np.zeros((64,2))
  r1 = 0.1
  for i in range(32):
    point_cloud[i][0] = random.uniform(-r1, r1)
    point_cloud[i][1] = float(i)*0.7 + random.uniform(-r1, r1)
    point_cloud[i+32][0] = random.uniform(-r1, r1) + 5. + float(i) * 0.2
    point_cloud[i+32][1] = float(i)*0.9 + random.uniform(-r1, r1)
  synthetic_test(point_cloud, point_cloud_true, device, 7500, 50, 0.1, 'test_3')

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Test 1: The learnable point cloud starts with 5 clusters, and the reference point cloud has 3 clusters
  test1(device)
  print("Test 1 done.")
  # Test 2: The learnable point cloud starts with 2 clusters, and the reference point cloud has 4 clusters
  test2(device)
  print("Test 2 done.")
  # Test 3: The learnable point cloud starts as 2 lines, and the reference point cloud is a circle
  test3(device)
  print("Test 3 done.")
