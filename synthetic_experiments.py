import os
import torch
import numpy as np
import random
import tadasets
import matplotlib.pyplot as plt

from topogen import get_dgm, loss_push0, TopologicalLoss, save_fig_dgm, save_fig_pc, generate_animation

def get_loss(point_cloud, point_cloud_true, dgm_true, topo_loss):
  """
  Computes the topological loss function for the point cloud. 
  """
  dgm = get_dgm(point_cloud, 1)
  loss, gotloss = topo_loss.compute_loss(point_cloud, point_cloud_true, dgm, dgm_true)
  if gotloss: return loss, loss.item()
  # If did not get topological loss: use loss_push0, which adds a small perturbation to the point cloud that "pushes" points away from each other
  # Empirically, this leads, in the following iterations, to obtain losses from the bottleneck functions
  return loss_push0(point_cloud, dgm), loss.item()

def create_point_cloud(centers, cluster_sizes, r):
    """
    Creates a point cloud in 2D with multiple clusters.
    Args:
        - centers (np.ndarray): Array with the centers of the clusters. 
        - cluster_sizes (list): Number of points per cluster. 
        - r (float): Radius for the random perturbation around each cluster center.
    Returns:
        - np.ndarray: Generated point cloud of shape (num_points, 2).
    """
    point_cloud = []
    for center, size in zip(centers, cluster_sizes):
        points = center + np.random.uniform(-r, r, size=(size, 2))
        point_cloud.append(points)
    point_cloud = np.vstack(point_cloud)
    return point_cloud

def synthetic_test(point_cloud, point_cloud_true, topo_weights=[1.,1.,0.,0.,0.,0.,0.], num_steps=2000, lr=0.001, test_name="synthetic_test", device="cpu", num_save=50, x1=-10., x2=40., y1=-40., y2=40.):
  """
  Function for running a synthetic test with an initial random point cloud in 2D, a ground truth point cloud, and an arbitrary topological loss. 
  Helpful for visualizing the impact of topological regularizers on 2D data.
  Args:
    - point_cloud (np.ndarray): Learnable point cloud, shape (number of points, dimension of each point).
    - point_cloud_true (np.ndarray): Ground truth point cloud, shape (number of points, dimension of each point).
    - topo_weights (list): List of weights for the total loss (created with the class TopologicalLoss). Each weight is associated with a topological regularizer.
    - num_steps (int): Number of training steps.
    - lr (float): Learning rate.
    - test_name (str): Test name for creating a folder and saving the generated files.
    - device: Device for performing the computations.
    - num_save: Interval (in training steps) at which the point cloud coordinates are saved, enabling the creation of the final animation.
    - x1, x2, y1, y2 (floats): the window limits for the animation (x-min, x-max, y-min, y-max, respectively).
  Output:
    - Figures of the initial true point cloud, its persistence diagram, initial learnable point cloud, and its persistence diagram.
    - Figures of the final learnable point cloud, its final persistence diagram, and loss evolution.
    - An animation of the point cloud evolution during training.
  """
  # Create folder to save the results of the test:
  os.makedirs(test_name, exist_ok=True)
  # Save figure of true point cloud:
  save_fig_pc(point_cloud_true, f'{test_name}/ini_true_pointcloud.png')
  # Save figure of its persistence diagram:
  dgm_true = get_dgm(point_cloud_true, 1)
  save_fig_dgm(dgm_true, f'{test_name}/ini_true_diagram.png')
  # Save figure of initial learnable point cloud:
  save_fig_pc(point_cloud, f'{test_name}/ini_pointcloud.png')
  # Save figure of its persistence diagram:
  dgm = get_dgm(point_cloud, 1)
  save_fig_dgm(dgm, f'{test_name}/ini_diagram.png')

  point_cloud_true = torch.tensor(point_cloud_true, dtype=torch.float32, device=device)
  point_cloud = torch.tensor(point_cloud, dtype=torch.float32, requires_grad=True, device=device)

  # Set the topological loss:
  topo_loss = topo_loss = TopologicalLoss(topo_weights)

  point_clouds = [point_cloud.detach().cpu().numpy()] # For the final animation
  losses = []
  xs = []
  optimizer = torch.optim.Adam([point_cloud], lr=lr)
  print("Training...")
  for i in range(num_steps):
      optimizer.zero_grad()
      loss, lossitem = get_loss(point_cloud, point_cloud_true, dgm_true, topo_loss)
      loss.backward()
      optimizer.step()
    
      if i % 5 == 0 or i == num_steps - 1: # Save loss values for final plot of loss evolution
        losses.append(lossitem)
        xs.append(i)
      
      if i % num_save == 0 or i == num_steps - 1: # Save point clouds for final animation
        point_clouds.append(np.copy(point_cloud.detach().cpu().numpy()))
      
      if (i+1) % 100 == 0 or i+1 == num_steps: # Show once every 100 steps the current loss
        print(f"Iteration {i+1}/{num_steps}, Loss: {lossitem}")

  print("Training ended")
  # Save persistence diagram of the final point cloud:
  save_fig_dgm(get_dgm(point_clouds[-1], 1), f'{test_name}/final_diagram.png')
  # Save final point cloud:
  save_fig_pc(point_clouds[-1], f'{test_name}/final_pointcloud.png')
  # Save loss evolution:
  plt.figure()
  plt.plot(xs, losses)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.savefig(f'{test_name}/loss_evolution.png')
  plt.close()
  # Save animation of the evolution of the point cloud:
  generate_animation(point_clouds, x1, x2, y1, y2, f'{test_name}/point_clouds_evolution.gif')
  print(f"Test {test_name} done!")

# Test 1: The learnable point cloud starts with 5 clusters, and the reference point cloud has 3 clusters.
def test1(topo_weights=[1.,1.,0.,0.,0.,0.,0.], num_steps=15000):
  # Generate a synthetic ground truth point cloud:
  point_cloud_true = np.array([[5.,5.], [10., 10.], [20.0, 6.0]])
  # Generate the learnable point cloud (5 clusters centered at [0., 0.], [10., 0.], etc.):
  point_cloud = create_point_cloud(np.array([[0.,0.], [10.,0.], [0.,20.], [30.,30.], [10.,-25.]]), [10,10,10,10,24], 0.5)
  synthetic_test(point_cloud, point_cloud_true, topo_weights, num_steps, 0.01, "test1", num_save=50)

# Test 2: The learnable point cloud starts with 2 clusters, and the reference point cloud has 4 clusters.
def test2(topo_weights=[1.,1.,0.,0.,0.,0.,0.], num_steps=2500):
  point_cloud_true = create_point_cloud(np.array([[0.,0.], [10.,0.], [-5.,4.], [8.,13.]]), [30,20,30,48], 0.3)
  point_cloud = create_point_cloud(np.array([[0.,0.], [10.,5.]]), [30, 34], 0.4)
  synthetic_test(point_cloud, point_cloud_true, topo_weights, num_steps, 0.05, "test2", num_save=25)

# Test 3: The learnable point cloud starts as 2 lines with added noise, and the reference point cloud is a circle.
def test3(topo_weights=[1.,1.,0.,0.,0.,0.,0.], num_steps=7500):
  point_cloud_true = tadasets.dsphere(d=1, n=100, noise=0.0) * 5.
  point_cloud = np.zeros((64,2))
  r1 = 0.1
  for i in range(32):
    point_cloud[i] = np.array([random.uniform(-r1, r1), 0.7 * i + random.uniform(-r1, r1)])
    point_cloud[i+32] = np.array([0.2 * i + random.uniform(-r1, r1) + 5., 0.9 * i + random.uniform(-r1, r1)])
  synthetic_test(point_cloud, point_cloud_true, topo_weights, num_steps, 0.1, "test3", num_save=50)

if __name__ == "__main__":
  test1()
  test2()
  test3()
