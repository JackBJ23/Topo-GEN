"""
Topological Regularizers Overview:
Each topological regularizer computes a measure of dissimilarity between the learnable point cloud's persistence diagram and the ground truth persistence diagram. 
The 7 regularizers are: 
- loss_bottleneck0, loss_bottleneck1: bottleneck distance for homology degree 0/1
- loss_persentropy0, loss_persentropy1: squared difference between persistence entropies for homology degree 0/1
- loss_dsigma0, loss_dsigma1: Reininghaus dissimilarity for homology degree 0/1
- loss_density: difference between the 4SGDE density functions of the two diagrams.

General input arguments:
- Required:
  - point_cloud (torch.Tensor): The learnable point cloud. Shape (number of points, dimensions of each point).
  - point_cloud2 (torch.Tensor): The true point cloud. Shape (number of points, dimensions of each point).
- Optional:
  - dgm (dict): Persistence diagram for the first point cloud. If None, it will be computed.
  - dgm2 (dict): Persistence diagram for the true point cloud. If None, it will be computed.
  - Additional hyperparameters that control the topological functions.

Output:
  - torch.Tensor: The computed loss value as a scalar tensor. 
  - bool: A status flag, True if the loss depends on the learnable point cloud, False otherwise.

Additionally, the topological regularizers are unified in the class TopologicalLoss for their efficient and straightforward combination. See details in class TopologicalLoss.

Note: The function loss_push0, although not considered a regularizer since it does not rely on ground truth data, can be used to "push" clusters away from each other. 
"""

import torch
import numpy as np
import math
# TDA libraries
import persim
from gph import ripser_parallel

def get_dgm(point_cloud, deg=1):
    """
    Computes the persistence diagrams of a point cloud up to a specified degree.
    Args:
      - point_cloud (torch.Tensor or np.ndarray): The input point cloud. Shape (number of points, dimension of each point) (i.e., each point is viewed as a vector).
      - deg (int): Homology degree of homology (0 or 1); persistence diagrams are computed up to degree deg. 1 is the more general option.
    Returns:
      - dgm: A dictionary storing the persistence diagrams of the point cloud and the generators. dgms['dgms'][i]: the persistence diagram of degree i.
    Note: The computation is performed using ripser_parallel, a fast algorithm for computing persistence diagrams that runs on the CPU and expects a NumPy array as input.
    """
    with torch.no_grad():
        # Convert point cloud to numpy if it is a torch tensor
        points = point_cloud.cpu().numpy() if isinstance(point_cloud, torch.Tensor) else point_cloud
        dgm = ripser_parallel(points, maxdim=deg, return_generators=True)
    return dgm

# Euclidean distance for torch tensors
def _dist(point1, point2):
    return torch.sqrt(torch.sum((point2 - point1)**2))

# Supremum distance for two torch tensors (b1, d1) and (b2, d2)
def _dist_sup_tc(b1, d1, b2, d2):
    return torch.max(torch.abs(b1 - b2), torch.abs(d1 - d2))

# Function used for the Reininghaus dissimilarity for two points (a, b) and (c, d)
def _dist_2(a, b, c, d):
    return (a - c)**2 + (b - d)**2

def loss_bottleneck0(point_cloud, point_cloud2, dgm=None, dgm2=None):
    """
    Topological regularizer: Computes the bottleneck distance for homology degree 0.
    """
    # Check if the diagrams have been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2.view(point_cloud2.size(0), -1), 0)
    # If dgm is empty, there is no topological loss:
    if len(dgm['dgms'][0]) <= 1: return torch.tensor(0., device=point_cloud.device), False
    # Compute bottleneck distance:
    with torch.no_grad():
        distance_bottleneck, matching = persim.bottleneck(dgm['dgms'][0][:-1], dgm2['dgms'][0][:-1], matching=True)
        # Find the pair that gives the maximum distance:
        index = np.argmax(matching[:, 2])
        i, j = int(matching[index][0]), int(matching[index][1]) 
        # i is the i-th pt in dgm and j is the j-th pt in dgm2 that give the bottleneck distance. If i==-1, j is matched to the diagonal, and viceversa. 
        # For the loss, need to express the coordinates of P_i (point in dgm) in terms of the point cloud: P_i=(0, d)=(0, _dist(point1_dgm1, point2_dgm1)),
        # where point1_dgm1, point2_dgm1 are the (dgm['gens'][0][i][1])-th and (dgm['gens'][0][i][2])-th points of point_cloud, respectively. 
    if i>=0:
      point1_dgm1 = point_cloud[dgm['gens'][0][i][1]]
      point2_dgm1 = point_cloud[dgm['gens'][0][i][2]]
    
    if i>=0 and j>=0:
      return torch.abs(_dist(point1_dgm1, point2_dgm1) - dgm2['dgms'][0][j][1]), True
    else:
      if i==-1: # So the j-th point from dgm2 is matched to the diagonal -> the bottleneck distance does not depend explicitely on dgm.
        return torch.tensor(0., device=point_cloud.device), False
      else: # Then  j==-1, so the i-th point from dgm is matched to the diagonal. 
        return _dist(point1_dgm1, point2_dgm1)/2., True

def loss_bottleneck1(point_cloud, point_cloud2, dgm=None, dgm2=None):
    """
    Topological regularizer: Computes the bottleneck distance for homology degree 1.
    """
    # Check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2.view(point_cloud2.size(0), -1), 1)
    # If dgm['dgms'][1] is empty, there is no loss:
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=point_cloud.device), False
    # If dgm2['dgms'][1] is empty, make a small change for simplifying the next calculations:
    if len(dgm2['dgms'][1]) == 0:
      dgm2_dgms1_empty = True
      dgm2['dgms'][1] = [[0., 0.]]
    else: 
      dgm2_dgms1_empty = False
    
    with torch.no_grad():
        distance_bottleneck, matching = persim.bottleneck(dgm['dgms'][1], dgm2['dgms'][1], matching=True)
        # Find the pair that gives the maximum distance:
        index = np.argmax(matching[:, 2])
        i, j = int(matching[index][0]), int(matching[index][1])
        # i is the i-th pt in dgm and j is the j-th pt in dgm2 that give the bottleneck distance. 
    
    # For the loss, we need to express the P_i (point in dgm) in terms of the learnable point cloud:
    if i>=0:
      point0_dgm1 = point_cloud[dgm['gens'][1][0][i][0]]
      point1_dgm1 = point_cloud[dgm['gens'][1][0][i][1]]
      point2_dgm1 = point_cloud[dgm['gens'][1][0][i][2]]
      point3_dgm1 = point_cloud[dgm['gens'][1][0][i][3]]
      birth_dgm1 = _dist(point0_dgm1, point1_dgm1)
      death_dgm1 = _dist(point2_dgm1, point3_dgm1)

    # Get the coordinates of the j-th pt in dgm2:
    if j>=0:
      birth_dgm2 = dgm2['dgms'][1][j][0]
      death_dgm2 = dgm2['dgms'][1][j][1]

    # If dgm2 had been modified, go back to its initial form (in case it is used in other topological functions):
    if dgm2_dgms1_empty: dgm2['dgms'][1] = []
    
    if i>=0 and j>=0:
      return _dist_sup_tc(birth_dgm1, death_dgm1, birth_dgm2, death_dgm2), True
    else:
      if i==-1: # So the j-th point from dgm2 is matched to the diagonal
        return torch.tensor(0., device=point_cloud.device), False
      else: # Then j==-1, so the i-th point from dgm is matched to the diagonal
        return (death_dgm1 - birth_dgm1)/2., True

def loss_persentropy0(point_cloud, point_cloud2, dgm=None, dgm2=None, delta=0.001):
    """
    Topological regularizer: Computes the squared difference between the persistence entropies of the two 0-degree persistence diagrams. 
    Only considers points with persistence > delta. (The persistence of a point (b,d) in a diagram is |d-b|. Since the homology degree is 0, 
    we work with points (0,d), so persistence is |d|.)
    Optional args:
      - delta (float): > 0.
    """
    device = point_cloud.device
    # Check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2.view(point_cloud2.size(0), -1), 0)
    
    if len(dgm['dgms'][0]) <= 1: return torch.tensor(0., device=device), False
    # Get persistent entropy of dgm:
    L = torch.tensor(0., device=device)
    pers = torch.tensor(0., device=device)
    for i in range(len(dgm['dgms'][0])-1):
      pers1 = _dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]])
      if pers1 > delta: L = L + pers1
  
    if L.item() == 0.: return torch.tensor(0., device=device), False
    for i in range(len(dgm['dgms'][0])-1):
      pers1 = _dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]])
      if pers1 > delta: pers = pers + pers1 * torch.log(pers1/L)
  
    # Get persistent entropy of dgm2:
    if len(dgm2['dgms'][0])<=1: return (pers/L)**2, True # The entropy of dgm2 is 0
    L2 = 0.
    pers2 = 0.
    for i in range(len(dgm2['dgms'][0])-1):
      if dgm2['dgms'][0][i][1] > delta: L2 += dgm2['dgms'][0][i][1]
    
    if L2 == 0.: return (pers/L) ** 2, True
  
    for i in range(len(dgm2['dgms'][0])-1):
      if dgm2['dgms'][0][i][1] > delta: pers2 += dgm2['dgms'][0][i][1] * math.log(dgm2['dgms'][0][i][1] / L2)
  
    return (pers/L - pers2/L2)**2, True

def loss_persentropy1(point_cloud, point_cloud2, dgm=None, dgm2=None, delta=0.001):
    """
    Topological regularizer: Computes the squared difference between the persistence entropies of the two 1-degree persistence diagrams. 
    Only considers points with persistence > delta.
    Optional args:
      - delta (float): > 0.
    """
    device = point_cloud.device
    # Check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2.view(point_cloud2.size(0), -1), 1)
    
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=device), False
    # Get persistent entropy of dgm:
    L = torch.tensor(0., device=device)
    pers = torch.tensor(0., device=device)
    for i in range(len(dgm['dgms'][1])):
      pers1 = _dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - _dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
      if pers1 > delta: L = L + pers1
  
    if L.item()==0.: return torch.tensor(0., device=device), False
  
    for i in range(len(dgm['dgms'][1])):
      pers1 = _dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - _dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
      if pers1 > delta: pers = pers + pers1 * torch.log(pers1/L)
  
    # Get persistent entropy of dgm2:
    if len(dgm2['dgms'][1])==0: return (pers/L)**2, True # The entropy of dgm2 is 0
    L2 = 0.
    pers2 = 0.
    for i in range(len(dgm2['dgms'][1])):
      if dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0] > delta: L2 += dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]
  
    if L2 == 0.: return (pers/L)**2, True # The entropy of dgm2 is 0
  
    for i in range(len(dgm2['dgms'][1])):
      if dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0] > delta: pers2 += (dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]) * math.log((dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0])/L2)
  
    return (pers/L - pers2/L2)**2, True

def _ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma, device):
    """
    Computes the Reininghaus kernel (or persistence scale space kernel) for two 0-degree
    persistence diagrams (using the formula for k_\sigma in https://arxiv.org/pdf/1412.6821.pdf). 
    This function is a helper for loss_dsigma0.
    """
    ksigma = torch.tensor(0., device=device)
    for i in range(len(dgm['dgms'][0])-1):
        # Point in dgm: (0,d1), d=_dist(p1,p2)
        p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]
        d1 = _dist(p1, p2)
        for j in range(len(dgm2['dgms'][0])-1):
           # Point in dgm2: (0,d2), d=_dist(q1,q2)
           q1, q2 = point_cloud2[dgm2['gens'][0][j][1]], point_cloud2[dgm2['gens'][0][j][2]]
           d2 = _dist(q1, q2)
           ksigma = ksigma + torch.exp(-_dist_2(0, d1, 0, d2)/(8*sigma)) - torch.exp(-_dist_2(0, d1, d2, 0)/(8*sigma))
    return ksigma * 1/(8 * math.pi * sigma)

def loss_dsigma0(point_cloud, point_cloud2, dgm=None, dgm2=None, sigma=0.05):
    """
    Topological regularizer: Given two 0-degree persistence diagrams, computes the squared pseudo-distance that comes from the Reininghaus kernel, 
    d_sigma ** 2 = k11 + k22 - 2*k12 (kij = _ksigma0(pc_i, pc_j, dgm_i, dgm_j).) See https://arxiv.org/pdf/1412.6821.pdf for details.
    However, since k22 = _ksigma0(point_cloud2, point_cloud2, dgm2, dgm2) only depends on ground truth data, it is a constant and not useful for backpropagation,
    hence not included in the loss for faster computation. Hence, the function only returns k11 - 2 * k12. 
    Optional args:
      - sigma (float): scale parameter, > 0. 
    """
    # Check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2.view(point_cloud2.size(0), -1), 0)
    device = point_cloud.device

    if len(dgm['dgms'][0]) <= 1: return torch.tensor(0., device=device), False # No dgm -> no loss
    return _ksigma0(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * _ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma, device), True

def _ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma, device):
    """
    Computes the Reininghaus kernel (or persistence scale space kernel) for two 1-degree
    persistence diagrams (using the formula for k_\sigma in https://arxiv.org/pdf/1412.6821.pdf). 
    This function is a helper for loss_dsigma1.
    """
    ksigma = torch.tensor(0., device=device)
    for i in range(len(dgm['dgms'][1])):
        # Point in dgm: (b1,d1), with b1, d1 = _dist(p2, p1), _dist(p3, p4)
        p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]], point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]
        b1 = _dist(p1,p2)
        d1 = _dist(p3,p4)
        for j in range(len(dgm2['dgms'][1])):
          # Point in dgm2: (b2,d2), with b2, d2 = _dist(q1,q2), _dist(q3,q4)
          q1, q2, q3, q4 = point_cloud2[dgm2['gens'][1][0][j][0]], point_cloud2[dgm2['gens'][1][0][j][1]], point_cloud2[dgm2['gens'][1][0][j][2]], point_cloud2[dgm2['gens'][1][0][j][3]]
          b2 = _dist(q1,q2)
          d2 = _dist(q3,q4)
          ksigma = ksigma + torch.exp(-_dist_2(b1, d1, b2, d2)/(8*sigma)) - torch.exp(-_dist_2(b1, d1, d2, b2)/(8*sigma))
    return ksigma * 1/(8 * math.pi * sigma)

def loss_dsigma1(point_cloud, point_cloud2, dgm=None, dgm2=None, sigma=0.05):
    """
    Topological regularizer: Given two 1-degree persistence diagrams, computes the squared pseudo-distance that comes from the Reininghaus kernel, 
    d_sigma ** 2 = k11 + k22 - 2*k12 (see https://arxiv.org/pdf/1412.6821.pdf). (kij = _ksigma0(pc_i, pc_j, dgm_i, dgm_j).)
    However, since k22 = _ksigma0(point_cloud2, point_cloud2, dgm2, dgm2) only depends on ground truth data, it is a constant and not useful for backpropagation,
    hence not included in the loss for faster computation. So the function only returns k11 - 2 * k12. 
    Optional args:
      - sigma (float): scale parameter, > 0.
    """
    # Check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2.view(point_cloud2.size(0), -1), 1)
    device = point_cloud.device
    
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=device), False
    if len(dgm2['dgms'][1])>0:
      return _ksigma1(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * _ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma, device), True
    else:
      return _ksigma1(point_cloud, point_cloud, dgm, dgm, sigma, device), True

def density(point_cloud, dgm, sigma, scale, x, device):
    """
    Computes the 4SGDE density at coordinate x for a 0-degree persistence diagram. 
    This function is a helper for loss_density.
    """
    density_x = torch.tensor(0., device=device) # Density at coordinate x
    for i in range(len(dgm['dgms'][0])-1):
      p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]
      d = _dist(p1, p2)  # Point (0,d) in the persistence diagram with d=_dist(p1,p2)
      density_x = density_x + d**4 * torch.exp(-((d-x)/sigma)**2)
    return density_x * scale

def loss_density(point_cloud, point_cloud2, dgm=None, dgm2=None, sigma=0.2, scale=0.002, maxrange=35., npoints=30):
    """
    Topological regularizer: Given two 0-degree persistence diagrams, computes a measure of the difference between the 4SGDE density functions of the two diagrams.
    In particular, computes the squared difference between the two density functions at multiple locations, and returns the mean.
    These locations are 'npoints' points equally spaced in [0, maxrange]. 
    Optional args:
      - sigma (float, >0): Controls the curvature of the density function and the relevance given to individual points 
      (sigma->0 yields high individual peaks for each point, while sigma->infty yields a smooth and low-curvature function that highlights 'clusters' of points instead of individual points). 
      - scale (float, >0): Scale factor.
      - maxrange (float, >0) and npoints (int, >1): Control the evaluation points: torch.linspace(0., maxrange, npoints).
    See Figures A.2, A.3 and A.4 in https://diposit.ub.edu/dspace/handle/2445/217016 for visualizing the impact of these values on the density functions.
    """
    # Check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2.view(point_cloud2.size(0), -1), 0)
    device = point_cloud.device
    
    if len(dgm['dgms'][0]) <= 1: return torch.tensor(0., device=device), False
    xs = torch.linspace(0., maxrange, npoints)
    loss = torch.tensor(0., device=device)
    # Compute difference between both functions in npoints points:
    for x in xs: loss = loss + (density(point_cloud, dgm, sigma, scale, x, device) - density(point_cloud2, dgm2, sigma, scale, x, device)) ** 2
    return loss / npoints, True

def loss_push0(point_cloud, dgm):
    """
    Computes the push function for a 0-degree persistence diagram. If used as a loss function and minimized through gradient descent, results
    in a deformation of the point cloud that "pushes" clusters away from each other. Can be used as a helper function whenever topological regularizers return False.  
    """
    # Check if the dgm has been provided:
    if dgm is None: dgm = get_dgm(point_cloud.view(point_cloud.size(0), -1), 0)
    if len(dgm['dgms'][0]) <= 1: return torch.tensor(0., device=point_cloud.device)

    loss = - torch.abs(_dist(point_cloud[dgm['gens'][0][0][1]], point_cloud[dgm['gens'][0][0][2]]))/2.
    for i in range(1, len(dgm['gens'][0])):
      # Point in the diagram: (0, _dist(p1,p2))
      loss = loss - torch.abs(_dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]))/2. # Distance to the diagonal for (0,d) is d/2
    return loss

class TopologicalLoss:
    """
    A class unifying all the topological regularizers for efficiently combining them in machine learning tasks.
    Attributes:
        - topo_weights (7-element list): List of weights for each topological loss. If 0, the corresponding loss is not used. 
        Corresponding functions: [loss_bottleneck0, loss_bottleneck1, loss_persentropy0, loss_persentropy1, loss_dsigma0, loss_dsigma1, loss_density].
        - deg (int): Homology degree for the persistence diagrams (0 or 1, with 1 the more general option).
        - Additional hyperparameters (pers0_delta, pers1_delta, ..., density_npoints): Control the topological functions. By default, they are set to reference values.
    Methods:
        - compute_loss: Computes the total topological loss.
    """
    def __init__(self, topo_weights=[15.,15.,0.,0.,0.,0.,0], deg=1, pers0_delta=0.001, pers1_delta=0.001,
                 dsigma0_sigma=0.05, dsigma1_sigma=0.05, density_sigma=0.2,
                 density_scale=0.002, density_maxrange=35., density_npoints=30):
        self._topo_weights = topo_weights
        self.deg = deg
        self._pers0_delta = pers0_delta
        self._pers1_delta = pers1_delta
        self._dsigma0_sigma = dsigma0_sigma
        self._dsigma1_sigma = dsigma1_sigma
        self._density_sigma = density_sigma
        self._density_scale = density_scale
        self._density_maxrange = density_maxrange
        self._density_npoints = density_npoints
        self._update_loss_functions()

    def _update_loss_functions(self):
        # Updates the loss functions and active losses based on the current parameters (called each time one is changed)
        self.loss_functions = {
            0: (loss_bottleneck0, []),
            1: (loss_bottleneck1, []),
            2: (loss_persentropy0, [self._pers0_delta]),
            3: (loss_persentropy1, [self._pers1_delta]),
            4: (loss_dsigma0, [self._dsigma0_sigma]),
            5: (loss_dsigma1, [self._dsigma1_sigma]),
            6: (loss_density, [self._density_sigma, self._density_scale, self._density_maxrange, self._density_npoints])
        }
        self.active_losses = [(i, func, args) for i, (func, args) in self.loss_functions.items() if self._topo_weights[i] != 0.]

    # Properties and setters to update the active losses and arguments whenever a parameter is changed:
    @property
    def pers0_delta(self):
        return self._pers0_delta
    @pers0_delta.setter
    def pers0_delta(self, value):
        self._pers0_delta = value
        self._update_loss_functions()

    @property
    def pers1_delta(self):
        return self._pers1_delta
    @pers1_delta.setter
    def pers1_delta(self, value):
        self._pers1_delta = value
        self._update_loss_functions()

    @property
    def dsigma0_sigma(self):
        return self._dsigma0_sigma
    @dsigma0_sigma.setter
    def dsigma0_sigma(self, value):
        self._dsigma0_sigma = value
        self._update_loss_functions()

    @property
    def dsigma1_sigma(self):
        return self._dsigma1_sigma
    @dsigma1_sigma.setter
    def dsigma1_sigma(self, value):
        self._dsigma1_sigma = value
        self._update_loss_functions()

    @property
    def density_sigma(self):
        return self._density_sigma
    @density_sigma.setter
    def density_sigma(self, value):
        self._density_sigma = value
        self._update_loss_functions()

    @property
    def density_scale(self):
        return self._density_scale
    @density_scale.setter
    def density_scale(self, value):
        self._density_scale = value
        self._update_loss_functions()

    @property
    def density_maxrange(self):
        return self._density_maxrange
    @density_maxrange.setter
    def density_maxrange(self, value):
        self._density_maxrange = value
        self._update_loss_functions()

    @property
    def density_npoints(self):
        return self._density_npoints
    @density_npoints.setter
    def density_npoints(self, value):
        self._density_npoints = value
        self._update_loss_functions()

    @property
    def topo_weights(self):
        return self._topo_weights
    @topo_weights.setter
    def topo_weights(self, weights):
        self._topo_weights = weights
        self.active_losses = [(i, func, args) for i, (func, args) in self.loss_functions.items() if self._topo_weights[i] != 0.]

    def compute_loss(self, points, true_points, dgm=None, dgm_true=None):
        """
        Computes the total topological loss based on active components.
        Args:
            - points (torch.Tensor): Learnable point cloud. Shape (batch size, additional dimensions).
            - true_points (torch.Tensor): Ground truth point cloud. Shape (batch size, additional dimensions).
            - dgm (dict, optional): Persistence diagram of points.
            - dgm_true (dict, optional): Persistence diagram of true_points.
        Returns:
            - torch.Tensor: Total loss (scalar).
            - bool: True if the loss depends on points, False otherwise.
        """
        if dgm is None: dgm = get_dgm(points.view(points.size(0), -1), self.deg)
        if dgm_true is None: dgm_true = get_dgm(true_points.view(true_points.size(0), -1), self.deg)
        loss = torch.tensor(0., device=points.device)
        # Check if the diagrams are empty:
        if len(dgm['dgms'][0]) <= 1: 
          if self.deg == 0:
            return loss, False
          else:
            if len(dgm['dgms'][1]) == 0:
              return loss, False
        # Compute the losses:
        gotloss = False
        for i, loss_func, args in self.active_losses:
            topoloss, gotloss = loss_func(points, true_points, dgm, dgm_true, *args)
            if gotloss:
                loss = loss + topoloss * self.topo_weights[i]
                n_gotloss = True

        return loss, gotloss
