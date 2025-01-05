import torch
import math
import numpy as np
# TDA libraries
import ripser
import persim
from gph import ripser_parallel

"""
The topological regularizers are: loss_bottleneck0, loss_bottleneck1, loss_persentropy0, loss_persentropy1,
loss_dsigma0, loss_dsigma1, loss_density. Each function computes a different measure of dissimilarity between the diagram
of the learnable point cloud and the ground truth persistence diagram. Each function has the following arguments and outputs:

Input arguments:
  - Required:
    - point_cloud (torch.Tensor): The learnable point cloud. Expected shape (number of points, dimension of each point).
    - point_cloud2 (torch.Tensor): The true point cloud. Expected shape (number of points, dimension of each point).
  - Optional:
    - dgm (dict, optional): Persistence diagram for the first point cloud. If None, it will be computed.
    - dgm2 (dict, optional): Persistence diagram for the true point cloud. If None, it will be computed.
    - Additional optional arguments that control the topological functions.

Output:
  - The computed loss value as a scalar tensor (torch.Tensor). 
  - A status flag (True if the loss depends on the learnable point cloud, False otherwise).

All topological regularizers are unified in the function topo_losses. See details under its definition.

Additionally, the function loss_push0, although not considered a regularizer since it does not rely on ground truth data, can be
used to "push" points or clusters away from each other. 
"""

def get_dgm(point_cloud, deg=1):
    """
    Computes the persistence diagrams of a point cloud up to a specified degree.
    Args:
      - point_cloud (torch.Tensor or np.ndarray): The input point cloud. Shape (number of points, dimension of each point).
      - deg (int): Homology degree of homology (0 or 1); persistence diagrams are computed up to degree deg.
    Returns:
      - A dictionary storing the persistence diagrams of the point cloud and the generators. dgms[i]: The persistence diagram for degree i.
    Note:
      - The computation is performed using ripser_parallel, a fast algorithm for computing persistence diagrams that runs on the CPU and expects a NumPy array as input.
    """
    with torch.no_grad():
        # Convert point cloud to numpy if it's a torch tensor
        points = point_cloud.cpu().numpy() if isinstance(point_cloud, torch.Tensor) else point_cloud
        dgm = ripser_parallel(points, maxdim=deg, return_generators=True)
    return dgm

# Euclidean distance for torch tensors
def _dist(point1, point2):
    return torch.sqrt(torch.sum((point2 - point1)**2))

# Function used for the Reininghaus dissimilarity for two points (a, b) and (c, d)
def _dist_2(a, b, c, d):
    return (a - c)**2 + (b - d)**2

# Supremum distance for two torch tensors (b1, d1) and (b2, d2)
def _dist_sup_tc(b1, d1, b2, d2):
    return torch.max(torch.abs(b1 - b2), torch.abs(d1 - d2))

def loss_bottleneck0(point_cloud, point_cloud2, dgm=None, dgm2=None):
    """
    Topological regularizer: Computes the bottleneck distance for homology degree 0.
    """
    # First, check if the diagrams have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
    # If dgm is empty, there is no topological loss:
    if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=point_cloud.device), False
    # Compute bottleneck distance:
    with torch.no_grad():
        distance_bottleneck, matching = persim.bottleneck(dgm['dgms'][0][:-1], dgm2['dgms'][0][:-1], matching=True)
        #find the pair that gives the max distance:
        index = np.argmax(matching[:, 2])
        i, j = int(matching[index][0]), int(matching[index][1]) #i, j: the i-th and j-th point of the dgm1, dgm2 respectively, that give the bottleneck dist.
        # (if the largest dist is point<->diagonal: i or j is -1)
        #i is the i-th pt in dgm and j is the j-th pt in dgm2 which give the bottleneck dist (i.e. it is the largest dim)
        #for the loss, need to know what is the point i (learnable), i=(distmatrix[xi,yi],distmatrix[ai,bi]) in the distance matrix for some 4 indices
        #but gen[0]
        # i is the index of a point of the PD. but (gens[i][1], gens[i][2]) is the pair of vertices of the point cloud that correspond to the point i=(0,d), with d=dist(gens[i][1]-gens[i][2])
        #get the 2 points that give the distance of the i-th pt in dgm in the 1st diagram and compute the loss:
    if i>=0:
      point1_dgm1 = point_cloud[dgm['gens'][0][i][1]]
      point2_dgm1 = point_cloud[dgm['gens'][0][i][2]]
    
    if i>=0 and j>=0:
      return torch.abs(_dist(point1_dgm1, point2_dgm1) - dgm2['dgms'][0][j][1]), True
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal -> backprop through loss would give 0 -> goal: make points further from diag
        #new_bdist = torch.abs(dist(point1_dgm2, point2_dgm2) - 0.)/2
        return torch.tensor(0., device=point_cloud.device), False
      else: #then  j==-1, so the i-th point from dgm1 is matched to the diagonal
        return _dist(point1_dgm1, point2_dgm1)/2., True

def loss_bottleneck1(point_cloud, point_cloud2, dgm=None, dgm2=None):
    """
    Topological regularizer: Computes the bottleneck distance for homology degree 1.
    """
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 1)
    
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=point_cloud.device), False
    # if dgm2['dgms'][1] is empty, make a small change for simplifying the following calculations:
    if len(dgm2['dgms'][1]) == 0:
      dgm2_dgms1_empty = True
      dgm2['dgms'][1] = [[0., 0.]]
    else: 
      dgm2_dgms1_empty = False
    
    with torch.no_grad():
        distance_bottleneck, matching = persim.bottleneck(dgm['dgms'][1], dgm2['dgms'][1], matching=True)
        #find the pair that gives the max distance:
        index = np.argmax(matching[:, 2])
        i, j = int(matching[index][0]), int(matching[index][1])
        #i is the i-th pt in dgm and j is the j-th pt in dgm2 which give the bottleneck dist (i.e. it is the largest dim)
        #for the loss, need to know what is the point i (learnable), i=(distmatrix[xi,yi],distmatrix[ai,bi]) in the distance matrix for some 4 indices
        # i is the index of a point of the PD. but (gens[i][1], gens[i][2]) is the pair of vertices of the point cloud that correspond to the point i=(0,d), with d=dist(gens[i][1]-gens[i][2])

    #get the 2 points that give the distance of the i-th pt in dgm in the 1st diagram:
    #if i>0, then the pt of dgm1 is off-diag:
    if i>=0:
      point0_dgm1 = point_cloud[dgm['gens'][1][0][i][0]]
      point1_dgm1 = point_cloud[dgm['gens'][1][0][i][1]]
      point2_dgm1 = point_cloud[dgm['gens'][1][0][i][2]]
      point3_dgm1 = point_cloud[dgm['gens'][1][0][i][3]]
      birth_dgm1 = _dist(point0_dgm1, point1_dgm1)
      death_dgm1 = _dist(point2_dgm1, point3_dgm1)

    #get the 2 points that give the distance of the j-th pt in dgm in the 2nd diagram:
    if j>=0:
      birth_dgm2 = dgm2['dgms'][1][j][0]
      death_dgm2 = dgm2['dgms'][1][j][1]

    # if dgm2 had been modified, go back to its initial form (in case it is used in other topofunctions):
    if dgm2_dgms1_empty: dgm2['dgms'][1] = []
    
    if i>=0 and j>=0:
      return _dist_sup_tc(birth_dgm1, death_dgm1, birth_dgm2, death_dgm2), True
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal
        return torch.tensor(0., device=point_cloud.device), False
      else: #then j==-1, so the i-th point from dgm is matched to the diagonal
        return (death_dgm1 - birth_dgm1)/2., True

def loss_persentropy0(point_cloud, point_cloud2, dgm=None, dgm2=None, delta=0.001):
    """
    Topological regularizer: Computes the squared difference between the persistence entropies of the two 0-degree persistence diagrams. 
    Only considers points with persistence > delta. (The persistence of a point (b,d) in the diagrma is |d-b|. Since the homology degree is 0, 
    we work with points (0,d), so persistence is |d|.)
    Optional args:
      delta (float): > 0.
    """
    device = point_cloud.device
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
    
    if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=device), False
    # Get persistent entropy of dgm:
    L = torch.tensor(0., device=device)
    pers = torch.tensor(0., device=device)
    for i in range(len(dgm['dgms'][0])-1):
      pers1 = _dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]])
      if pers1 > delta: L = L + pers1
  
    if L.item() == 0.: return torch.tensor(0., device=device), False
    for i in range(len(dgm['dgms'][0])-1):
      pers1 = _dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]) #p1, p2: pt (0,d) with d=dist(p1,p2) (euclidean dist)
      if pers1 > delta: pers = pers + pers1 * torch.log(pers1/L) #pt of pt cloud is (0,dist(p1, p2))
  
    # Get persistent entropy of dgm2:
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
      delta (float): > 0.
    """
    device = point_cloud.device
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 1)
    
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=device), False
    # Get persistent entropy of dgm:
    L = torch.tensor(0., device=device)
    pers = torch.tensor(0., device=device)
    for i in range(len(dgm['dgms'][1])):
      # pt in dgm: (b1,d1), with b1 = dist(p1, p2), d1 = dist(dist(p3, p4), and pers1=d1-b1.
      pers1 = _dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - _dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
      if pers1 > delta: L = L + pers1
  
    if L.item()==0.: return torch.tensor(0., device=device), False
  
    for i in range(len(dgm['dgms'][1])):
      pers1 = _dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - _dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
      if pers1 > delta: pers = pers + pers1 * torch.log(pers1/L)
  
    if len(dgm2['dgms'][1])==0: return (pers/L)**2, True # the entropy of dgm2 is 0
  
    # Get persistent entropy of dgm2:
    L2 = 0.
    pers2 = 0.
    for i in range(len(dgm2['dgms'][1])):
      if dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0] > delta: L2 += dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]
  
    if L2 == 0.: return (pers/L)**2, True # the entropy of dgm2 is 0
  
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
    for i in range(len(dgm['gens'][0])):
        # pt in dgm: (0,d), d=dist(p1,p2)
        p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]
        d1 = _dist(p1, p2)
        for j in range(len(dgm2['gens'][0])):
           # pt in dgm2: (0,d), d=dist(q1,q2)
           q1, q2 = point_cloud2[dgm2['gens'][0][j][1]], point_cloud2[dgm2['gens'][0][j][2]]
           d2 = _dist(q1, q2)
           ksigma = ksigma + torch.exp(-_dist_2(0, d1, 0, d2)/(8*sigma)) - torch.exp(-_dist_2(0, d1, d2, 0)/(8*sigma))
    return ksigma * 1/(8 * math.pi * sigma)

def loss_dsigma0(point_cloud, point_cloud2, dgm=None, dgm2=None, sigma=0.05):
    """
    Topological regularizer: Given two 0-degree persistence diagrams, computes the squared pseudo-distance that comes from the Reininghaus kernel, 
    d_sigma ** 2 = k11 + k22 - 2*k12 (see https://arxiv.org/pdf/1412.6821.pdf). (kij = _ksigma0(pc_i, pc_j, dgm_i, dgm_j).)
    However, since k22 = _ksigma0(point_cloud2, point_cloud2, dgm2, dgm2) only depends on ground truth data, it is a constant and not useful for backpropagation,
    hence not included in the loss for faster computation. So the function only returns k11 - 2 * k12. 
    Optional args:
      sigma (float): scale parameter, > 0. 
    """
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
    device = point_cloud.device

    if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=device), False
    # Return squared pseudo-distance that comes from ksigma, dsigma**2: k11 + k22 - 2*k12
    # But no need of k22 = ksigma(point_cloud2, point_cloud2) since it is fixed (no backpropagation) -> return k11 - 2 * k12
    return _ksigma0(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * _ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma, device), True

# Same as ksigma0, but here we take the points in diagrams of degree 1 instead of degree 0
def _ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma, device):
    """
    Computes the Reininghaus kernel (or persistence scale space kernel) for two 1-degree
    persistence diagrams (using the formula for k_\sigma in https://arxiv.org/pdf/1412.6821.pdf). 
    This function is a helper for loss_dsigma1.
    """
    ksigma = torch.tensor(0., device=device)
    ## use formula for k_sigma from paper (https://arxiv.org/pdf/1412.6821.pdf):
    for i in range(len(dgm['gens'][1])):
        # pt in dgm: (b1,d1), with b1, d1 = dist(p2, p1), dist(dist(p3, p4)
        p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]], point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]
        b1 = _dist(p1,p2)
        d1 = _dist(p3,p4)
        for j in range(len(dgm2['gens'][1])):
          #pt in dgm2: (b2,d2)
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
      sigma (float): scale parameter, > 0.
    """
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 1)
    device = point_cloud.device
    
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=device), False
    if len(dgm2['gens'][1])>0:
      return _ksigma1(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * _ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma, device), True
    else:
      return _ksigma1(point_cloud, point_cloud, dgm, dgm, sigma, device), True

def density(point_cloud, dgm, sigma, scale, x, device):
    """
    Computes the 4-SGDE density at coordinate x for a 0-degree persistence diagram. 
    This function is a helper for loss_density.
    """
    density_x = torch.tensor(0., device=device) # Density at coordinate x
    for i in range(len(dgm['dgms'][0])-1):
      p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]] #pt (0,d) with d=dist(p1,p2) (euclidean dist)
      d = _dist(p1, p2) #pt of pt cloud is (0,d)
      density_x = density_x + d**4 * torch.exp(-((d-x)/sigma)**2)
    return density_x * scale

def loss_density(point_cloud, point_cloud2, dgm=None, dgm2=None, sigma=0.2, scale=0.002, maxrange=35., npoints=30):
    """
    Topological regularizer: Given two 0-degree persistence diagrams, computes a measure of the difference between the 4-SGDE density functions of the two diagrams.
    In particular, computes the squared difference between the two density functions at multiple locations, and returns the mean.
    These locations correspond to 'npoints' points equally spaced between 0 and 'maxrange'. 
    Optional args:
      sigma (float, >0): Controls the curvature of the density function and the relevance given to individual points 
        (sigma->0 yields high individual peaks for each point, while sigma->infty yields a smooth and low-curvature function that highlights the 'clusters' of points instead of individual points). 
      scale (float, >0): Scale factor for the density function.
      maxrange (float, >0) and npoints (int, >1): Control the evaluation points xs = torch.linspace(0., maxrange, npoints).
    See Figures A.2, A.3 and A.4 in for visualizing the impact of these values on the density functions.
    """
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
    device = point_cloud.device
    
    if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=device), False
    xs = torch.linspace(0., maxrange, npoints)
    loss = torch.tensor(0., device=device)
    # Compute difference between both functions in npoints points:
    for x in xs: loss = loss + (density(point_cloud, dgm, sigma, scale, x, device) - density(point_cloud2, dgm2, sigma, scale, x, device)) ** 2
    return loss / npoints, True

#auxiliary loss when d(D,D0) (in deg0) only depends on D0 (so gradients are 0):
def loss_push0(point_cloud, dgm):
    # First, check if the dgm has been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 0)
    
    loss = -torch.abs(_dist(point_cloud[dgm['gens'][0][0][1]], point_cloud[dgm['gens'][0][0][2]]))/2.
    for i in range(1, len(dgm['gens'][0])):
      # Point in the diagram: (0,dist(p1,p2))
      loss = loss - torch.abs(_dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]))/2. #dist to diagonal of (0,d) is d/2
    return loss

class TopologicalLoss:
    """
    A class unifying all the topological regularizers, for computing the total topological loss in machine learning models.

    Attributes:
        topo_weights (7-element list): List of weights for each topological loss. If 0, the corresponding loss is not used. 
        Corresponding functions: [loss_bottleneck0, loss_bottleneck1, loss_persentropy0, loss_persentropy1, loss_dsigma0, loss_dsigma1, loss_density] 
        deg (int): Degree of homology for the persistence diagrams (0 or 1, with 1 the more general option).
        Additional parameters (pers0_delta, pers1_delta, ..., density_npoints) that control the topological functions, which are set
        to reference values by default but can be modified depending on the dataset, model, or other considerations.
    
    Methods:
        compute_loss(points, true_points, dgm=None, dgm_true=None): Computes the total topological loss based on active components.
    """
    def __init__(self, topo_weights=[15.,15.,0.,0.,0.,0.,0], deg=1, pers0_delta=0.001, pers1_delta=0.001,
                 dsigma0_sigma=0.05, dsigma1_sigma=0.05, density_sigma=0.2,
                 density_scale=0.002, density_maxrange=35., density_npoints=30):
        self.deg = deg
        self._topo_weights = topo_weights
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

    # Use properties and setters to update the active losses and arguments whenever a parameter (e.g. topo_weights) is changed:
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
            points (torch.Tensor): Learnable point cloud. Shape: (batch size, additional dimensions).
            true_points (torch.Tensor): Ground truth point cloud. Shape: (batch size, additional dimensions).
            dgm (torch.Tensor, optional): Persistence diagram for points.
            dgm_true (torch.Tensor, optional): Persistence diagram for true_points.
        Returns:
            torch.Tensor: Total loss.
            bool: True if the loss depends on the input points, False otherwise.
        """
        if dgm is None: dgm = get_dgm(points.view(points.size(0), -1), self.deg)
        if dgm_true is None: dgm_true = get_dgm(true_points.view(true_points.size(0), -1), self.deg)
        loss = torch.tensor(0., device=points.device)
        n_gotloss = 0

        for i, loss_func, args in self.active_losses:
            topoloss, gotloss = loss_func(points, true_points, dgm, dgm_true, *args)
            if gotloss:
                loss = loss + topoloss * self.topo_weights[i]
                n_gotloss += 1

        return loss, n_gotloss > 0
