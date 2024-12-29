import torch
import math
import numpy as np
# TDA libraries
import ripser
import persim
from gph import ripser_parallel

"""
  The topological regularizers are the functions loss_bottleneck0, loss_bottleneck1, loss_persentropy0, loss_persentropy1,
  loss_dsigma0, loss_dsigma1, loss_density. Each function computes a different measure of dissimilarity between the diagram
  of the learnable point cloud and the ground truth persistence diagram. Each function has the following arguments:
  
  Input arguments:
    - point_cloud (tensor-like:): The learnable point cloud. Expected shape (number of points, dimension of each point).
    - point_cloud2 (tensor-like): The true point cloud. Expected shape (number of points, dimension of each point).
    - dgm (dict, optional): Persistence diagram for the first point cloud. If None, it will be computed.
    - dgm2 (dict, optional): Persistence diagram for the true point cloud. If None, it will be computed.
    - device (str, optional): The device to use for computations. Defaults to "cpu".
    - Additional optional arguments that control the topological functions.
  
  Returns:
    - torch.Tensor: The computed loss value as a scalar tensor.
    - bool: A status flag (True if the loss depends on the learnable point cloud, False otherwise).
"""

"""
  All topological regularizers are unified in the function topo_losses. 

  Input arguments:
    - Required:
      - points: Learnable point cloud.
      - true_points: Ground truth point cloud.
      - topo_weights: Associated to each topological loss: [w_topo0, w_topo1, w_pers0, w_pers1, w_dsigma0, w_dsigma1, w_density0]. 
      If weight set as 0, its topofunction is not used.
    - Optional:
      - deg (default: 1): Homology degree (0 or 1, 1 is the more general option).
      - dgm_true (default: None): Persistence diagram of the ground truth data. If None, calculated inside the function.
      - device (default: "cpu"): The device to use for computations.
      - Parameters for topological functions (set to reference values, but can be modified depending on the dataset, model, etc.):
      pers0_delta=0.001, pers1_delta=0.001, dsigma0_scale=0.05, dsigma1_scale=0.05, density_sigma=0.2, density_scale=0.002, density_maxrange=35., density_npoints=30

  Returns:
    - torch.Tensor: The computed total loss value as a scalar tensor.
    - bool: A status flag (True if the loss depends on the diagrams, False otherwise).
"""

def get_dgm(point_cloud, deg=1):
  # Compute the persistence diagram without backprop
  with torch.no_grad():
        # Convert points for computing PD:
        if isinstance(point_cloud, torch.Tensor): points = point_cloud.cpu().numpy()
        else: points = point_cloud
        # Get PD with generators:
  return ripser_parallel(points, maxdim=deg, return_generators=True)

# Euclidean dist for torch tensors:
def dist(point1, point2):
    return torch.sqrt(torch.sum((point2 - point1)**2))

def dist_2(a, b, c, d):
    return (a - c)**2 + (b - d)**2

# Supremum dist for torch tensors:
def dist_sup_tc(b1, d1, b2, d2):
    # Calculate the sup norm between points (b1, d1) and (b2, d2)
    return torch.max(torch.abs(b1 - b2), torch.abs(d1 - d2))

def loss_bottleneck0(point_cloud, point_cloud2, dgm=None, dgm2=None, device="cpu"): # second value returned: 1 if got loss, 0 if the loss does not depend on dgm
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
    # If dgm is empty, there is no topological loss:
    if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=device), False
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
      ll = torch.abs(dist(point1_dgm1, point2_dgm1) - dgm2['dgms'][0][j][1])
      print(f"b0: device: {ll.device}, grad {ll.requires_grad}")
      return torch.abs(dist(point1_dgm1, point2_dgm1) - dgm2['dgms'][0][j][1]), True
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal -> backprop through loss would give 0 -> goal: make points further from diag
        #new_bdist = torch.abs(dist(point1_dgm2, point2_dgm2) - 0.)/2
        return torch.tensor(0., device=device), False
      else: #then  j==-1, so the i-th point from dgm1 is matched to the diagonal
        ll = dist(point1_dgm1, point2_dgm1)/2.
        print(f"b0: device: {ll.device}, grad {ll.requires_grad}")
        return dist(point1_dgm1, point2_dgm1)/2., True

def loss_bottleneck1(point_cloud, point_cloud2, dgm=None, dgm2=None, device="cpu"): # second value returned: 1 if got loss, 0 if the loss does not depend on dgm
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 1)
    
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=device), False
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
      birth_dgm1 = dist(point0_dgm1, point1_dgm1)
      death_dgm1 = dist(point2_dgm1, point3_dgm1)

    #get the 2 points that give the distance of the j-th pt in dgm in the 2nd diagram:
    if j>=0:
      birth_dgm2 = dgm2['dgms'][1][j][0]
      death_dgm2 = dgm2['dgms'][1][j][1]

    # if dgm2 had been modified, go back to its initial form (in case it is used in other topofunctions):
    if dgm2_dgms1_empty: dgm2['dgms'][1] = []
    
    if i>=0 and j>=0:
      ll = dist_sup_tc(birth_dgm1, death_dgm1, birth_dgm2, death_dgm2)
      print(f"b1: device: {ll.device}, grad {ll.requires_grad}")
      return dist_sup_tc(birth_dgm1, death_dgm1, birth_dgm2, death_dgm2), True
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal
        return torch.tensor(0., device=device), False
      else: #then j==-1, so the i-th point from dgm is matched to the diagonal
        ll = (death_dgm1 - birth_dgm1)/2.
        print(f"b1: device: {ll.device}, grad {ll.requires_grad}")
        return (death_dgm1 - birth_dgm1)/2., True

def loss_persentropy0(point_cloud, point_cloud2, dgm=None, dgm2=None, device="cpu", delta=0.001): # dgm of deg0. only looks at points with pers=|d|>delta in both dgms
  # First, check if the dgms have been provided:
  if dgm is None: dgm = get_dgm(point_cloud, 0)
  if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
  
  if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=device), False
  # Get persistent entropy of dgm:
  L = torch.tensor(0., device=device)
  pers = torch.tensor(0., device=device)
  for i in range(len(dgm['dgms'][0])-1):
    pers1 = dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]])
    if pers1 > delta: L = L + pers1

  if L.item() == 0.: return torch.tensor(0., device=device), False
  for i in range(len(dgm['dgms'][0])-1):
    pers1 = dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]) #p1, p2: pt (0,d) with d=dist(p1,p2) (euclidean dist)
    if pers1 > delta: pers = pers + pers1 * torch.log(pers1/L) #pt of pt cloud is (0,dist(p1, p2))

  # Get persistent entropy of dgm2:
  L2 = 0.
  pers2 = 0.
  for i in range(len(dgm2['dgms'][0])-1):
    if dgm2['dgms'][0][i][1] > delta: L2 += dgm2['dgms'][0][i][1]
  
  if L2 == 0.: return (pers/L) ** 2, True

  for i in range(len(dgm2['dgms'][0])-1):
    if dgm2['dgms'][0][i][1] > delta: pers2 += dgm2['dgms'][0][i][1] * math.log(dgm2['dgms'][0][i][1] / L2)

  ll = (pers/L - pers2/L2)**2
  print(f"e0: device: {ll.device}, grad {ll.requires_grad}")
  return (pers/L - pers2/L2)**2, True

def loss_persentropy1(point_cloud, point_cloud2, dgm=None, dgm2=None, device="cpu", delta=0.001): #dgm of deg1. returns loss, got_loss (0 if did not get it). only looks at points with pers=|d-b|>delta (in both dgms) (for avoiding large gradients)
  # First, check if the dgms have been provided:
  if dgm is None: dgm = get_dgm(point_cloud, 1)
  if dgm2 is None: dgm2 = get_dgm(point_cloud2, 1)
  
  if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=device), False
  # Get persistent entropy of dgm:
  L = torch.tensor(0., device=device)
  pers = torch.tensor(0., device=device)
  for i in range(len(dgm['dgms'][1])):
    # pt in dgm: (b1,d1), with b1 = dist(p1, p2), d1 = dist(dist(p3, p4), and pers1=d1-b1.
    pers1 = dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
    if pers1 > delta: L = L + pers1

  if L.item()==0.: return torch.tensor(0., device=device), False

  for i in range(len(dgm['dgms'][1])):
    pers1 = dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
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

  ll = (pers/L - pers2/L2)**2
  print(f"e1: device: {ll.device}, grad {ll.requires_grad}")
  return (pers/L - pers2/L2)**2, True

#return Reininghaus kernel ksigma: (could make it slightly faster with different functions for each dgm (dgm2 does not need backpropagation)), but let it same for all dgms
def ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma, device): #maxdim of both dgms: 0
    ksigma = torch.tensor(0., device=device)
    ## use formula for k_sigma from paper (https://arxiv.org/pdf/1412.6821.pdf):
    for i in range(len(dgm['gens'][0])):
        # pt in dgm: (0,d), d=dist(p1,p2)
        p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]
        d1 = dist(p1, p2)
        for j in range(len(dgm2['gens'][0])):
           # pt in dgm2: (0,d), d=dist(q1,q2)
           q1, q2 = point_cloud2[dgm2['gens'][0][j][1]], point_cloud2[dgm2['gens'][0][j][2]]
           d2 = dist(q1, q2)
           ksigma = ksigma + torch.exp(-dist_2(0, d1, 0, d2)/(8*sigma)) - torch.exp(-dist_2(0, d1, d2, 0)/(8*sigma))
    return ksigma * 1/(8 * math.pi * sigma)

def loss_dsigma0(point_cloud, point_cloud2, dgm=None, dgm2=None, device="cpu", sigma=0.05):
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 0)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
    
    if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=device), False
    # Return squared pseudo-distance that comes from ksigma, dsigma**2: k11 + k22 - 2*k12
    # But no need of k22 = ksigma(point_cloud2, point_cloud2) since it is fixed (no backpropagation) -> return k11 - 2 * k12
    ll = ksigma0(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma, device)
    print(f"ds0: device: {ll.device}, grad {ll.requires_grad}")
    return ksigma0(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma, device), True

# Same as ksigma0, but here we take the points in diagrams of degree 1 instead of degree 0
def ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma, device):
    ksigma = torch.tensor(0., device=device)
    ## use formula for k_sigma from paper (https://arxiv.org/pdf/1412.6821.pdf):
    for i in range(len(dgm['gens'][1])):
        # pt in dgm: (b1,d1), with b1, d1 = dist(p2, p1), dist(dist(p3, p4)
        p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]], point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]
        b1 = dist(p1,p2)
        d1 = dist(p3,p4)
        for j in range(len(dgm2['gens'][1])):
          #pt in dgm2: (b2,d2)
          q1, q2, q3, q4 = point_cloud2[dgm2['gens'][1][0][j][0]], point_cloud2[dgm2['gens'][1][0][j][1]], point_cloud2[dgm2['gens'][1][0][j][2]], point_cloud2[dgm2['gens'][1][0][j][3]]
          b2 = dist(q1,q2)
          d2 = dist(q3,q4)
          ksigma = ksigma + torch.exp(-dist_2(b1, d1, b2, d2)/(8*sigma)) - torch.exp(-dist_2(b1, d1, d2, b2)/(8*sigma))
    return ksigma * 1/(8 * math.pi * sigma)

def loss_dsigma1(point_cloud, point_cloud2, dgm=None, dgm2=None, device="cpu", sigma=0.05):
    # First, check if the dgms have been provided:
    if dgm is None: dgm = get_dgm(point_cloud, 1)
    if dgm2 is None: dgm2 = get_dgm(point_cloud2, 1)
    
    if len(dgm['dgms'][1]) == 0: return torch.tensor(0., device=device), False
    if len(dgm2['gens'][1])>0:
      ll = ksigma1(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma, device)
      print(f"ds1: device: {ll.device}, grad {ll.requires_grad}")
      return ksigma1(point_cloud, point_cloud, dgm, dgm, sigma, device) - 2.0 * ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma, device), True
    else:
      return ksigma1(point_cloud, point_cloud, dgm, dgm, sigma, device), True

def density(point_cloud, dgm, sigma, scale, x, device):
  density_x = torch.tensor(0.0, device=device) # Density at coordinate x
  for i in range(len(dgm['dgms'][0])-1):
    p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]] #pt (0,d) with d=dist(p1,p2) (euclidean dist)
    d = dist(p1, p2) #pt of pt cloud is (0,d)
    density_x = density_x + d**4 * torch.exp(-((d-x)/sigma)**2)
  return density_x * scale

def loss_density(point_cloud, point_cloud2, dgm=None, dgm2=None, device="cpu", sigma=0.2, scale=0.002, maxrange=35., npoints=30):
  # First, check if the dgms have been provided:
  if dgm is None: dgm = get_dgm(point_cloud, 0)
  if dgm2 is None: dgm2 = get_dgm(point_cloud2, 0)
  
  if len(dgm['dgms'][0]) == 0: return torch.tensor(0., device=device), False
  xs = torch.linspace(0., maxrange, npoints)
  loss = torch.tensor(0., device=device)
  # Compute difference between both functions in npoints points:
  for x in xs: loss = loss + (density(point_cloud, dgm, sigma, scale, x, device) - density(point_cloud2, dgm2, sigma, scale, x, device)) ** 2
  
  ll = loss / npoints
  print(f"density: device: {ll.device}, grad {ll.requires_grad}")
  return loss / npoints, True

#auxiliary loss when d(D,D0) (in deg0) only depends on D0 (so gradients are 0):
def loss_push0(point_cloud, dgm):
  # First, check if the dgm has been provided:
  if dgm is None: dgm = get_dgm(point_cloud, 0)
  
  loss = -torch.abs(dist(point_cloud[dgm['gens'][0][0][1]], point_cloud[dgm['gens'][0][0][2]]))/2.
  for i in range(1, len(dgm['gens'][0])):
    # Point in the diagram: (0,dist(p1,p2))
    loss = loss - torch.abs(dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]))/2. #dist to diagonal of (0,d) is d/2
  return loss

def topo_losses(points, true_points, topo_weights, deg=1, dgm=None, dgm_true=None, device="cpu", pers0_delta=0.001, pers1_delta=0.001, dsigma0_scale=0.05, dsigma1_scale=0.05,
                density_sigma=0.2, density_scale=0.002, density_maxrange=35., density_npoints=30):
    n_gotloss = 0
    if dgm is None: dgm = get_dgm(points.view(points.size(0), -1), deg)
    if dgm_true is None: dgm_true = get_dgm(true_points.view(true_points.size(0), -1), deg)
    
    loss = torch.tensor(0., device=device)
    if topo_weights[0] != 0.:
      topoloss, gotloss0 = loss_bottleneck0(points, true_points, dgm, dgm_true, device)
      if gotloss0:
        loss = loss + topoloss * topo_weights[0]
        n_gotloss += 1
    if topo_weights[1] != 0.:
      topoloss, gotloss1 = loss_bottleneck1(points, true_points, dgm, dgm_true, device)
      if gotloss1: 
        loss = loss + topoloss * topo_weights[1]
        n_gotloss += 1 
    if topo_weights[2] != 0.:
      topoloss, gotloss2 = loss_persentropy0(points, true_points, dgm, dgm_true, device, pers0_delta)
      if gotloss2: 
        loss = loss + topoloss *  topo_weights[2]
        n_gotloss += 1
    if topo_weights[3] != 0.:
      topoloss, gotloss3 = loss_persentropy1(points, true_points, dgm, dgm_true, device, pers1_delta)
      if gotloss3: 
        loss = loss + topoloss * topo_weights[3]
        n_gotloss += 1
    if topo_weights[4] != 0.:
      topoloss, gotloss4 = loss_dsigma0(points, true_points, dgm, dgm_true, device, dsigma0_scale)
      if gotloss4: 
        loss = loss + topoloss * topo_weights[4]
        n_gotloss += 1
    if topo_weights[5] != 0.:
      topoloss, gotloss5 = loss_dsigma1(points, true_points, dgm, dgm_true, device, dsigma1_scale)
      if gotloss5: 
        loss = loss + topoloss * topo_weights[5]
        n_gotloss += 1
    if topo_weights[6] != 0.:
      topoloss, gotloss6 = loss_density(points, true_points, dgm, dgm_true, device, density_sigma, density_scale, density_maxrange, density_npoints)
      if gotloss6:
        loss = loss + topoloss * topo_weights[6]
        n_gotloss += 1
    return loss, n_gotloss > 0
