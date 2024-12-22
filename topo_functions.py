import sys
import os
import math
import random
import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import make_grid, save_image

# TDA libraries
import ripser
import persim
import tadasets
from gph import ripser_parallel

def get_dgm(point_cloud, deg):
  # Compute the persistence diagram without backprop
  with torch.no_grad():
        # Convert points for computing PD:
        points_np = point_cloud.numpy()
        # Get PD with generators:
        dgm = ripser_parallel(point_cloud, maxdim=deg, return_generators=True)
  return dgm

# Euclidean dist for torch tensors:
def dist(point1, point2):
    return torch.sqrt(torch.sum((point2 - point1)**2))

def dist_2(a, b, c, d):
    return (a - c)**2 + (b - d)**2

# Supremum dist for torch tensors:
def dist_sup_tc(b1, d1, b2, d2):
    # Calculate the sup norm between points (b1, d1) and (b2, d2)
    return torch.max(torch.abs(b1 - b2), torch.abs(d1 - d2))

def loss_bottleneck0(point_cloud, dgm, dgm2): # got_loss=1 if got loss, =0 if loss does not depend on dgm
    got_loss = 1
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
      loss = torch.abs(dist(point1_dgm1, point2_dgm1) - torch.tensor(dgm2['dgms'][0][j][1]))
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal -> backprop through loss would give 0 -> goal: make points further from diag
        #new_bdist = torch.abs(dist(point1_dgm2, point2_dgm2) - 0.)/2
        loss = 0
        got_loss = 0
      else: #then  j==-1, so the i-th point from dgm1 is matched to the diagonal
        loss = dist(point1_dgm1, point2_dgm1)/2.

    return loss, got_loss

def loss_bottleneck1(point_cloud, dgm, dgm2): # got_loss=1 if got loss, =0 if loss does not depend on dgm
    got_loss = 1
    if len(dgm2['dgms'][1])==0: dgm2['dgms'][1] = [[0.,0.]]
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
      birth_dgm2 = torch.tensor(dgm2['dgms'][1][j][0])
      death_dgm2 = torch.tensor(dgm2['dgms'][1][j][1])

    if i>=0 and j>=0:
      loss = dist_sup_tc(birth_dgm1, death_dgm1, birth_dgm2, death_dgm2)
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal
        loss = 0
        got_loss = 0
      else: #then j==-1, so the i-th point from dgm1 is matched to the diagonal
        loss = (death_dgm1 - birth_dgm1)/2.

    return loss, got_loss

def loss_persentropy0(point_cloud, dgm, dgm2, delta): # dgm of deg0. only looks at points with pers=|d|>delta in both dgms
  # Get persistent entropy of dgm:
  L = torch.tensor(0.0, requires_grad=True)
  pers = torch.tensor(0.0, requires_grad=True)
  for i in range(len(dgm['dgms'][0])-1):
    pers1 = dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]])
    if pers1 > delta: L = L + pers1

  if L.item() == 0.0: return torch.tensor(0.0), 0
  for i in range(len(dgm['dgms'][0])-1):
    pers1 = dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]) #p1, p2: pt (0,d) with d=dist(p1,p2) (euclidean dist)
    if pers1 > delta: pers = pers + pers1 * torch.log(pers1/L) #pt of pt cloud is (0,dist(p1, p2))

  # Get persistent entropy of dgm2:
  L2 = torch.tensor(0.0)
  pers2 = torch.tensor(0.0)
  for i in range(len(dgm2['dgms'][0])-1):
    if dgm2['dgms'][0][i][1] > delta: L2 += dgm2['dgms'][0][i][1]
  
  if L2.item() == 0.0: return (pers/L) ** 2, 1

  for i in range(len(dgm2['dgms'][0])-1):
    if dgm2['dgms'][0][i][1] > delta: pers2 += dgm2['dgms'][0][i][1] * torch.log(dgm2['dgms'][0][i][1] / L2)

  return (pers/L - pers2/L2)**2, 1

def loss_persentropy1(point_cloud, dgm, dgm2, delta): #dgm of deg1. returns loss, got_loss (0 if did not get it). only looks at points with pers=|d-b|>delta (in both dgms) (for avoiding large gradients)
  # Get persistent entropy of dgm:
  L = torch.tensor(0.0, requires_grad=True)
  pers = torch.tensor(0.0, requires_grad=True)
  for i in range(len(dgm['dgms'][1])):
    # pt in dgm: (b1,d1), with b1 = dist(p1, p2), d1 = dist(dist(p3, p4), and pers1=d1-b1.
    pers1 = dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
    if pers1 > delta: L = L + pers1

  if L.item()==0.0: return torch.tensor(0.0), 0

  for i in range(len(dgm['dgms'][1])):
    pers1 = dist(point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]) - dist(point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]])
    if pers1 > delta: pers = pers + pers1 * torch.log(pers1/L)

  if len(dgm2['dgms'][1])==0: return (pers/L)**2, 1 # the entropy of dgm2 is 0

  # Get persistent entropy of dgm2:
  L2 = torch.tensor(0.0)
  pers2 = torch.tensor(0.0)
  for i in range(len(dgm2['dgms'][1])):
    if dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0] > delta: L2 += dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]

  if L2.item()==0.0: return (pers/L)**2, 1 # the entropy of dgm2 is 0

  for i in range(len(dgm2['dgms'][1])):
    if dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0] > delta: pers2 += (dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]) * torch.log((dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0])/L2)

  return (pers/L - pers2/L2)**2, 1

#return Reininghaus kernel ksigma: (could make it slightly faster with different functions for each dgm (dgm2 does not need backpropagation)), but let it same for all dgms
def ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma): #maxdim of both dgms: 0
    ksigma = torch.tensor(0.0, requires_grad=True)
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

def loss_dsigma0(point_cloud, point_cloud2, dgm, dgm2, sigma=0.05):
    k11 = ksigma0(point_cloud, point_cloud, dgm, dgm, sigma)
    k12 = ksigma0(point_cloud, point_cloud2, dgm, dgm2, sigma)
    # Return squared pseudo-distance that comes from ksigma, dsigma**2: k11 + k22 - 2*k12
    # But no need of k22 = ksigma(point_cloud2, point_cloud2) since it is fixed (no backpropagation)
    return k11 - 2.0 * k12

# Same as ksigma0, but here we take the points in diagrams of degree 1 instead of degree 0
def ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma):
    ksigma = torch.tensor(0.0, requires_grad=True)
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

def loss_dsigma1(point_cloud, point_cloud2, dgm, dgm2, sigma=0.05):
    if len(dgm2['gens'][1])>0:
      return ksigma1(point_cloud, point_cloud, dgm, dgm, sigma) - 2.0 * ksigma1(point_cloud, point_cloud2, dgm, dgm2, sigma)
    else:
      return ksigma1(point_cloud, point_cloud, dgm, dgm, sigma)

def density(point_cloud, dgm, sigma, scale, x):
  density_x = torch.tensor(0.0, requires_grad=True) # Density at coordinate x
  for i in range(len(dgm['dgms'][0])-1):
    p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]] #pt (0,d) with d=dist(p1,p2) (euclidean dist)
    d = dist(p1, p2) #pt of pt cloud is (0,d)
    density_x = density_x + d**4 * torch.exp(-((d-x)/sigma)**2)
  return density_x * scale

def loss_density(point_cloud, point_cloud2, dgm, dgm2, sigma=0.2, scale=0.002, maxrange=35., npoints=30):
  xs = torch.linspace(0.0, maxrange, npoints)
  loss = torch.tensor(0.0, requires_grad=True)
  # Compute difference between both functions in npoints points:
  for x in xs: loss = loss + (density(point_cloud, dgm, sigma, scale, x) - density(point_cloud2, dgm2, sigma, scale, x)) ** 2
  return loss / npoints

#auxiliary loss when d(D,D0) (in deg0) only depends on D0 (so gradients are 0):
def loss_push0(point_cloud, dgm):
    loss = -torch.abs(dist(point_cloud[dgm['gens'][0][0][1]], point_cloud[dgm['gens'][0][0][2]]))/2.
    for i in range(1, len(dgm['gens'][0])):
      # Point in the diagram: (0,dist(p1,p2))
      loss = loss - torch.abs(dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]))/2. #dist to diagonal of (0,d) is d/2
    return loss

# topo_losses combines all the previous functions into a single function:
# args has to contain:
# - topo_weights: [w_topo0, w_topo1, w_pers0, w_pers1, w_dsigma0, w_dsigma1, w_density0]. weight set as 0: topofunction not used
# - hyperparameters for topological functions: pers0_delta=0.001, pers1_delta=0.001, dsigma0_scale=0.05, dsigma1_scale=0.05, density_sigma=0.2, density_scale=0.002, density_maxrange=35., density_npoints=30
def topo_losses(points, true_points, dgm, dgm_true, args)
    dgm0_notempty = len(dgm['dgms'][0]) > 0
    dgm1_notempty = len(dgm['dgms'][1]) > 0
    loss = torch.tensor(0.0, requires_grad=True)
    if args.topo_weights[0] != 0. and dgm0_notempty:
      topoloss, gotloss = loss_bottleneck0(points, dgm, dgm_true)
      if gotloss==1: loss = loss + topoloss * args.topo_weights[0]
    if args.topo_weights[1] != 0. and dgm1_notempty:
      topoloss, gotloss = loss_bottleneck1(points, dgm, dgm_true)
      if gotloss==1: loss = loss + topoloss * args.topo_weights[1]
    if args.topo_weights[2] != 0. and dgm0_notempty:
      topoloss, gotloss = loss_persentropy0(points, dgm, dgm_true, args.pers0_delta)
      if gotloss==1: loss = loss + topoloss * args.topo_weights[2]
    if args.topo_weights[3] != 0. and dgm1_notempty:
      topoloss, gotloss = loss_persentropy1(points, dgm, dgm_true, args.pers1_delta)
      if gotloss==1: loss = loss + topoloss * args.topo_weights[3]
    if args.topo_weights[4] != 0. and dgm0_notempty:
      loss = loss + loss_dsigma0(points, true_points, dgm, dgm_true, args.dsigma0_scale) * args.topo_weights[4]
    if args.topo_weights[5] != 0. and dgm1_notempty:
      loss = loss + loss_dsigma1(points, true_points, dgm, dgm_true, args.dsigma1_scale) * args.topo_weights[5]
    if args.topo_weights[6] != 0. and dgm0_notempty:
      loss = loss + loss_density(points, true_points, dgm, dgm_true, args.density_sigma, args.density_scale, args.density_maxrange, args.density_npoints) * args.topo_weights[6]
    return loss
