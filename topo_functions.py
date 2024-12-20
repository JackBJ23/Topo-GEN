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

## push0 and push1 are not used here

def get_dgm(point_cloud, deg):
  # Compute the persistence diagram without backprop
  with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        # get PD with generators:
        dgm = ripser_parallel(points_np, maxdim=deg, return_generators=True)
  return dgm

#euclidean dist for torch tensors:
def dist(point1, point2):
    return torch.sqrt(torch.sum((point2 - point1)**2))

def dist_2(a, b, c, d):
    return (a - c)**2 + (b - d)**2

#euclidean dist for numpy points:
def dist_np(point1, point2):
    return np.sqrt(np.sum((point2 - point1)**2))

#supremum dist for torch tensors:
def dist_sup_tc(b1, d1, b2, d2):
    # Calculate the sup norm between points (b1, d1) and (b2, d2)
    return torch.max(torch.abs(b1 - b2), torch.abs(d1 - d2))

def d_bottleneck0(point_cloud, dgm, dgm2): # got_loss=1 if got loss, =0 if loss does not depend on dgm
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
      loss = torch.abs(dist(point1_dgm1, point2_dgm1) - dgm2['dgms'][0][j][1])
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal -> backprop through loss would give 0 -> goal: make points further from diag
        #new_bdist = torch.abs(dist(point1_dgm2, point2_dgm2) - 0.)/2
        loss = 0
        got_loss = 0
      else: #then  j==-1, so the i-th point from dgm1 is matched to the diagonal
        loss = dist(point1_dgm1, point2_dgm1)/2.

    return loss, got_loss

def d_bottleneck1(point_cloud, dgm, dgm2): # got_loss=1 if got loss, =0 if loss does not depend on dgm
    got_loss = 1
    if len(dgm['dgms'][1])==0: return 0, 0
    if len(dgm2['dgms'][1])==0: dgm2['dgms'][1] = [[0.,0.]] #print("error2")
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

    if i>=0 and j>=0:
      loss = dist_sup_tc(birth_dgm1, death_dgm1, birth_dgm2, death_dgm2)
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal
        loss = 0
        got_loss = 0
      else: #then j==-1, so the i-th point from dgm1 is matched to the diagonal
        loss = (death_dgm1 - birth_dgm1)/2.

    return loss, got_loss

#return Reininghaus kernel ksigma: (could make it slightly faster with different functions for each dgm (dgm2 does not need backpropagation)), but let it same for all dgms
def ksigma0(point_cloud, point_cloud2, dgm, dgm2): #maxdim of both dgms: 0
    sigma = 0.05
    ksigma = 0
    ## use formula for k_sigma from paper (https://arxiv.org/pdf/1412.6821.pdf):
    for i in range(len(dgm['gens'][0])):
        # pt in dgm: (0,d), d=dist(p1,p2)
        p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]
        d1 = dist(p1, p2)
        for j in range(len(dgm2['gens'][0])):
           #pt in dgm2: (0,d), d=dist(q1,q2)
           q1, q2 = point_cloud2[dgm2['gens'][0][j][1]], point_cloud2[dgm2['gens'][0][j][2]]
           d2 = dist(q1, q2)
           ksigma += torch.exp(-dist_2(0, d1, 0, d2)/(8*sigma)) - torch.exp(-dist_2(0, d1, d2, 0)/(8*sigma))

    ksigma *= 1/(8*3.141592*sigma)
    return ksigma

#return pseudo-distance that comes from ksigma and squared, dsigma**2:
def dsigma0(point_cloud, point_cloud2, dgm, dgm2):
    k11 = ksigma0(point_cloud, point_cloud, dgm, dgm)
    #k22 = ksigma(point_cloud2, point_cloud2)
    k12 = ksigma0(point_cloud, point_cloud2, dgm, dgm2)
    #return k11 + k22 - 2*k12
    return k11 - 2*k12 #no need of k22 since no backpropagation through it (fixed point cloud)

#the only diff from ksigma (deg0) is how to take the pt of dgms (b,d) wrt the pts of the point clouds, for the backpropagation
def ksigma1(point_cloud, point_cloud2, dgm, dgm2): #maxdim of both dgms: 1
    sigma = 0.05
    if len(dgm['dgms'][1])==0 or len(dgm['dgms'][1])==0: return 0, 0

    ksigma1 = 0
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

          ksigma1 += torch.exp(-dist_2(b1, d1, b2, d2)/(8*sigma)) - torch.exp(-dist_2(b1, d1, d2, b2)/(8*sigma))

    ksigma1 *= 1/(8*3.141592*sigma)
    return ksigma1, 1

def dsigma1(point_cloud, point_cloud2, dgm, dgm2):
    k12, gotloss = ksigma1(point_cloud, point_cloud2, dgm, dgm2)
    if gotloss == 0: return 0, 0
    # if both dgm and dgm2 have at least 1 point:
    k11, gotloss = ksigma1(point_cloud, point_cloud, dgm, dgm)
    return k11 + (-2.) * k12, 1

def density(point_cloud, dgm, sigma, scale, x):
  tot = 0
  density_x = 0 #density at x
  for i in range(len(dgm['dgms'][0])-1):
    p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]] #pt (0,d) with d=dist(p1,p2) (euclidean dist)
    d = dist(p1, p2) #pt of pt cloud is (0,d)
    density_x += d**4 * torch.exp(-((d-x)/sigma)**2)

  return density_x * scale

def loss_density(point_cloud, point_cloud2, dgm, dgm2, sigma, scale, maxrange, npoints, plot): #dgm of deg0
  xs = np.linspace(0., maxrange, npoints)
  loss = 0
  ## compute difference between both functions in 100 pts (those given by xs)
  dens=[]
  dens0=[]
  for x in xs:
    dx = density(point_cloud, dgm, sigma, scale, x)
    d0x = density(point_cloud2, dgm2, sigma, scale, x)
    loss += (dx - d0x)**2
    if plot:
      with torch.no_grad():
        dens.append(dx.detach().numpy())
        dens0.append(d0x.detach().numpy())

  if plot: #plot both density functions (blue: the one of dgm being learned)
    plt.plot(xs, dens, color='blue')
    plt.plot(xs, dens0, color='red')
    plt.show()

  return loss

## use: sigma=0.2, scale=0.002, maxrange=35., npoints=100

def loss_persentropy0(point_cloud, dgm, dgm2, delta): #dgm of deg0. only looks at points with pers=|d|>delta (for avoiding large gradients) (in both dgms)
  L = 0
  for i in range(len(dgm['dgms'][0])-1):
    with torch.no_grad(): pers1 = dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]])
    if pers1>delta: L += dist(point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]])

  with torch.no_grad():
    if L==0: return 0, 0

  pers = 0
  for i in range(len(dgm['dgms'][0])-1):
    p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]] #pt (0,d) with d=dist(p1,p2) (euclidean dist)
    with torch.no_grad(): d1 = dist(p1, p2)
    if d1>delta: pers += dist(p1, p2) * torch.log(dist(p1, p2)/L) #pt of pt cloud is (0,dist(p1, p2))

  ##get pers entropy of dgm2:
  L2 = 0
  pers2 = 0
  for i in range(len(dgm2['dgms'][0])-1):
    if dgm2['dgms'][0][i][1] > delta: L2 += dgm2['dgms'][0][i][1]
  if L2==0: return (pers/L)**2, 1
  for i in range(len(dgm2['dgms'][0])-1):
    if dgm2['dgms'][0][i][1] > delta: pers2 += dgm2['dgms'][0][i][1] * math.log(dgm2['dgms'][0][i][1] / L2)

  return (pers/L - pers2/L2)**2, 1

def loss_persentropy1(point_cloud, dgm, dgm2, delta): #dgm of deg1. returns loss, got_loss (0 if did not get it). only looks at points with pers=|d-b|>delta (in both dgms) (for avoiding large gradients)
  if len(dgm['dgms'][1])==0: return 0, 0 #no loss if dgm has no off-diag points

  #entropy of dgm:
  L = 0
  for i in range(len(dgm['dgms'][1])):
        # pt in dgm: (b1,d1), with b1, d1 = dist(p2, p1), dist(dist(p3, p4)
        p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]], point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]
        b1 = dist(p1,p2)
        d1 = dist(p3,p4)
        with torch.no_grad(): pers1 = d1-b1
        if pers1 > delta: L += d1 - b1

  with torch.no_grad():
    if L==0: return 0, 0

  pers = 0
  for i in range(len(dgm['gens'][1])):
        # pt in dgm: (b1,d1), with b1, d1 = dist(p2, p1), dist(dist(p3, p4)
        p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]], point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]
        b1 = dist(p1,p2)
        d1 = dist(p3,p4)
        with torch.no_grad(): pers1 = d1-b1
        if pers1 > delta: pers += (d1-b1) * torch.log((d1-b1)/L)

  if len(dgm2['dgms'][1])==0: return (pers/L)**2, 1 # the pers of dgm2 is taken as 0

  #entropy of dgm2:
  L2 = 0
  for i in range(len(dgm2['dgms'][1])):
        if dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]>delta: L2 += dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]

  if L2==0: return (pers/L)**2, 1 # the pers of dgm2 is taken as 0
  pers2 = 0
  for i in range(len(dgm2['dgms'][1])):
        lifespan = dgm2['dgms'][1][i][1] - dgm2['dgms'][1][i][0]
        if lifespan>delta: pers2 += torch.tensor(lifespan) * torch.log(torch.tensor(lifespan/L2))

  return (pers/L - pers2/L2)**2, 1
