# Topo-GEN: Topology-Informed Generative Models

This repository presents a new approach for training generative models leveraging computational topology. In particular, we use persistence diagrams, a mathematical tool arising from computational algebraic topology. Persistence diagrams provide a description of the "shape" of point clouds in any metric space, and with shape we refer to features such as the presence of clusters and their pairwise distances, loops, higher-dimensional holes, among with other properties. This repository implements, for the first time to the best of our knowledge, persistence diagrams into the training process of generative models using several different approaches. 

Additionally, to provide a proof-of-concept example, we have included a file where we apply the topological loss terms into synthetic point clouds in the plane. These synthetic experiments show that the point clouds learn to continuously deform to acquired the desired topological features. 
