# Topo-GEN: Topology-Informed Generative Models

This repository presents a new approach for training generative models leveraging computational topology. In particular, we use persistence diagrams, a mathematical tool arising from computational algebraic topology. Persistence diagrams provide a description of the "shape" of point clouds in any metric space, and with shape we refer to features such as the presence of clusters and their pairwise distances, loops, higher-dimensional holes, among with other properties. This repository implements persistence diagrams into the training process of generative models using several new approaches. 

Additionally, to provide a proof-of-concept example, we have included a file where we apply the topological loss terms into synthetic point clouds in the plane. These synthetic experiments show that the point clouds learn to continuously deform to acquired the desired topological features. 

<img src="path/to/demo.gif" width="300"/>

Briefly explained, applying topological regularizers into generative models works as follows. 

<img src="assets/topovae_architecture.png" alt="TopoVAE Architecture" width="700"/>

In our experiments, we find interesting behaviors. For instance, a VAE trained with parameters [15.0, 15.0, 0., 0., 0., 0., 0.] yields an improved image quality and diversity in early training, as shown below (left: input data, middle: output from VAE, right: output from TopoVAE). 

<img src="assets/imgs_generated.png" alt="Images Generated" width="700"/>

This performance correlates with a faster decay of the Binary Cross-Entropy loss during the first 100 training steps, as shown below. 

<img src="assets/losses.png" alt="Loss Evolution" width="300"/>

Furthermore, as shown below, we observe an interesting behavior: when applying topological regularizers, the latent vectors seem to redistribute them selves in a more organized way according to their classes---compared to the standard VAE.

<img src="assets/latent_dist.png" alt="Latent Distribution" width="700"/>

We believe that the integration of topology into generative models through differentiable loss functions represents a promising new direction, with our initial results suggesting promising potential for future applications.
