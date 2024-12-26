# Topo-GEN: Topology-Informed Generative Models

This repository presents a new approach for training generative models leveraging computational topology. In particular, we use persistence diagrams, a mathematical tool providing a description of the "shape" of point clouds in any metric space. With shape we refer to features such as clusters, loops, higher-dimensional holes, among other properties. This union seeks to provide models with previously unexplored information about the structure of the true and the generated data distribution, in order to enhance their training process. 

In particular, we introduce a new family of topological regularizers that can be implemented into the training process of any generative model. Algorithms in this repository are fully supported to run on the GPU.

# Proof-of-concept Example: Synthetic Experiments

To visualize the information captured by the topological regularizers, we provide three proof-of-concept examples. In each case, we start with a random point cloud in 2D, and we set their coordinates as learnable parameters, updated through gradient descent. In particular, we impose a ground truth persistence diagram that captures some topological properties. In each training step, we compute the persistence diagram of the learnable point cloud, and measure its dissimilarity with the ground truth diagram using the bottleneck loss. Using backpropagation and gradient descent to minimize this loss, we update the coordinates of the point cloud. Overall, we see that the topological loss teaches the point cloud to continuously deform and rearrange itself to reach the desired topological properties. 

In the first test (left), we start at 5 clusters, and the ground truth persistence diagram indicates the presence of 3 clusters. The point cloud thus deforms itself to reach this goal. 

In the second test (middle), we start at 2 clusters, and the ground truth persistence diagram indicates the presence of 4 clusters. 

In the third test (right), we start at 2 segments, and the ground truth persistence diagram indicates the presence of one circle. 

<div style="display: flex; justify-content: space-between;">
    <img src="assets/synthetic1_video.gif" width="30%">
    <img src="assets/synthetic2_video.gif" width="30%">
    <img src="assets/synthetic3_video.gif" width="30%">
</div>

To run these synthetic experiments:
```
--python synthetic_experiments.py
```
To run new synthetic experiments with new point clouds:
```
--point cloud (optional) --true_point_cloud (optional) --loss_parameters (optional)
```
The algorithm will directly convert the true point cloud into the ground truth diagram capturing its properties (to avoid the need of manually designing the diagram). Leaving --loss_parameters blank will lead to using the bottleneck loss of degrees 0 and 1. Otherwise, fill the vector of weights [w_bottleneck0, w_bottleneck1, w_entropy0, w_entropy1, w_ksigma0, w_ksigma1, w_density] with some non-negative float values. Each is asociated with a different topological loss. 

# Architecture of Topology-Informed Models

Briefly explained, applying topological regularizers into generative models works as follows. 

<img src="assets/topovae_architecture.png" alt="TopoVAE Architecture" width="700"/>

# Example: TopoVAE

In our experiments, we find interesting behaviors. For instance, a VAE trained wit the bottleneck loss for homology degrees 0 and 1 yields improved image quality and diversity in early training, as shown below (taken at training step 50). Left: input data, middle: output from standard VAE, right: output from TopoVAE. 

<img src="assets/imgs_generated.png" alt="Images Generated" width="700"/>

This performance correlates with a faster decay of the Binary Cross-Entropy loss during the first 100 training steps, as shown below. 

<img src="assets/losses.png" alt="Loss Evolution" width="300"/>

Furthermore, as shown below, we observe an interesting behavior: when applying topological regularizers, the latent vectors seem to redistribute them selves in a more organized way according to their classes---compared to the standard VAE.

<img src="assets/latent_dist.png" alt="Latent Distribution" width="700"/>

We believe that the integration of topology into generative models through differentiable loss functions represents a promising new direction, with our initial results suggesting promising potential for future applications.

# Further Information

# References

If you found this library useful in your research, please consider citing. 

```bibtex
@article{benarroch2024topogen,
  title={Topogen: topology-informed generative models},
  author={Benarroch Jedlicki, Jack},
  year={2024}
}
