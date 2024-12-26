# Topo-GEN: Topology-Informed Generative Models

This repository presents a new approach for training generative models leveraging computational topology. In particular, we use persistence diagrams, a mathematical tool providing a description of the "shape" of point clouds in any metric space. With shape we refer to features such as clusters, loops, higher-dimensional holes, or other properties. This union seeks to provide models with previously unexplored information about the structure of the true and the generated data distributions. 

This repository implements persistence diagrams into the training process of generative models using several new approaches. 

# Examples: Synthetic Experiments

Additionally, to provide a proof-of-concept example, we have included a file where we apply the topological loss terms into synthetic point clouds in the plane. These synthetic experiments show that the point clouds learn to continuously deform to acquired the desired topological features. 

<div style="display: flex; justify-content: space-between;">
    <img src="assets/synthetic1_video.gif" width="30%">
    <img src="assets/synthetic2_video.gif" width="30%">
    <img src="assets/synthetic3_video.gif" width="30%">
</div>

# Architecture of Topology-Informed Models

Briefly explained, applying topological regularizers into generative models works as follows. 

<img src="assets/topovae_architecture.png" alt="TopoVAE Architecture" width="700"/>

# 

In our experiments, we find interesting behaviors. For instance, a VAE trained with parameters [15.0, 15.0, 0., 0., 0., 0., 0.] yields an improved image quality and diversity in early training, as shown below (left: input data, middle: output from VAE, right: output from TopoVAE). 

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
