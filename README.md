# Topo-GEN: Topology-Informed Generative Models

This repository introduces a new approach for training generative models leveraging computational topology. Specifically, it employs persistence diagrams, a mathematical tool that captures the "shape" of point clouds within any metric space. This union seeks to provide models with previously unexplored information about the structure of the true and the generated data, in order to enhance their training process. 

To this aim, we introduce `topogen`, a library that provides a new family of topological regularizers. These functions can be implemented as loss functions in any machine learning problem that involves learning to generate a point cloud from another input point cloud, regardless of their dimensions, number of points, or format. In fact, a key aspect of persistence diagrams is that they translate the topological and geometrical features of point clouds into an object that can be efficiently compared among different point clouds, even if they lie in different spaces and have different numbers of points.

Furthermore, we provide additional files to test topological regularizers:
- `synthetic_experiments.py` provides synthetic proof-of-concept experiments with point clouds in 2D, and a "virtual playground" to test topological regularizers on arbitrary point clouds in 2D.
- `topovae_experiments.py` allows to train and test variational autoencoders (VAEs) using topological regularizers.

Algorithms in this repository are fully supported to run on the GPU.

## Installation

To install the `topogen` library:
```
pip install git+https://github.com/JackBJ23/Topo-GEN.git
```

To clone the repository and run the tests, for instance on Google Colab, do:
```
!git clone https://github.com/JackBJ23/Topo-GEN.git
%cd Topo-GEN
!pip install -r requirements.txt
```

## Basic usage

The library provides seven topological regularizers, each computing a different measure of dissimilarity between the persistence diagram of the learnable point cloud and the ground truth persistence diagram. These functions are:
- `loss_bottleneck0`, `loss_bottleneck1`: bottleneck distance for homology degrees 0 and 1, respectively.
- `loss_persentropy0`, `loss_persentropy1`: squared difference between persistence entropies for homology degrees 0 and 1, respectively.
- `loss_dsigma0`, `loss_dsigma1`: Reininghaus dissimilarity for homology degrees 0 and 1, respectively.
- `loss_density`: difference between the 4SGDE density functions of the two persistence diagrams.

We provide a unified class that allows the simple and efficient combination of these functions. To use it, do:
```
from topogen import TopologicalLoss

topo_loss = TopologicalLoss(topo_weights)
loss, gotloss = topo_loss.compute_loss(point_cloud, true_point_cloud)
```
Where `topo_weights` is a 7-element list of weights associated with the topological regularizers. The function `topo_loss.compute_loss` returns the weighted sum of the topological losses as a scalar tensor (`loss`). Specifically, it computes (topoloss_i * topo_weights[i], only if topo_weights[i] ≠ 0) for i = 0, 1, ..., 6, where the topological losses are in the same order as in the list above. If a weight is set to 0, its corresponding function is not used. Additionally, `topo_loss.compute_loss` returns a boolean (`gotloss`) that is `True` if the loss depends on the learnable point cloud and `False` otherwise. 

Furthermore, `point_cloud` is the learnable point cloud or output of a machine learning model, and `true_point_cloud` is the ground truth point cloud, both expected to be torch tensors of shape `(number of points, dimensions for each point)`. Additional attributes controlling the topological functions can be set, see [`topogen/topo_functions.py`](https://github.com/JackBJ23/Topo-GEN/blob/main/topogen/topo_functions.py) for details. 

For a more manual control of individual topological functions, do, for instance (the same principle applies to the other regularizers):
```
from topogen import loss_bottleneck0
loss, gotloss = loss_bottleneck0(point_cloud, point_cloud2)
```
Where `point_cloud` is the learnable point cloud or output of a machine learning model and `point_cloud2` is the ground truth point cloud, both also expected to be torch tensors with shapes `(number of points, dimension of each point)`. See [`topogen/topo_functions.py`](https://github.com/JackBJ23/Topo-GEN/blob/main/topogen/topo_functions.py) for details about additional optional arguments.

The library also includes visualization tools to observe the impact of topological regularizers on generative models and 2D point clouds, see [`topogen/visualizations.py`](https://github.com/JackBJ23/Topo-GEN/blob/main/topogen/visualizations.py).

## Synthetic experiments

To visualize the information captured by the topological regularizers, we provide three proof-of-concept examples. In each case, we start with a random point cloud in 2D, and we set their coordinates as learnable parameters, which are updated through gradient descent. In each test, we impose a ground truth persistence diagram that captures some topological properties. At each training step we compute the bottleneck losses of degrees 0 and 1, which are used to update the coordinates of the point cloud.

In the first test (left), we start with 5 clusters, and the ground truth persistence diagram indicates the presence of 3 clusters. The point cloud thus deforms itself to reach this goal.

In the second test (middle), we start with 2 clusters, and the ground truth persistence diagram indicates the presence of 4 clusters. 

In the third test (right), we start with 2 segments, and the ground truth persistence diagram indicates the presence of one circle.

In each case, we see that the topological loss teaches the point cloud to continuously deform and rearrange itself to reach the desired topological properties.

<div style="display: flex; justify-content: space-between;">
    <img src="assets/synthetic1_video.gif" width="30%">
    <img src="assets/synthetic2_video.gif" width="30%">
    <img src="assets/synthetic3_video.gif" width="30%">
</div>

To run these experiments:
```
!python synthetic_experiments.py
```

Furthermore, we provide a "virtual playground" where the user can run other synthetic experiments and test different combinations of topological regularizers on arbitrary point clouds in the plane. To do so, run:
```
import numpy as np
from synthetic_experiments import synthetic_test

synthetic_test(point_cloud, point_cloud_true, topo_weights=[1.,1.,0.,0.,0.,0.,0.], num_steps=2000, lr=0.001, test_name="test", num_save=50, x1=-10., x2=40., y1=-40., y2=40.)
```
Where `point_cloud` and `point_cloud_true` are expected to be numpy arrays of shape `(number of points, dimension of each point)`, e.g., an input point cloud of three points in 2D can be `np.array([[0., 0.], [1., 0.], [0., 1.]])`. Only the first two arguments are required. The function saves figures of the true point cloud and its persistence diagram, initial learnable point cloud and its diagram, final point cloud and its diagram, loss evolution, and an animation of the point cloud evolution. 

## Working principle of topology-informed generative models

The working principle of topology-informed generative models is illustrated below, using a variational autoencoder as an example, and can be summarized as follows. At each training step, a batch of N data points is given to the model and N points are generated as output. When implementing the topological regularizers, at each training iteration we compute the persistence diagram of the input batch and the persistence diagram of the generated batch, both viewed as point clouds (where each point is a data element, e.g., an image viewed as a 1D vector). The regularizer captures a measure of dissimilarity between the two persistence diagrams. Hence, the modification of the weights of the generative model through gradient descent aims to produce data with a spatial distribution that "looks like" the distribution of the true data. There is an extension of this method, also illustrated below, where the topological regularizers are applied on the batch of latent vectors instead of the final outputs in order to control the distribution of the latent space.

<img src="assets/topovae_architecture.png" alt="TopoVAE Architecture" width="700"/>

## Example: TopoVAE

We provide a file to directly train and test VAEs on the FashionMNIST dataset using topological regularizers. To do so, run:
```
!python topovae_experiments.py --topo_weights w_bottleneck0,w_bottleneck1,w_entropy0,w_entropy1,w_ksigma0,w_ksigma1,w_density
```
Where `topo_weights` is the list of weights associated with each topological loss. To manually set additional arguments, do `!python topovae_experiments.py --help` for details. The file automatically saves plots comparing true images, VAE-generated images, and TopoVAE-generated images, and saves plots comparing the evolution of the BCE losses for the two models.

## Some results

In our experiments, we observe interesting behaviors. For instance, a VAE trained with the bottleneck loss for homology degrees 0 and 1 yields improved image quality and diversity in early training, as shown below (taken at training step 50). Left: input data, middle: output from standard VAE, right: output from TopoVAE. 

<img src="assets/imgs_generated.png" alt="Images Generated" width="700"/>

This performance correlates with a faster decay of the Binary Cross-Entropy loss during the first 100 training steps, as shown below. 

<img src="assets/losses.png" alt="Loss Evolution" width="300"/>

Furthermore, as shown below, we observe an interesting behavior: when applying topological regularizers, the latent vectors seem to redistribute themselves in a more organized way according to their classes (right), compared to the standard VAE (left).

<img src="assets/latent_dist.png" alt="Latent Distribution" width="700"/>

We believe that the integration of topology into generative models through differentiable loss functions represents an exciting new direction, with our initial results suggesting promising potential for future applications.

## More information

For more details about persistence diagrams, topological regularizers, stability and differentiability properties, and more information about the meaning of these functions, see B. Jedlicki, Jack. [2024](https://diposit.ub.edu/dspace/handle/2445/217016).

## References

If you found this library useful in your research, please consider citing. 

```bibtex
@article{benarroch2024topogen,
  title={Topogen: topology-informed generative models},
  author={Benarroch Jedlicki, Jack},
  year={2024}
}
