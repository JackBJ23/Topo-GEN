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

The library provides seven topological regularizers, each computing a different measure of dissimilarity between the diagram of the learnable point cloud and the ground truth persistence diagram. To use them, do:
```
from topogen import *

loss_bottleneck0(point_cloud, point_cloud2, dgm, dgm2, device)
loss_bottleneck1(point_cloud, point_cloud2, dgm, dgm2, device)
loss_persentropy0(point_cloud, point_cloud2, dgm, dgm2, device, delta0=0.01)
loss_persentropy1(point_cloud, point_cloud2, dgm, dgm2, device, delta1=0.01)
loss_dsigma0(point_cloud, point_cloud2, dgm, dgm2, device, sigma0=0.05)
loss_dsigma1(point_cloud, point_cloud2, dgm, dgm2, device, sigma1=0.05)
loss_density(point_cloud, point_cloud2, dgm, dgm2, device, sigma=0.2, scale=0.002, maxrange=35., npoints=30)
```
For each function, the input arguments and outputs are the following:

#### Input arguments:
- **Required:**
  - `point_cloud` (torch.Tensor): Learnable point cloud or output of a machine learning model. Expected shape `(number of points, dimension of each point)`.
  - `point_cloud2` (torch.Tensor): Ground truth point cloud. Expected shape `(number of points, dimension of each point)`.
- **Optional:**
  - `dgm`: Persistence diagram for the first point cloud. If `None`, it will be computed.
  - `dgm2`: Persistence diagram for the true point cloud. If `None`, it will be computed.
  - `device`: The device to use for computations. Defaults to `"cpu"`.
  - Additional arguments that control the topological functions.

#### Output:
- `loss`: The computed loss value as a scalar tensor.
- `gotloss`: A status flag, `True` if the loss depends on the learnable point cloud and has to be added to the total loss, `False` otherwise.

Furthermore, we recommend pre-computing the persistence diagrams before training to enhance training speed. It is also preferable to compute the persistence diagram of the learnable point cloud only once before for calling the functions. To generate a persistence diagram, do:
```
from topogen import get_dgm

dgm = get_dgm(point_cloud, deg)
```
Where the shape of the point cloud is expected to be `(number of points, dimension of each point)`, and `deg` is the homology degree (0 or 1), with 1 the more general option. 

Additionally, we have unified all the topological regularizers into a single function, `topo_losses`, to enable their efficient and straightforward combination. To use it, do:
```
from topogen import topo_losses

topoloss, gotloss = topo_losses(points, true_points, topo_weights)
```
The function returns the total topological loss, `topoloss`, and `gotloss`, which is `True` if the total loss depends on the input point cloud and `False` otherwise. The required arguments are:
- `points` (torch.Tensor): Learnable point cloud or output of a machine learning model. Expected shape `(number of points, additional dimensions)`.
- `true_points` (torch.Tensor): Ground truth point cloud. Expected shape `(number of points, additional dimensions)`.
- `topo_weights`: List of weights associated with each topological loss:
  `[w_topo0, w_topo1, w_pers0, w_pers1, w_dsigma0, w_dsigma1, w_density0]`. If a weight is set to `0`, its corresponding topological function is not used.

See [`topogen/topo_functions.py`](https://github.com/JackBJ23/Topo-GEN/blob/main/topogen/topo_functions.py) for details about its optional keyword arguments. 

## Synthetic experiments

To visualize the information captured by the topological regularizers, we provide three proof-of-concept examples. In each case, we start with a random point cloud in 2D, and we set their coordinates as learnable parameters, which are updated through gradient descent. In each test, we impose a ground truth persistence diagram that captures some topological properties. At each training step we compute the persistence diagram of the learnable point cloud and measure its dissimilarity with the ground truth diagram using the bottleneck loss. Using backpropagation and gradient descent to minimize this loss, we update the coordinates of the point cloud. In each case, we see that the topological loss teaches the point cloud to continuously deform and rearrange itself to reach the desired topological properties. 

In the first test (left), we start with 5 clusters, and the ground truth persistence diagram indicates the presence of 3 clusters. The point cloud thus deforms itself to reach this goal. 

In the second test (middle), we start with 2 clusters, and the ground truth persistence diagram indicates the presence of 4 clusters. 

In the third test (right), we start with 2 segments, and the ground truth persistence diagram indicates the presence of one circle. 

<div style="display: flex; justify-content: space-between;">
    <img src="assets/synthetic1_video.gif" width="30%">
    <img src="assets/synthetic2_video.gif" width="30%">
    <img src="assets/synthetic3_video.gif" width="30%">
</div>

To run these experiments:
```
!python synthetic_experiments.py
```

Furthermore, we provide a "virtual playground" where the user can run other synthetic experiments and experiment with different combinations of topological regularizers on arbitrary point clouds in the plane. To do so, run:
```
import numpy as np
from synthetic_experiments import synthetic_test

synthetic_test(point_cloud, point_cloud_true, topo_weights=[1.,1.,0.,0.,0.,0.,0.], num_steps=2000, lr=0.001, test_name="test", device="cpu", num_save=50, x1=-10., x2=40., y1=-40., y2=40.)
```
Where `point_cloud` and `point_cloud_true` are expected to be numpy arrays of shape `(number of points, dimension of each point)`, e.g., an input point cloud of three points in 2D can be `np.array([[0., 0.], [1., 0.], [0., 1.]])`. Only the first two arguments are required. The function saves figures of the initial true point cloud, initial true persistence diagram, initial learnable point cloud, final point cloud, its final persistence diagram, loss evolution, and an animation of the point cloud evolution. 

## Working principle of topology-informed generative models

The working principle of topology-informed generative models is illustrated below, using a variational autoencoder as an example, and can be summarized as follows. Assume the generative model produces images or any data that can be represented as an array of real numbers, and view each data element as an individual vector or point. At each training step, a batch of N data points is given to the model and N points are generated as output. When implementing the topological regularizers, in each training iteration we compute the persistence diagram of the batch of N true points, and the persistence diagram of the N generated points, both viewed as point clouds. The two resulting persistence diagrams are then compared using some measure of dissimilarity, and the regularizer captures this measure. Hence, the modification of the weights of the generative model through gradient descent aims to produce data with a spatial distribution that looks like the distribution of the true data. Furthermore, there is also an extension of this method, also illustrated below, which relies on applying the topological regularizers on the batch of latent vectors instead of the final outputs of the model, in order to control the distribution in the latent space.

<img src="assets/topovae_architecture.png" alt="TopoVAE Architecture" width="700"/>

## Example: TopoVAE

We provide a way to directly train and test VAEs on the FashionMNIST dataset using topological regularizers. To do so, run:
```
!python topovae_experiments.py --topo_weights w_bottleneck0,w_bottleneck1,w_entropy0,w_entropy1,w_ksigma0,w_ksigma1,w_density
```
Where `topo_weights` is the list of weights associated with each topological loss. Other arguments can be manually set, do `!python train.py --help` for details. The file automatically saves plots of true, VAE, and TopoVAE-generated images for each training epoch, for evaluation in each epoch, and it saves a plot of the evolution of the BCE losses for the VAE and the TopoVAE. 

## Some results

In our experiments, we observe interesting behaviors. For instance, a VAE trained wit the bottleneck loss for homology degrees 0 and 1 yields improved image quality and diversity in early training, as shown below (taken at training step 50). Left: input data, middle: output from standard VAE, right: output from TopoVAE. 

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
