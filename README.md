# Topo-GEN: Topology-Informed Generative Models

This repository presents a new approach for training generative models leveraging computational topology. In particular, we use persistence diagrams, a mathematical tool providing a description of the "shape" of point clouds in any metric space. With shape we refer to features such as clusters, loops, or higher-dimensional holes, among other properties. This union seeks to provide models with previously unexplored information about the structure of the true and the generated data, in order to enhance their training process. 

To this aim, we introduce a new family of topological regularizers that can be implemented in the training process of any generative model. More in general, they can be implemented as loss functions in any machine learning problem that involves learning to generate a point cloud from another input point cloud, regardless of their dimensions, number of points, or format. In fact, a key aspect of persistence diagrams is that they translate the topological and geometrical features of point clouds into an object that can be efficiently compared among different point clouds, even if they lie in different spaces and have different numbers of points.

Algorithms in this repository are fully supported to run on the GPU. 

## Installation

To install the repository:
```
pip install https://github.com/JackBJ23/Topo-GEN.git
```
Or, to run directly on Google Colab:
```
!git clone https://github.com/JackBJ23/Topo-GEN.git
%cd Topo-GEN
!pip install -r requirements.txt
```

## Proof-of-concept example: synthetic experiments

To visualize the information captured by the topological regularizers, we provide three proof-of-concept examples. In each case, we start with a random point cloud in 2D, and we set their coordinates as learnable parameters, updated through gradient descent. In particular, we impose a ground truth persistence diagram that captures some topological properties. In each training step, we compute the persistence diagram of the learnable point cloud, and measure its dissimilarity with the ground truth diagram using the bottleneck loss. Using backpropagation and gradient descent to minimize this loss, we update the coordinates of the point cloud. In each case, we see that the topological loss teaches the point cloud to continuously deform and rearrange itself to reach the desired topological properties. 

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
To run new synthetic experiments with new point clouds:
```
!python synthetic_experiments.py --point_cloud (add) --true_point_cloud (add)
```
For instance, an input point cloud of three points in 2D can be [[0., 0.], [1., 0.], [0., 1.]]. The algorithm will directly convert the true point cloud into the ground truth diagram capturing its properties (to avoid the need for manually designing the diagram). 

## Working principle of topology-informed generative models

The working principle of topology-informed variational autoencoders (or other generative models) is illustrated below, and can be summarized as follows. Take a generative model that produces images or any type of data that can be represented as an array of real numbers, and view each data element as an individual point. During each training iteration, a batch of N data points is given to the model and N points are generated as output. Then, some measure of dissimilarity between the true and the generated data (e.g., binary cross-entropy loss) is computed and used as a loss function. When implementing the topological regularizers, in each training iteration we compute the persistence diagram of the batch of N true points, and the persistence diagram of the N generated points, both viewed as point clouds. The two resulting persistence diagrams are then compared using some measure of dissimilarity, and the regularizer captures this measure. Hence, the modification of the weights of the generative model through gradient descent aims to produce data with a spatial distribution that looks like the distribution of the true data. Furthermore, there is also an extension of this method, also illustrated below, which relies on applying the topological regularizers on the batch of latent vectors instead of the final outputs of the model, in order to control the distribution in the latent space.

<img src="assets/topovae_architecture.png" alt="TopoVAE Architecture" width="700"/>

## Basic usage

There are seven topological regularizers, presented below. Note that `point_cloud` is the learnable point cloud or output of a machine learning model; `dgm` is its persistence diagram; `true_point_cloud` is the ground truth point cloud, and `true_dgm` is its diagram. The other arguments are optional and control the topological functions. 
```
from topo_functions import *
loss_bottleneck0(point_cloud, dgm, true_dgm)
loss_bottleneck1(point_cloud, dgm, true_dgm)
loss_persentropy0(point_cloud, dgm, true_dgm, device, delta0=0.01)
loss_persentropy1 (point_cloud, dgm, true_dgm, device, delta1=0.01)
loss_dsigma0(point_cloud, true_point_cloud, dgm, true_dgm, device, sigma0=0.05),
loss_dsigma1(point_cloud, true_point_cloud, dgm, true_dgm, device, sigma1=0.05)
loss_density(point_cloud, true_point_cloud, dgm, true_dgm, device, sigma=0.2, scale=0.002, maxrange=35., npoints=30)
```
Each function returns two values: `loss, gotloss`. If `gotloss` is 1, the loss value depends on the learnable point cloud and can be added to the total loss. If `gotloss` is 0, the topological loss only depends on ground truth data and does not need to be added to the total loss. To generate a persistence diagram, do:
```
dgm = get_dgm(point_cloud, deg)
```
Where the shape of the point cloud is expected to be (Number of points, Dimension of each point), and `deg` is the homology degree (0 or 1, where 1 is the more general option). 

Additionally, we have unified all the topological regularizers into a single function, `topo_losses`, in order to combine them in a straightforward way. To use it, do:
```
from topo_functions import topo_losses
topoloss = topo_losses(point_cloud, true_point_cloud, topo_weights, deg=1, dgm_true=None, pers0_delta=0.001, pers1_delta=0.001, dsigma0_scale=0.05, dsigma1_scale=0.05, density_sigma=0.2, density_scale=0.002, density_maxrange=35., density_npoints=30, device="cpu")
```
Details about this function are given below. 

## Keyword arguments for topo_losses

The `topo_losses` function combines the seven topological regularizers into a single, unified function.
#### Required Arguments
- **`points`**: Learnable point cloud or output of a machine learning model. 
- **`true_points`**: Ground truth point cloud.
- **`topo_weights`**: List of weights associated with each topological loss:
  `[w_topo0, w_topo1, w_pers0, w_pers1, w_dsigma0, w_dsigma1, w_density0]`.
  - If a weight is set to `0`, its corresponding topological function is not used.
#### Optional Arguments
- **`deg`**: Homology degree (`0` or `1`, where `1` is the more general option).
- **`dgm_true`**: Persistence diagram of the ground truth data. If `None`, it is calculated inside the function.
#### Parameters for Topological Functions
The following parameters are set to reference values by default but can be modified depending on the dataset, model, or other considerations:
- **`pers0_delta`**: Default = `0.001`
- **`pers1_delta`**: Default = `0.001`
- **`dsigma0_scale`**: Default = `0.05`
- **`dsigma1_scale`**: Default = `0.05`
- **`density_sigma`**: Default = `0.2`
- **`density_scale`**: Default = `0.002`
- **`density_maxrange`**: Default = `35.0`
- **`density_npoints`**: Default = `30`
For details about the meaning of these values, see B. Jedlicki, Jack. [2024](https://diposit.ub.edu/dspace/handle/2445/217016).
#### Device
- Specify `"cuda"` or `"cpu"` for the device on which to perform the calculations.

## Example: TopoVAE

We provide a way to directly train VAEs on the FashionMNIST dataset using topological regularizers. To do so, run:
```
!python train.py --topo_weights w_bottleneck0,w_bottleneck1,w_entropy0,w_entropy1,w_ksigma0,w_ksigma1,w_density
```
Where `topo_weights` is the list of weights associated with each topological loss, and if a weight is set to 0, its corresponding topological function is not used. Other arguments can be manually set, do `!python train.py --help` for details. 

Furthermore, the file will automatically save plots of true, VAE, and TopoVAE-generated images for each training epoch, for evaluation in each epoch, and it will save the evolution of the BCE losses.

## Some results

In our experiments, we find interesting behaviors. For instance, a VAE trained wit the bottleneck loss for homology degrees 0 and 1 yields improved image quality and diversity in early training, as shown below (taken at training step 50). Left: input data, middle: output from standard VAE, right: output from TopoVAE. 

<img src="assets/imgs_generated.png" alt="Images Generated" width="700"/>

This performance correlates with a faster decay of the Binary Cross-Entropy loss during the first 100 training steps, as shown below. 

<img src="assets/losses.png" alt="Loss Evolution" width="300"/>

Furthermore, as shown below, we observe an interesting behavior: when applying topological regularizers, the latent vectors seem to redistribute them selves in a more organized way according to their classes---compared to the standard VAE.

<img src="assets/latent_dist.png" alt="Latent Distribution" width="700"/>

We believe that the integration of topology into generative models through differentiable loss functions represents a promising new direction, with our initial results suggesting promising potential for future applications.

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
