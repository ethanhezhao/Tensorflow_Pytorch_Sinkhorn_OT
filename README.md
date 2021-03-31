# AutoDiffSinkhorn

Tensorflow (1.0 or 2.0) and Pytorch implementations of the Sinkhorn algorithm [1] for computing the optimal transport (OT) distance between two discrete distributions.

## Overview

The implementations are the adaptions from the [Matlab implemention](https://marcocuturi.net/SI.html) by Cuturi to Tensorflow and Pytorch, which are able to leverage their auto-diff power and ability to run on GPUs. The implementations compute the OT distances between N pairs of discrete distributions (i.e., probability vectors) in parallel. It corresponds to the "N times 1-vs-1 mode" in Cuturi's implementation.

## Input

- ```a```: A D_1 by N matrix, each column is a D_1 dimensional (normalised) probability vector.
- ```b```: A D_2 by N matrix, each column is a D_2 dimensional (normalised) probability vector.
- ```M```: A D_1 by D_2 matrix, the cost function, positive, diagonal shall be zero.
- ```lambda_sh, numItermax, stopThr```: The parameters of the algorithm, the same as Cuturi's implementation.

```a, b, M``` are tensors of Tensorflow or Pytorch, thus, backpropagation is applicable.

## Output

The algorithm outputs a N dimentional vector, the n<sup>th</sup> element is the (apprioximated) OT distance between ```a[:,n]``` and ```b[:,n]```.

## Test

In the file of ```test.py```, the comparison between the outputs from Cuturi's Matlab implementation and my implementations is provided. It can be seen that the outputs are very close. To run the comparison, first, download Cuturi's [Matlab implemention](https://marcocuturi.net/SI.html) and then put ```sinkhornTransport.m``` under the same folder. It also requires Matlab installation and [Matlab Engine API for Python](https://au.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). 

## Misc

- Other related implementations, e.g.:
  - [SinkhornAutoDiff](https://github.com/gpeyre/SinkhornAutoDiff)
  - [PyTorchOT](https://github.com/rythei/PyTorchOT)
  - [Sinkhorn-solver](https://gist.github.com/wohlert/8589045ab544082560cc5f8915cc90bd)
  - [TF-OT-Sinkhorn](https://github.com/MarkMoHR/TF-OT-Sinkhorn)
- To my knowledge, the above ones may not as general as Cuturi's Matlab implementation. For example,  ```a``` and ```b``` are usually assumed to be uniformly distributed, and ```M``` is assumed to be the Euclidean distance. That's why I wanted to re-implement the Matlab code.
- The code was originally used in the paper [2]. If you find the code helpful, please consider citing the paper.
- The code comes with no support.

[1] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport, NIPS 2013

[2] He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine: Neural Topic Model via Optimal Transport, ICLR 2021
