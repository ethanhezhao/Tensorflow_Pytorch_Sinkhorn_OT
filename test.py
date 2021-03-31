import numpy as np

d1 = 120
d2 = 100
N = 40

a = np.random.rand(d1, N)
a = a / np.sum(a, 0)

b = np.random.rand(d2, N)
b = b / np.sum(b, 0)

M = np.random.rand(d1, d2)
M = M / np.median(M)

sh_lambda = 200.0


import matlab
import matlab.engine

a_mt = matlab.double(a.tolist())
b_mt = matlab.double(b.tolist())

K_mt = np.exp(- sh_lambda * M)
U_mt = np.multiply(K_mt, M)

K_mt = matlab.double(K_mt.tolist())
U_mt = matlab.double(U_mt.tolist())

eng = matlab.engine.start_matlab()

sinkhorn_divergences_mt = eng.sinkhornTransport(a_mt, b_mt, K_mt, U_mt, sh_lambda, [], [], [], [], 0)
sinkhorn_divergences_mt = sinkhorn_divergences_mt[0]

from auto_diff_sinkhorn import sinkhorn_tf, sinkhorn_torch
sinkhorn_divergences_tf = sinkhorn_tf(M, a, b, sh_lambda)

import torch

M_torch = torch.Tensor(M)
a_torch = torch.Tensor(a)
b_torch = torch.Tensor(b)
sinkhorn_divergences_torch = sinkhorn_torch(M_torch, a_torch, b_torch, sh_lambda)

print('Matlab')
print(sinkhorn_divergences_mt)

print('Tensorflow')
print(sinkhorn_divergences_tf)

print('Pytorch')
print(sinkhorn_divergences_torch)



assert np.allclose(sinkhorn_divergences_tf.numpy(), sinkhorn_divergences_mt, rtol=1e-04, atol=1e-04)

assert np.allclose(sinkhorn_divergences_torch.detach().cpu().numpy(), sinkhorn_divergences_mt, rtol=1e-04, atol=1e-04)

