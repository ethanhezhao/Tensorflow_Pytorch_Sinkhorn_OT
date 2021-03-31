import tensorflow as tf
import numpy as np


def sinkhorn_tf(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2):


    u = tf.ones(shape=(tf.shape(a)[0], tf.shape(a)[1]), dtype=tf.float64) / tf.cast(tf.shape(a)[0], tf.float64)
    v = tf.zeros_like(b, dtype=tf.float64)
    K = tf.exp(-M * lambda_sh)

    cpt = tf.constant(0, dtype=tf.float64)
    err = tf.constant(1.0, dtype=tf.float64)

    c = lambda cpt, u, v, err: tf.logical_and(cpt < numItermax, err > stopThr)

    def v_update(u_, v_):
        v_ = tf.divide(b, tf.matmul(tf.transpose(K), u_))  
        u_ = tf.divide(a, tf.matmul(K, v_)) 
        return u_, v_

    def no_v_update(u_, v_):
        return u_, v_

    def err_f1(K_, u_, v_, b_):
        bb = tf.multiply(v_, tf.matmul(tf.transpose(K_), u_))
        err_ = tf.norm(tf.reduce_sum(tf.abs(bb - b_), axis=0), ord=np.inf)
        return err_

    def err_f2(err_):
        return err_

    def loop_func(cpt_, u_, v_, err_):
        u_ = tf.divide(a, tf.matmul(K, tf.divide(b, tf.transpose(tf.matmul(tf.transpose(u_), K)))))
        cpt_ = tf.add(cpt_, 1)
        u_, v_ = tf.cond(tf.logical_or(tf.equal(cpt_ % 20, 1), tf.equal(cpt, numItermax)), lambda: v_update(u_, v_), lambda: no_v_update(u_, v_))
        err_ = tf.cond(tf.logical_or(tf.equal(cpt_ % 20, 1), tf.equal(cpt, numItermax)), lambda: err_f1(K, u_, v_, b), lambda: err_f2(err_))
        return cpt_, u_, v_, err_

    cpt, u, v, err = tf.while_loop(c, loop_func, loop_vars=[cpt, u, v, err])

    sinkhorn_divergences = tf.reduce_sum(tf.multiply(u, tf.matmul(tf.multiply(K, M), v)), axis=0)
    return sinkhorn_divergences



import torch

def sinkhorn_torch(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2, cuda=False):    

    if cuda:
        u = (torch.ones_like(a) / a.size()[0]).double().cuda() 
        v = (torch.ones_like(b)).double().cuda()
    else:
        u = (torch.ones_like(a) / a.size()[0])
        v = (torch.ones_like(b))

    K = torch.exp(-M * lambda_sh) 
    err = 1
    cpt = 0
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t()))) 
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))  
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)
    return sinkhorn_divergences