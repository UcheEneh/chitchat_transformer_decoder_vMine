import os
import re
import sys
import json
import math
import time
import unicodedata
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tqdm import tqdm
from functools import partial

# Single asterisk as used in function declaration allows variable number of arguments to be passed from calling
# environment. Inside the function it behaves as a tuple.


def encode_dataset(*splits, encoder):
    """

    for RocStories example:

    *splits: (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

        where trX1: combination of input sentences from spring2016 test_val train split
              trX2: fifth sentence 1 from spring2016 test_val train split
              trX3: fifth sentence 2 from spring2016 test_val train split
              trY:  correct label for the fifth sentence

              vaX1: combination of input sentences from spring2016 test_val validation split
              vaX2: fifth sentence 1 from spring2016 test_val validation split
              vaX3: fifth sentence 2 from spring2016 test_val validation split
              vaY:  correct label for the fifth sentence

              teX1: combination of input sentences from spring2016 test_test test
              teX2: fifth sentence 1 from spring2016 test_test test
              teX3: fifth sentence 2 from spring2016 test_test test
    """
    encoded_splits = []
    for split in splits[0]:     # (trX1, trX2, trX3, trY)
        fields = []
        for field in split:     # list of all joined input sentences
            if isinstance(field[0], str):   # if the first content is a string
                field = encoder.encode(field)   # basically, input: list of all joined input sentences, returns: list of input sentences in bit-pairs encoded format
            fields.append(field)
        encoded_splits.append(fields)
    return encoded_splits       # basically returns the encodings of all the inputs using bpe

def stsb_label_encoding(labels, nclass=6):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype(np.float32)
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y

def shape_list(x):  # x: embedded input with addition of positional encoding
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()    # just returns the shape in a list.     # diff between get_shape and tf.shape: https://stackoverflow.com/questions/37096225/how-to-understand-static-shape-and-dynamic-shape-in-tensorflow
    ts = tf.shape(x)        # basically this is an op to get the dynamic shape of a tensor
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]      # not sure how but I think: what is returned is ps[i, :, :, ...] depending on the length of the list, ps
                                                                            # Note: this isn't returning actual values, just the shape of the tensor. ts[i] becomes useful if the value in the list is None, as it dynamically calls for the value during computation.
def np_softmax(x, t=1):
    x = x/t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex/np.sum(ex, axis=-1, keepdims=True)

def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

def _identity_init(shape, dtype, partition_info, scale):
    n = shape[-1]
    w = np.eye(n)*scale
    if len([s for s in shape if s != 1]) == 2:
        w = w.reshape(shape)
    return w.astype(np.float32)

def identity_init(scale=1.0):
    return partial(_identity_init, scale=scale)

def _np_init(shape, dtype, partition_info, w):
    return w

def np_init(w):
    return partial(_np_init, w=w)

class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs)+'\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs)+'\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()

def find_trainable_variables(key):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))

def flatten(outer):
    return [el for inner in outer for el in inner]

def remove_none(l):
    return [e for e in l if e is not None]

def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n//n_batch)*n_batch
    n = min(n, max_batches*n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n//n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i+n_batch]
        else:
            yield (d[i:i+n_batch] for d in datas)
        n_batches += 1

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x

def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":   # returns the original device (gpu or cpu) if it's a Variable
            return ps_dev
        else:                           # else returns a (new) gpu with number from gpu value
            return "/gpu:%d" % gpu
    return _assign

def average_grads(tower_grads):     # tower_grads:  contains the four lists of gradients from each gpu     [list(zip(grads, trainable_vars)), ...]
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:     # if only one gpu used, no need for finding the average
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:     # if no gradient has been calculated yet
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):     # if the tensors in the grads from gpu is an indexed slice (*Note: for loop, so this is done for each gpu)
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)     # the average of the gradients in returned with the corresponding variables
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
