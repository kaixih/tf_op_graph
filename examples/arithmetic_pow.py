import os
# This arithmetic optimization on pow op only happens on CPUs.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tf_op_graph_vis

m, n, k = (3, 3, 4)  # Matrix dimensions
precision = tf.float32

def pow_model(x):
  x = tf.cast(x, precision)

  out = tf.math.pow(x, 3.0)
  return out

tf_op_graph_vis.grappler_optimized_graph(
    pow_model, (m, k), "arithmetic_pow.png", ['arithmetic'])

