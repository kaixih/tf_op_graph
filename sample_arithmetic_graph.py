import os
# This arithmetic optimization on pow op only happens on CPUs.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

from tf_op_graph_utils import print_op_graph

m, n, k = (3, 3, 4)  # Matrix dimensions
precision = tf.float32

def pow_model(x):
  x = tf.cast(x, precision)

  out = tf.math.pow(x, 3.0)
  return out

print_op_graph(pow_model, (m, k), "pow_arithmetic_pass.png",
               ['arithmetic'])

