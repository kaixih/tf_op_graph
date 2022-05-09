import os
os.environ['TF_USE_CUBLASLT'] = '1'

import tensorflow as tf

from tf_op_graph_utils import print_op_graph

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  init_fn = tf.keras.initializers.GlorotUniform(seed=0)
  return tf.Variable(lambda: init_fn(shape))

m, n, k = (3, 3, 4)  # Matrix dimensions
precision = tf.float32

def matmul_bias_gelu_model(x):
  w = _weight([k, n])
  b = _weight([n])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.linalg.matmul(x, w)
  z = tf.nn.bias_add(y, b)
  out = tf.nn.gelu(z, approximate=True)
  return out

print_op_graph(matmul_bias_gelu_model, (m, k), "gelu_fused.png",
               remapping_on=True, highlight_patterns=['_Fused'])
print_op_graph(matmul_bias_gelu_model, (m, k), "gelu_unfused.png",
               remapping_on=False)

n, h, w, c = (5, 3, 3, 4)
precision = tf.float32

def conv_bias_relu_model(x):
  w = _weight([2, 2, c, c])
  b = _weight([c])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format='NCHW')
  z = tf.nn.bias_add(y, b, data_format='NC..')
  out = tf.nn.relu(z)
  return out

print_op_graph(conv_bias_relu_model, (n, c, h, w), "conv_fused.png",
               remapping_on=True, highlight_patterns=['_Fused'])
print_op_graph(conv_bias_relu_model, (n, c, h, w), "conv_unfused.png",
               remapping_on=False)

