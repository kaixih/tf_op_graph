import os
# The fusion relies on cuBlasLt.
os.environ['TF_USE_CUBLASLT'] = '1'

import tensorflow as tf
import tf_op_graph_vis

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  init_fn = tf.keras.initializers.GlorotUniform(seed=0)
  return tf.Variable(lambda: init_fn(shape))

m, n, k = (3, 3, 4)  # Matrix dimensions
precision = tf.float32

def matmul_bias_relu_model(x):
  w = _weight([k, n])
  b = _weight([n])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.linalg.matmul(x, w)
  z = tf.nn.bias_add(y, b)
  out = tf.nn.relu(z)
  return out

tf_op_graph_vis.grappler_optimized_graph(
    matmul_bias_relu_model, (m, k), "remapper_matmul_bias_relu.png",
    ['remapper'])

