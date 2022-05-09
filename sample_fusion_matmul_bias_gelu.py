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

def gelu_model(x):
  w = _weight([k, n])
  b = _weight([n])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.linalg.matmul(x, w)
  z = tf.nn.bias_add(y, b)
  out = tf.nn.gelu(z, approximate=True)
  return out

print_op_graph(gelu_model, (m, k), True, "gelu_fused.png")
print_op_graph(gelu_model, (m, k), False, "gelu_unfused.png")

