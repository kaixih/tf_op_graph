import tensorflow as tf

from tf_op_graph_utils import print_op_graph

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  init_fn = tf.keras.initializers.GlorotUniform(seed=0)
  return tf.Variable(lambda: init_fn(shape))

n, h, w, c = (2, 10, 10, 4)
k, r, s = (4, 3, 3)

def multi_add_model(x):
  w0 = _weight([r, s, c, k])
  w1 = _weight([r, s, c, k])
  w2 = _weight([r, s, c, k])
  w3 = _weight([r, s, c, k])
  y0 = tf.nn.conv2d(x, w0, 1, 'SAME')
  y1 = tf.nn.conv2d(x, w1, 1, 'SAME')
  y2 = tf.nn.conv2d(x, w2, 1, 'SAME')
  y3 = tf.nn.conv2d(x, w3, 1, 'SAME')

  z0 = tf.math.add(y0, y1)
  z1 = tf.math.add(y2, y3)
  add_out = tf.math.add(z0, z1)

  wt = _weight([r, s, c, k])
  out = tf.nn.conv2d(add_out, wt, 1, 'SAME')
  out = tf.nn.relu(out)
  return tf.identity(out)

print_op_graph(multi_add_model, (n, h, w, c), "multi_add_to_one_add_n.png",
               ["arithmetic"])

