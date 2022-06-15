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
  wt = _weight([r, s, c, k])

  with tf.GradientTape() as g:
    g.watch(x)
    # Add a relu node to make sure the conv2d_backprop is not the last node.
    x0 = tf.nn.relu(x)
    y0 = tf.nn.conv2d(x0, w0, 1, 'SAME')
    y1 = tf.nn.conv2d(y0, w1, 1, 'SAME')
    y2 = tf.nn.conv2d(y0, w2, 1, 'SAME')
    y3 = tf.nn.conv2d(y0, w3, 1, 'SAME')

    add_out = tf.math.add(y1, y2)
    add_out = tf.math.add(add_out, y3)
    out = tf.nn.relu(add_out)
    loss = tf.reduce_sum(out)
  grads = g.gradient(loss, x)

  return grads

print_op_graph(multi_add_model, (n, h, w, c), "demo.png",
               ["arithmetic"])

