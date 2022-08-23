import tensorflow as tf
import tf_op_graph_vis

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  init_fn = tf.keras.initializers.GlorotUniform(seed=0)
  return tf.Variable(lambda: init_fn(shape))

M, K = (8, 4)
precision = tf.float32

def condition_model(x):
  w0 = _weight([M, K])
  w1 = _weight([M, K])
  a = _weight([1])
  b = _weight([1])

  w = tf.cond(a > b, lambda: w0, lambda: w1)
  out = tf.multiply(w, x)

  return out

tf_op_graph_vis.grappler_optimized_graph(
    condition_model, (M, K), "graph_control_edges.png")
