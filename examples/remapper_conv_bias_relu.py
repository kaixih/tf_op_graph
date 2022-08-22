import tensorflow as tf
import tf_op_graph_vis

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  init_fn = tf.keras.initializers.GlorotUniform(seed=0)
  return tf.Variable(lambda: init_fn(shape))

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

tf_op_graph_vis.grappler_optimized_graph(
    conv_bias_relu_model, (n, c, h, w), "remapper_conv_bias_relu.png",
    ['remapper'])

