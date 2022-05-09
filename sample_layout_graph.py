import tensorflow as tf

from tf_op_graph_utils import print_op_graph

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

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format='NHWC')
  z = tf.nn.bias_add(y, b, data_format='N..C')
  x = tf.nn.relu(z)
  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format='NHWC')
  z = tf.nn.bias_add(y, b, data_format='N..C')
  out = tf.nn.relu(z)
  return tf.identity(out)

print_op_graph(conv_bias_relu_model, (n, h, w, c), "layout_opt.png",
               remapping_on=False, layout_on=True,
               highlight_patterns=['Transpose'])
print_op_graph(conv_bias_relu_model, (n, h, w, c), "layout_unopt.png",
               remapping_on=False, layout_on=False)

