import argparse
import tensorflow as tf

from tf_op_graph_utils import print_op_graph

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  init_fn = tf.keras.initializers.GlorotUniform(seed=0)
  return tf.Variable(lambda: init_fn(shape))

parser = argparse.ArgumentParser(description="Test conv fusion.")
parser.add_argument('--fp16', action='store_true', help='Whether to use fp16.')
args, _ = parser.parse_known_args()

n, h, w, c = (5, 3, 3, 8)
precision = tf.float16 if args.fp16 else tf.float32

x_shape = (n, c, h, w)
x_format = 'NCHW'
b_format = 'NC..'
if precision == tf.float16:
  x_shape = (n, h, w, c)
  x_format = 'NHWC'
  b_format = 'N..C'

def conv_bias_relu_model(x):
  w = _weight([2, 2, c, c])
  b = _weight([c])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format=x_format)
  z = tf.nn.bias_add(y, b, data_format=b_format)
  out = tf.nn.relu(z)
  return out

def conv_bias_relu6_model(x):
  w = _weight([2, 2, c, c])
  b = _weight([c])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format=x_format)
  z = tf.nn.bias_add(y, b, data_format=b_format)
  out = tf.nn.relu6(z)
  return out

def conv_bias_elu_model(x):
  w = _weight([2, 2, c, c])
  b = _weight([c])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format=x_format)
  z = tf.nn.bias_add(y, b, data_format=b_format)
  out = tf.nn.elu(z)
  return out

def conv_bias_leakyrelu_model(x):
  w = _weight([2, 2, c, c])
  b = _weight([c])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format=x_format)
  z = tf.nn.bias_add(y, b, data_format=b_format)
  out = tf.nn.leaky_relu(z, alpha=0.3)
  return out

model_fns = [conv_bias_relu_model, conv_bias_relu6_model, conv_bias_elu_model,
             conv_bias_leakyrelu_model]
for model_fn in model_fns:
  filename = model_fn.__name__ + ".png"
  print_op_graph(model_fn, x_shape, filename, ['remapper'])
  print(">>> output written to:", filename)

