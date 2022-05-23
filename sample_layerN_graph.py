import argparse
import nv_norms
import socket
import tensorflow as tf

from tf_op_graph_utils import print_op_graph

parser = argparse.ArgumentParser(description='Use --fp16 for fp16 inputs')
parser.add_argument('--fp16', action='store_true',
                    help="""Whether to enable fp16 input.""")
args, _ = parser.parse_known_args()

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  init_fn = tf.keras.initializers.GlorotUniform(seed=0)
  return tf.Variable(lambda: init_fn(shape))

n, h, w, c = (5, 3, 3, 4)
features = c * h * w
precision = tf.float16 if args.fp16 else tf.float32

def conv_layerN_relu_model(x):
  w = _weight([2, 2, c, c])
  b = _weight([c])
  # gamma and beta have to stay as float32, as required by layerN.
  gamma = _weight([features])
  beta = _weight([features])
  x = tf.cast(x, precision)
  w = tf.cast(w, precision)
  b = tf.cast(b, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format='NHWC')
  z, _, _ = nv_norms.fused_layer_norm(y, gamma, beta)
  x = tf.nn.relu(z)
  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME', data_format='NHWC')
  z, _, _ = nv_norms.fused_layer_norm(y, gamma, beta)
  out = tf.nn.relu(z)
  return tf.identity(out)

print_op_graph(
    conv_layerN_relu_model, (n, h, w, c),
    f"layout_pass_layerN_{precision.name}_{socket.gethostname()}.png",
    ['layout'])

