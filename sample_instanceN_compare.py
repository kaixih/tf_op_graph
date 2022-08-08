import argparse
import nv_norms
import socket
import tensorflow as tf
import tensorflow_addons as tfa

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

def conv_instanceN_keras(x):
  w = _weight([2, 2, c, c])

  x = tf.cast(x, precision)
  w = tf.cast(w, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME')
  out = tfa.layers.InstanceNormalization(axis=-1)(y)
  return tf.identity(out)

def conv_layerN_keras(x):
  w = _weight([2, 2, c, c])

  x = tf.cast(x, precision)
  w = tf.cast(w, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME')
  out = tf.keras.layers.LayerNormalization(axis=(-1, -2, -3))(y)
  return tf.identity(out)

def conv_instanceN_nv(x):
  w = _weight([2, 2, c, c])

  # gamma and beta have to stay as float32, as required by instanceN.
  gamma = _weight([c])
  beta = _weight([c])

  x = tf.cast(x, precision)
  w = tf.cast(w, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME')
  out, _, _ = nv_norms.fused_instance_norm(y, gamma, beta)
  return tf.identity(out)

def conv_layerN_nv(x):
  # gamma and beta have to stay as float32, as required by layerN.
  gamma = _weight([features])
  beta = _weight([features])
  w = _weight([2, 2, c, c])

  x = tf.cast(x, precision)
  w = tf.cast(w, precision)

  y = tf.nn.conv2d(x, w, strides=(1, 1), padding='SAME')
  out, _, _ = nv_norms.fused_layer_norm(y, gamma, beta)
  return tf.identity(out)


print_op_graph(
    conv_instanceN_nv, (n, h, w, c),
    f"compare_instanceN_{precision.name}_{socket.gethostname()}_nv.png", [])
print_op_graph(
    conv_instanceN_keras, (n, h, w, c),
    f"compare_instanceN_{precision.name}_{socket.gethostname()}_keras.png", [])
print_op_graph(
    conv_layerN_nv, (n, h, w, c),
    f"compare_layerN_{precision.name}_{socket.gethostname()}_nv.png", [])
print_op_graph(
    conv_layerN_keras, (n, h, w, c),
    f"compare_layerN_{precision.name}_{socket.gethostname()}_keras.png", [])

