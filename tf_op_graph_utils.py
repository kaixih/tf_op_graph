import os

try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.eager import context

def check_pydot():
  """Returns True if PyDot and Graphviz are available."""
  if pydot is None:
    return False
  try:
    # Attempt to create an image of a blank graph
    # to check the pydot/graphviz installation.
    pydot.Dot.create(pydot.Dot())
    return True
  except (OSError, pydot.InvocationException):
    return False

def add_edge(dot, src, dst):
  """Adds edge from src to dst. """
  if not dot.get_edge(src, dst):
    dot.add_edge(pydot.Edge(src, dst))

def plot_ops_graph(graph, to_file, highlight_patterns):
  """Converts ops to dot format and save to a file. """
  if not check_pydot():
    message = (
        'You must install pydot and install graphviz (see instructions at '
        'https://graphviz.gitlab.io/download/) for plot_to_file option to '
        'work. For example: \n'
        '`pip install pydot && apt update && apt install -y graphviz`.')
    raise ImportError(message)

  dot = pydot.Dot()
  dot.set('rankdir', 'TB')
  dot.set('concentrate', True)
  dot.set('dpi', 96)
  dot.set_node_defaults(shape='record')

  # Add all the nodes to the dot.
  for node in graph.node:
    def format_shape(shape):
      return str(shape).replace(str(None), 'None')

    label = node.op

    fillcolor = 'white'
    for pattern in highlight_patterns:
      if label.startswith(pattern):
        fillcolor = 'red'

    node = pydot.Node(node.name, label=label, style='filled',
                      fillcolor=fillcolor)
    dot.add_node(node)

  # Create edges for these nodes.
  for dst_node in graph.node:
    dst_node_name = dst_node.name
    for src_node_name in dst_node.input:
      add_edge(dot, src_node_name, dst_node_name)

  file_name, extension = os.path.splitext(to_file)
  if not extension:
    extension = 'png'
  else:
    extension = extension[1:]
  dot.write(file_name + '.' + extension, format=extension)
  print("[TF-OP-GRAPH] The op graph is plotted to %s." % to_file)

def _get_config(remapping_on=False, layout_on=False):
  """Returns a CongfigProto with remapper optimizer on/off."""
  rewrite_config = rewriter_config_pb2.RewriterConfig(
      remapping=rewriter_config_pb2.RewriterConfig
      .ON if remapping_on else rewriter_config_pb2.RewriterConfig.OFF,
      layout_optimizer=rewriter_config_pb2.RewriterConfig
      .ON if layout_on else rewriter_config_pb2.RewriterConfig.OFF,
      )
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_config)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  return config

def print_op_graph(model_fn, input_shape, plot_file, remapping_on=True,
                   layout_on=True, highlight_patterns=[]):
  with context.graph_mode():
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    ops.reset_default_graph()
    x = variables.Variable(random_ops.truncated_normal(input_shape, seed=0))
    out = model_fn(x)

    # Compute reference value.
    config = _get_config(remapping_on=remapping_on, layout_on=layout_on)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      output_val_ref = sess.run(
          out, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    plot_ops_graph(graph, plot_file, highlight_patterns)
