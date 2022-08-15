import enum
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

class DtType(enum.Enum):
  DT_FLOAT = 1
  DT_HALF = 19

def check_pydot():
  """Returns True if PyDot and Graphviz are available."""
  passed = False
  if pydot is not None:
    try:
      # Attempt to create an image of a blank graph
      # to check the pydot/graphviz installation.
      pydot.Dot.create(pydot.Dot())
      passed = True
    except (OSError, pydot.InvocationException):
      passed = False

  if not passed:
    message = (
        'You must install pydot and install graphviz (see instructions at '
        'https://graphviz.gitlab.io/download/) for plot_to_file option to '
        'work. For example: \n'
        '`pip install pydot && apt update && apt install -y graphviz`.')
    raise ImportError(message)

def add_edge_helper(dot, src, dst, name_suffix = ""):
  """Adds edge from src to dst. """
  src_name = src + name_suffix
  dst_name = dst + name_suffix
  if not dot.get_edge(src_name, dst_name):
    dot.add_edge(pydot.Edge(src_name, dst_name))

def add_node_helper(cluster, node, name_suffix = "", marked_node_names = {},
                    highlight_patterns = []):
  """Adds node to cluster. """
  label = node.op

  fillcolor = 'white'
  style = 'filled'

  # Display the layout conversion direction.
  if label == 'Transpose':
    for input_name in node.input:
      if 'nhwctonchw' in input_name.lower():
        label += "\n(NHWC to NCHW)"
      if 'nchwtonhwc' in input_name.lower():
        label += "\n(NCHW to NHWC)"
      if 'ndhwctoncdhw' in input_name.lower():
        label += "\n(NDHWC to NCDHW)"
      if 'ncdhwtondhwc' in input_name.lower():
        label += "\n(NCDHW to NDHWC)"

  # Display the fused ops.
  if label == '_FusedMatMul':
    fused_ops = node.attr['fused_ops'].list.s
    label += "\n("
    for op in fused_ops:
      label += op.decode("utf-8") + ","
    label += ")"

  if 'T' in node.attr:
    dt_type = node.attr['T'].type
    if dt_type == DtType.DT_FLOAT.value:
      label += '\ndtype=fp32'
    if dt_type == DtType.DT_HALF.value:
      label += '\ndtype=fp16'

  if 'gpu' in node.device.lower():
    label += '\ndevice=GPU'
  else:
    label += '\ndevice=CPU'

  graph_name = node.name + name_suffix
  if graph_name in marked_node_names:
    fillcolor = marked_node_names[graph_name]

  for pattern in highlight_patterns:
    if label.startswith(pattern):
      fillcolor = 'yellow'
      style = '"dashed,filled"'

  pynode = pydot.Node(graph_name, label=label, style=style, fillcolor=fillcolor)
  cluster.add_node(pynode)

def plot_graph_def(graph, dest_path):
  """Plots graph def with dot format. """
  try:
    check_pydot()
  except ImportError as e:
    raise e

  dot = pydot.Dot()
  dot.set('rankdir', 'TB')
  dot.set('concentrate', True)
  dot.set('dpi', 96)
  dot.set_node_defaults(shape='record')

  # Add all the nodes to the dot.
  for node in graph.node:
    add_node_helper(dot, node)

  # Create edges for these nodes.
  for dst_node in graph.node:
    dst_node_name = dst_node.name
    for src_node_name in dst_node.input:
      add_edge_helper(dot, src_node_name, dst_node_name)

  file_name, extension = os.path.splitext(dest_path)
  extension = extension[1:] if extension else 'png'

  dot.write(file_name + '.' + extension, format=extension)
  print("[TF-OP-GRAPH] The graph_def is plotted to %s." % dest_path)


def plot_ops_graph(graph, graph_opt, to_file, highlight_patterns):
  """Converts ops to dot format and save to a file. """
  try:
    check_pydot()
  except ImportError as e:
    raise e

  dot = pydot.Dot()
  dot.set('rankdir', 'TB')
  dot.set('concentrate', True)
  dot.set('dpi', 96)
  dot.set_node_defaults(shape='record')

  graph_suffix = '_orig'
  graph_opt_suffix = '_opt'
  # We mark the nodes that only appear in the original graph as "green" and the
  # nodes that only in the optimized graph as "red".
  marked_node_names = {}
  for node in graph.node:
    found = False
    for node_opt in graph_opt.node:
      if node.name == node_opt.name and node.op == node_opt.op:
        found = True
        break
    if not found:
      marked_node_names[node.name + graph_suffix] = 'green'

  for node_opt in graph_opt.node:
    found = False
    for node in graph.node:
      if node.name == node_opt.name and node.op == node_opt.op:
        found = True
        break
    if not found:
      marked_node_names[node_opt.name + graph_opt_suffix] = 'red'

  cluster_before = pydot.Cluster('Before', label='Before')
  cluster_after = pydot.Cluster('After', label='After')
  # Add all the nodes to the dot.
  for node in graph.node:
    add_node_helper(cluster_before, node, graph_suffix, marked_node_names,
                    highlight_patterns)
  dot.add_subgraph(cluster_before)

  for node in graph_opt.node:
    add_node_helper(cluster_after, node, graph_opt_suffix, marked_node_names,
                    highlight_patterns)
  dot.add_subgraph(cluster_after)

  # Create edges for these nodes.
  for dst_node in graph.node:
    dst_node_name = dst_node.name
    for src_node_name in dst_node.input:
      add_edge_helper(dot, src_node_name, dst_node_name, graph_suffix)

  for dst_node in graph_opt.node:
    dst_node_name = dst_node.name
    for src_node_name in dst_node.input:
      add_edge_helper(dot, src_node_name, dst_node_name, graph_opt_suffix)

  file_name, extension = os.path.splitext(to_file)
  if not extension:
    extension = 'png'
  else:
    extension = extension[1:]
  dot.write(file_name + '.' + extension, format=extension)
  print("[TF-OP-GRAPH] The op graph is plotted to %s." % to_file)

def _get_config(options):
  """Returns a CongfigProto with remapper optimizer on/off."""
  rewrite_config = rewriter_config_pb2.RewriterConfig(
      remapping=options['remapper'],
      layout_optimizer=options['layout'],
      arithmetic_optimization=options['arithmetic'])
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_config)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  return config

def print_op_graph(model_fn, input_shape, plot_file, optimizers,
                   highlight_patterns=[]):
  with context.graph_mode():
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    ops.reset_default_graph()
    x = variables.Variable(random_ops.truncated_normal(input_shape, seed=0))
    out = model_fn(x)

    options = {}
    opt_on = rewriter_config_pb2.RewriterConfig.ON
    opt_off = rewriter_config_pb2.RewriterConfig.OFF
    options['remapper'] = opt_off if 'remapper' in optimizers else opt_on
    options['layout'] = opt_off if 'layout' in optimizers else opt_on
    options['arithmetic'] = opt_off if 'arithmetic' in optimizers else opt_on

    # Compute reference value.
    config = _get_config(options)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      output_val_ref = sess.run(
          out, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    for key in options:
      options[key] = opt_on
    config = _get_config(options)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(
          out, options=run_options, run_metadata=metadata)
      graph_opt = metadata.partition_graphs[0]

    plot_ops_graph(graph, graph_opt, plot_file, highlight_patterns)
