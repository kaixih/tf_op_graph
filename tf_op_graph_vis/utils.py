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

_name_suffix = "_opt"

# This is copied from enum DataType in TF.
class DataType(enum.Enum):
  DT_INVALID = 0
  DT_FLOAT = 1
  DT_DOUBLE = 2
  DT_INT32 = 3
  DT_UINT8 = 4
  DT_INT16 = 5
  DT_INT8 = 6
  DT_STRING = 7
  DT_COMPLEX64 = 8
  DT_INT64 = 9
  DT_BOOL = 10
  DT_QINT8 = 11
  DT_QUINT8 = 12
  DT_QINT32 = 13
  DT_BFLOAT16 = 14
  DT_QINT16 = 15
  DT_QUINT16 = 16
  DT_UINT16 = 17
  DT_COMPLEX128 = 18
  DT_HALF = 19
  DT_RESOURCE = 20
  DT_VARIANT = 21
  DT_UINT32 = 22
  DT_UINT64 = 23
  DT_FLOAT_REF = 101
  DT_DOUBLE_REF = 102
  DT_INT32_REF = 103
  DT_UINT8_REF = 104
  DT_INT16_REF = 105
  DT_INT8_REF = 106
  DT_STRING_REF = 107
  DT_COMPLEX64_REF = 108
  DT_INT64_REF = 109
  DT_BOOL_REF = 110
  DT_QINT8_REF = 111
  DT_QUINT8_REF = 112
  DT_QINT32_REF = 113
  DT_BFLOAT16_REF = 114
  DT_QINT16_REF = 115
  DT_QUINT16_REF = 116
  DT_UINT16_REF = 117
  DT_COMPLEX128_REF = 118
  DT_HALF_REF = 119
  DT_RESOURCE_REF = 120
  DT_VARIANT_REF = 121
  DT_UINT32_REF = 122
  DT_UINT64_REF = 123


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


def add_edge_helper(cluster, src, dst, need_suffix):
  """Adds edge of src->dst to cluster. """
  global _name_suffix
  if not cluster.get_edge(src, dst):
    src = src + (_name_suffix if need_suffix else "")
    dst = dst + (_name_suffix if need_suffix else "")
    cluster.add_edge(pydot.Edge(src, dst))


def get_custom_label(node):
  label = node.op

  # For Transpose node, display the layout conversion.
  if label == 'Transpose':
    patterns = ['NHWCToNCHW', 'NCHWToNHWC', 'NDHWCToNCDHW', 'NCDHWToNDHWC']
    for pattern in patterns:
      if (any(pattern.lower() in inp.lower() for inp in node.input)):
        label += "\n(" + pattern + ")"

  # For _Fused[MatMul|Conv2D] node, display the fused ops.
  if label == '_FusedMatMul' or label == '_FusedConv2D':
    fused_ops = node.attr['fused_ops'].list.s
    label += "\n(" + ','.join(op.decode('utf-8') for op in fused_ops) + ")"

  # For every node, display the data type.
  if 'T' in node.attr:
    dt_type = node.attr['T'].type
    label += "\nT=" + DataType(dt_type).name

  # For every node, display the device info if it is GPU.
  if 'gpu' in node.device.lower():
    label += '\ndevice=GPU'

  return label


def add_node_helper(cluster, node, need_suffix):
  """Adds node to cluster. """
  global _name_suffix
  node_name = node.name + (_name_suffix if need_suffix else "")
  label = get_custom_label(node)

  node = pydot.Node(node_name, label=label, style='filled', fillcolor='white')
  cluster.add_node(node)


def graphdef_to_pydot_cluster(graphdef, cluster_name, need_suffix):
  """Converts GraphDef to pydot cluster. """
  cluster = pydot.Cluster(cluster_name, label=cluster_name)

  # Add all nodes to the cluster.
  for node in graphdef.node:
    add_node_helper(cluster, node, need_suffix)

  # Add all edges to the cluster.
  for dst_node in graphdef.node:
    dst_node_name = dst_node.name
    for src_node_name in dst_node.input:
      add_edge_helper(cluster, src_node_name, dst_node_name, need_suffix)

  return cluster


def pydot_dot_to_pic(dot, to_file):
  """Converts pydot dot format to a file. """
  file_name, extension = os.path.splitext(to_file)
  extension = extension[1:] if extension else 'png'
  dot.write(file_name + '.' + extension, format=extension)
  print("[TF-OP-GRAPH] The GraphDef is plotted to %s." % to_file)


def compare_graphs(cluster0, cluster1, dot):
  global _name_suffix

  def color_nodes(c0, c1, color, need_suffix):
    for n0 in c0.get_nodes():
      need_strip = False
      if n0.get_name().endswith('\"'):
        need_strip = True

      node_name = n0.get_name().strip('\"')
      if need_suffix:
        node_name += _name_suffix
      else:
        node_name = node_name[:-len(_name_suffix)]
      if need_strip:
        node_name = '\"' + node_name + '\"'
      matched_nodes = c1.get_node(node_name)

      # We view the two nodes are identical if they have the same name (a unique
      # identifier) and the same label (which includes op/device info, data
      # type, etc.).
      node_label = n0.get_label()
      if (len(matched_nodes) == 0):
        n0.set_style('filled')
        n0.set_fillcolor('grey')
      elif (matched_nodes[0].get_label() != node_label):
        n0.set_style('filled')
        n0.set_fillcolor(color)

  color_nodes(cluster0, cluster1, 'green', True)
  color_nodes(cluster1, cluster0, 'red', False)


def graphdef_to_image(graphdefs, to_file):
  """Converts graphdef(s) to a file. """
  try:
    check_pydot()
  except ImportError as e:
    raise e

  if len(graphdefs) < 1 or len(graphdefs) > 2:
    raise ValueError("The lenghth of graphdefs must be contain 1 or 2.")

  dot = pydot.Dot()
  dot.set('rankdir', 'TB')
  dot.set('concentrate', True)
  dot.set('dpi', 96)
  dot.set_node_defaults(shape='record')

  cluster0_name = "" if len(graphdefs) == 1 else "Before"
  cluster0 = graphdef_to_pydot_cluster(graphdefs[0], cluster0_name, False)
  dot.add_subgraph(cluster0)

  if len(graphdefs) == 2:
    cluster1 = graphdef_to_pydot_cluster(graphdefs[1], "After", True)
    dot.add_subgraph(cluster1)
    compare_graphs(cluster0, cluster1, dot)

  pydot_dot_to_pic(dot, to_file)


def grappler_optimized_graph(model_fn, input_shape, to_file, options = []):
  def get_config(is_optimizer_off):
    """Returns a CongfigProto with remapper optimizer on/off."""
    ON = rewriter_config_pb2.RewriterConfig.ON
    OFF = rewriter_config_pb2.RewriterConfig.OFF
    toggle = lambda name: OFF if name in options and is_optimizer_off else ON

    rewrite_config = rewriter_config_pb2.RewriterConfig(
        remapping=toggle('remapper'),
        layout_optimizer=toggle('layout'),
        arithmetic_optimization=toggle('arithmetic'),
    )
    rewrite_config.min_graph_nodes = -1
    graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_config)
    config = config_pb2.ConfigProto(graph_options=graph_options)
    return config

  def get_graph(is_optimizer_off):
    with context.graph_mode():
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      metadata = config_pb2.RunMetadata()

      ops.reset_default_graph()
      x = variables.Variable(random_ops.truncated_normal(input_shape, seed=0))
      out = model_fn(x)

      config = get_config(is_optimizer_off)
      with session.Session(config=config) as sess:
        sess.run(variables.global_variables_initializer())
        _ = sess.run(out, options=run_options, run_metadata=metadata)
        graph = metadata.partition_graphs[0]
    return graph

  graph_before = get_graph(True)

  graphs = [graph_before]
  if len(options) != 0:
    graph_after = get_graph(False)
    graphs.append(graph_after)

  graphdef_to_image(graphs, to_file)

