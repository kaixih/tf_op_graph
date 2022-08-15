import argparse
import os
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tf_op_graph_utils import plot_graph_def

def dir_path(m):
  if os.path.isfile(m) and m.lower().endswith(('.pb.txt', '.pbtxt')):
    return m
  raise NotADirectoryError(m)

parser = argparse.ArgumentParser(description="Visualize GraphDef file.")
parser.add_argument('--path', help="""GraphDef file path.""")
parser.add_argument('--output', help="""PyDot output file path.""")
args, _ = parser.parse_known_args()

gd_path = ""
if args.path:
  gd_path = args.path
if (not os.path.isfile(gd_path) or
    not gd_path.lower().endswith(('.pb.txt', '.pbtxt'))):
  raise ValueError('Invalid *.pb.txt file path: %s' % (gd_path))

output_path = "default_graph_def.png"
if args.output:
  output_path = args.output

# Read the pbtxt file into a Graph protobuf
with open(args.path, "r") as f:
  graph = text_format.Parse(f.read(), graph_pb2.GraphDef())

plot_graph_def(graph, output_path)

