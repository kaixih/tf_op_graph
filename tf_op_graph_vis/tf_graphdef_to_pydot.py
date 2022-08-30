import argparse
import os
import sys
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2

from .utils import graphdef_to_image

def main():
  parser = argparse.ArgumentParser(description="Visualize GraphDef files.")
  parser.add_argument('-g0', '--graph0',
                      help="""The first GraphDef file path.""")
  parser.add_argument('-g1', '--graph1',
                      help="""The second GraphDef file path.""")
  parser.add_argument('-o', '--output', help="""PyDot output file path.""")

  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

  args, _ = parser.parse_known_args()

  gd0_path = args.graph0 if args.graph0 else ""
  gd1_path = args.graph1 if args.graph1 else ""

  def verify_path(m):
    if m == "" or os.path.isfile(m) and m.lower().endswith(('.pb.txt',
                                                            '.pbtxt')):
      return True
    return False

  if not verify_path(gd0_path):
    raise ValueError('Invalid *.pb.txt file path: %s' % (gd0_path))
  if not verify_path(gd1_path):
    raise ValueError('Invalid *.pb.txt file path: %s' % (gd1_path))
  if gd0_path == "" and gd1_path == "":
    raise ValueError('At least, one of graph0 and graph1 needs to be '
                     'specified.')
  if gd0_path == "":
    gd0_path, gd1_path = gd1_path, gd0_path

  to_path = args.output if args.output else "default_pydot.png"

  graphs = []
  with open(gd0_path, "r") as f:
    graph0 = text_format.Parse(f.read(), graph_pb2.GraphDef())
  graphs.append(graph0)

  if gd1_path != "":
    with open(gd1_path, "r") as f:
      graph1 = text_format.Parse(f.read(), graph_pb2.GraphDef())
    graphs.append(graph1)

  graphdef_to_image(graphs, to_path)

