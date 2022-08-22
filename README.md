## TF Grappler Optimized Operation Graph

A visualization tool to display TF-Grappler optimized op graph. Grappler is the
default graph optimization system in the TensorFlow runtime. Many different
graph optimization passes will be applied before the actual execution, such as
layout optimization, remapping optimizations, etc. (For the full list of
optimization passes, see
[here](https://www.tensorflow.org/guide/graph_optimization)). The traditional
way to display the op graph is via Tensorboard; however, Tensorboard only shows
the op graph before the grappler passes. To better understand how the grappler
changes the graph, this tool can be used to print out the op graphs before and
after any specified optimization pass.

### Usage
1. Install the dependencies.
```bash
pip install pydot && apt update && apt install -y graphviz
```
2. Clone the project and install it.
```bash
pip install .
```
3. Using the API to plot the op graphs before and after a specified optimizer.
```python
import tf_op_graph_vis
tf_op_graph_vis.grappler_optimized_graph(
    conv_bias_relu_model, (n, c, h, w), "remapper_conv_bias_relu.png",
    ['remapper'])
```
The above example generate the op graphs before and after the `remapping`
optimization. In the generated graphs, we do a simple graph identity check so
that the nodes only in the "before" graph will be colored to "green" and the
nodes only in the right graph are "red".

Note, at this point we only support three optimizers: `remapper`, `layout`, and
`arithmetic`. There are many sample codes in [examples](examples). For example,
this sample will generate the following graph.

```bash
python examples/remapper_conv_bias_relu.py
```
![Remapping pass](pics/remapper_conv_bias_relu.png)

