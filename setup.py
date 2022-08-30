from setuptools import setup, find_packages

__version__ = '0.0.1'
setup(
    name='tf_op_graph_vis',
    version=__version__,
    description=('tf_op_graph_vis is for visualizing TF graph_def files.'),
    author='Kaixi Hou',
    packages=find_packages(),
    entry_points = {
        'console_scripts':
            ['graphdef2pydot=tf_op_graph_vis.tf_graphdef_to_pydot:main'],
    },
    license='Apache 2.0',
)
