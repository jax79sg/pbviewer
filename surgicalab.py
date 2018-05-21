import tensorflow as tf
import sys
from tensorflow.core.framework import graph_pb2
import copy


INPUT_GRAPH_DEF_FILE = "/home/dh/optimized_graph.pb"
OUTPUT_GRAPH_DEF_FILE = "/home/dh/converted_optimized_graph.pb"

# load our graph
def load_graph(filename):
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def



graph_def = load_graph(INPUT_GRAPH_DEF_FILE)

target_node_name = "phase_train"
phase_train = tf.constant(False, dtype=bool, shape=[], name="phase_train")
batch_size = tf.constant(1, dtype=tf.int32, shape=[1], name="batch_size")

# Create new graph, and rebuild it from original one
# replacing phase train node def with constant
new_graph_def = graph_pb2.GraphDef()
for node in graph_def.node:
    if node.name == "phase_train":
        new_graph_def.node.extend([phase_train.op.node_def])
    # elif node.name == "batch_size":
    #     new_graph_def.node.extend([batch_size.op.node_def])
    # elif node.name in "fifo_queue, batch_join, batch_size, label_batch, image_batch":
    #     pass
    else:
        new_graph_def.node.extend([copy.deepcopy(node)])

# save new graph
with tf.gfile.GFile(OUTPUT_GRAPH_DEF_FILE, "wb") as f:
    f.write(new_graph_def.SerializeToString())