#PB graph stuff

##viewer.py
This script simply load a tensorflow pb graph file and write out in a format that can be displayed by tensorboard.

## surgicalab.py
This script loads a tensorflow pb graph and copy the entire graph. 
A simple if-else that checks for a ops name and allow you to make changes to that node. (E.g. Changing a placeholder into a
constant). If you want to remove notes, strongly suggest you use Tensorflow's transform graph script.

## keras_to_tensorflow.py
As the name suggests, convert a keras serialised H5 model into Tensorflow pb graph.

## freeze_graph.py
Part of tensorflow's utility for freezing graphs.