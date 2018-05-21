import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    # model_filename ='/media/dh/DATA4TB/datasets/OPENSOURCE/Openface/models/CASIA-WebFace-Transformed/rebuilt20180408-102900.pb'
    model_filename = '/home/dh/snpe-sdk/examples/android/FRnPRonPhones/app/standby/CASIA-WebFace/20180408-102900.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='/home/dh/jax/pbviewer/logs'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
print("Run \'tensorboard --logdir=./logs\' to view the graph on browser")