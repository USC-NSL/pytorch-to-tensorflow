import tensorflow as tf
import torch
import numpy as np

test_size = 2000
input_size = 20

X_test = np.random.randn(test_size, input_size).astype(np.float32)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dummy_input = torch.from_numpy(X_test[0].reshape(1, -1)).float().to(device)
dummy_input = X_test[0].reshape(1, -1)

def load_pb(path_to_pb):
  with tf.gfile.GFile(path_to_pb, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
    return graph

tf_graph = load_pb('./models/model_simple.pb')
sess = tf.Session(graph=tf_graph)

# Show tensor names in graph
for op in tf_graph.get_operations():
  print(op.values())

output_tensor = tf_graph.get_tensor_by_name('test_output:0')
input_tensor = tf_graph.get_tensor_by_name('test_input:0')

output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
print(output)