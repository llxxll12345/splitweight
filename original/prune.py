# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


import tensorflow as tf
import numpy as np
import scipy.sparse
import scipy.io

import papl


from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = None

def _complex_concat(a, b):
    tmp = []
    for i in a:
        for j in b:
            tmp.append(i+j)
    return tmp


def gen_sparse_dict(dense_w):
    sparse_w = dense_w
    target_layer = [ "conv1/weights", "conv2/weights", "local3/weights", "local4/weights"]
    for target in target_layer:
        target_arr = np.transpose(dense_w.get_tensor(target))
        if 'local' in target:
            sparse_arr = papl.prune_tf_sparse_fc(target_arr, name=target, thresh1=0.005,thresh2=120, l=16)
        if 'conv' in target:
            sparse_arr = papl.prune_tf_sparse_conv(target_arr, name=target, thresh1=0.005,thresh2=120, l=16)
        sparse_w[target+"_idx"]=tf.Variable(tf.constant(sparse_arr[0],dtype=tf.int32),
            name=target+"_idx")
        sparse_w[target]=tf.Variable(tf.constant(sparse_arr[1],dtype=tf.float32),
            name=target)
        sparse_w[target+"_shape"]=tf.Variable(tf.constant(sparse_arr[2],dtype=tf.int32),
            name=target+"_shape")
    return sparse_w


def prune_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
  """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
  """
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    target_layer = [ "conv1/weights", "conv2/weights", "local3/weights", "local4/weights"]
    target_dat = [ "conv1.dat", "conv2.dat", "local3.dat", "local4.dat"]
    target_p_dat = _complex_concat(target_layer, ["_p.dat"])
    target_tp_dat = _complex_concat(target_layer, ["_tp.dat"])
    sparse_w ={
        "conv1/weights": tf.Variable(tf.truncated_normal([5,5,3,64],stddev=0.1), name="conv1/weights"),
        "conv2/weights": tf.Variable(tf.truncated_normal([5,5,64,64],stddev=0.1), name="conv2/weights"),
        "local3/weights": tf.Variable(tf.truncated_normal([6*6*64,384],stddev=0.1), name="local3/weights"),
        "local4/weights": tf.Variable(tf.truncated_normal([384,192],stddev=0.1), name="local4/weights"),
    }
    if all_tensors:
      print("ALL TENSORS")
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in target_layer:#sorted(var_to_shape_map):
        print("tensor_name: ", key)
        print(reader.get_tensor(key).shape)
        if 'local' in key:
            # papl.prune_dense_fc(reader.get_tensor(key), name=key, thresh1=0.01,thresh2=63, l=8)
            sparse_arr = papl.prune_tf_sparse_fc(reader.get_tensor(key), name=key, thresh1=0.001,thresh2=120, l=16)
        if 'conv' in key:
            # papl.prune_dense_conv(reader.get_tensor(key) , name=key, thresh1=0.01,thresh2=63, l=8)
            sparse_arr = papl.prune_tf_sparse_conv(reader.get_tensor(key) , name=key, thresh1=0.001,thresh2=120, l=16)
#-------------------
        sparse_w[key+"_idx"]=tf.Variable(tf.constant(sparse_arr[0],dtype=tf.int32),
            name=key+"_idx")
        sparse_w[key]=tf.Variable(tf.constant(sparse_arr[1],dtype=tf.float32),
            name=key)
        sparse_w[key+"_shape"]=tf.Variable(tf.constant(sparse_arr[2],dtype=tf.int32),
            name=key+"_shape")
        # sparse_w = gen_sparse_dict(reader)

      with tf.Session() as sess:
          # Initialize new variables in a sparse form
          for var in tf.global_variables():
             if tf.is_variable_initialized(var).eval() == False:
                sess.run(tf.variables_initializer([var]))

          # Save model objects to serialized format
          final_saver = tf.train.Saver(sparse_w)
          # Save model objects to readable format
          papl.print_weight_vars(sparse_w, target_layer,
                                 target_dat, show_zero=False)
          print ("print pruned model to ./model_ckpt_sparse_cifar")
          final_saver.save(sess, "./model_ckpt_sparse_cifar0.001_120")
#-------------------


    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("NOT ALL TENSORS")
      print("tensor_name: ", tensor_name)
      print(reader.get_tensor(tensor_name))
      W= reader.get_tensor(tensor_name)
    #   print (len(W[0,:]))
    #   print (np.sum(W != 0))
    #   sparse_matrix = scipy.sparse.csr_matrix(W)
    #   print(sparse_matrix.indices)
      #np.savetxt(fd,sparse_matrix.indices,fmt="%i")
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in file_name for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))



def parse_numpy_printoption(kv_str):
  """Sets a single numpy printoption from a string of the form 'x=y'.

  See documentation on numpy.set_printoptions() for details about what values
  x and y can take. x can be any option listed there other than 'formatter'.

  Args:
    kv_str: A string of the form 'x=y', such as 'threshold=100000'

  Raises:
    argparse.ArgumentTypeError: If the string couldn't be used to set any
        nump printoption.
  """
  k_v_str = kv_str.split("=", 1)
  if len(k_v_str) != 2 or not k_v_str[0]:
    raise argparse.ArgumentTypeError("'%s' is not in the form k=v." % kv_str)
  k, v_str = k_v_str
  printoptions = np.get_printoptions()
  if k not in printoptions:
    raise argparse.ArgumentTypeError("'%s' is not a valid printoption." % k)
  v_type = type(printoptions[k])
  if v_type is type(None):
    raise argparse.ArgumentTypeError(
        "Setting '%s' from the command line is not supported." % k)
  try:
    v = (v_type(v_str) if v_type is not bool
         else flags.BooleanParser().parse(v_str))
  except ValueError as e:
    raise argparse.ArgumentTypeError(e.message)
  np.set_printoptions(**{k: v})


def main(unused_argv):

  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
          "[--tensor_name=tensor_to_print]")
    sys.exit(1)
  else:
    prune_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.tensor_name,
                                     FLAGS.all_tensors)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--file_name", type=str, default="", help="Checkpoint filename. "
                    "Note, if using Checkpoint V2 format, file_name is the "
                    "shared prefix between all files in the checkpoint.")
  parser.add_argument(
      "--tensor_name",
      type=str,
      default="",
      help="Name of the tensor to inspect")
  parser.add_argument(
      "--all_tensors",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="If True, print the values of all the tensors.")
  parser.add_argument(
      "--printoptions",
      nargs="*",
      type=parse_numpy_printoption,
      help="Argument for numpy.set_printoptions(), in the form 'k=v'.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

