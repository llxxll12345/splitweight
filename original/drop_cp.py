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

import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = None

def drop_tensors_in_checkpoint_file(file_name, out_file, drop_layer, drop_percentage):
    #target = ['InceptionV3/Conv2d_4a_3x3/weights']
    target = drop_layer
    dropped = {}
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        weights = reader.get_tensor(key)
        indecies = np.transpose(np.nonzero(reader.get_tensor(key)))
        shape = reader.get_tensor(key).shape 
        #######
        if key in target:
            print("tensor_name: ", key, '===', reader.get_tensor(key).shape)
            dropped[key] = tf.Variable(tf.constant(weights))
            print (drop_percentage)
            if drop_percentage==1.0:
                dropped[key] = tf.Variable(tf.zeros(shape))
                print("dropping all")
            elif drop_percentage==0:
                continue
            else:
                flatten_weights = weights.flatten()
                number_of_zeros = int(drop_percentage * flatten_weights.size)
                indexes_of_zeros = xrange(number_of_zeros)
                flatten_weights[indexes_of_zeros] = 0.0
                weight_new = flatten_weights.reshape(shape)
                dropped[key] = tf.Variable(tf.constant(weight_new), dtype=tf.float32)
                continue
            dropped[key+"_idx"] = tf.Variable(tf.constant(indecies)) 
            dropped[key+"_shape"] = tf.Variable(tf.constant(shape)) 
        #######
        else:
            print("tensor_name: ", key, '===', reader.get_tensor(key).shape)
            dropped[key] = tf.Variable(tf.constant(weights))
            dropped[key+"_idx"] = tf.Variable(tf.constant(indecies)) 
            dropped[key+"_shape"] = tf.Variable(tf.constant(shape)) 

    with tf.Session() as sess:
	# Initialize new variables in a sparse form
	for var in tf.global_variables():
		if tf.is_variable_initialized(var).eval() == False:
			sess.run(tf.variables_initializer([var]))
	# Save model objects to serialized format
        final_saver = tf.train.Saver(dropped)
        print ("print pruned model to " + out_file)
        final_saver.save(sess, out_file)



def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names=False):
  """Prints tensors in a checkpoint file.
  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.
  If `tensor_name` is provided, prints the content of the tensor.
  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
    all_tensor_names: Boolean indicating whether to print all tensor names.
  """
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        print("tensor_name: ", key)
        if all_tensors:
          print(reader.get_tensor(key))
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      print(reader.get_tensor(tensor_name))
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
    v = (
        v_type(v_str)
        if v_type is not bool else flags.BooleanParser().parse(v_str))
  except ValueError as e:
    raise argparse.ArgumentTypeError(e.message)
  np.set_printoptions(**{k: v})


def main(unused_argv):
  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
          "[--tensor_name=tensor_to_print] "
          "[--all_tensors] "
          "[--all_tensor_names] "
          "[--printoptions]")
    sys.exit(1)
  else:
    #print_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.tensor_name,
    #                                 FLAGS.all_tensors, FLAGS.all_tensor_names)
    drop_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.out_file,
                                    FLAGS.drop_layer, FLAGS.drop_percentage)
                                     


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--file_name",
      type=str,
      default="",
      help="Checkpoint filename. "
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
      help="If True, print the names and values of all the tensors.")
  parser.add_argument(
      "--all_tensor_names",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="If True, print the names of all the tensors.")
  parser.add_argument(
      "--printoptions",
      nargs="*",
      type=parse_numpy_printoption,
      help="Argument for numpy.set_printoptions(), in the form 'k=v'.")
  parser.add_argument(
      "--out_file",
      nargs="?",
      default="dropped_ckpt",
      type=str,
      help="Output file name")
  parser.add_argument(
      "--drop_layer",
      nargs="?",
      default="",
      type=str,
      help="Target layer to drop name")
  parser.add_argument(
      "--drop_percentage",
      nargs="?",
      default=0,
      type=float,
      help="drop percentage name")

  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
