import numpy as np
import tensorflow as tf
import csv
import sys
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, Input
import os
from os import listdir
from os.path import isfile, join
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

weights_map = {}
var_map = {}

def restore(file_name, k, v, use_double=False):
    dtypee = tf.float64 if use_double else tf.float32
    weight = tf.get_variable(k, initializer = tf.zeros_initializer, shape = v, dtype=dtypee)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({k: weight})
        saver.restore(sess, file_name)
        weight_val = weight.eval()
        return weight_val
        

def inspect(file_name):
    res = []
    try:
        myreader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var2shape = myreader.get_variable_to_shape_map()
        for k in sorted(var2shape):
            print("name: ", k, "shape: ", var2shape[k])
            res.append(k)
        restore(file_name)
        return res
    except Exception as e:
        print(str(e))

def is_number(str):
    for i in str:
        if i < '0' or i > '9':
            return False
    return True

def format_csv():
    with open('csv_output/format.csv', 'w+') as f:
        writer = csv.DictWriter(f, ['name','shape'])
        writer.writeheader()
        for k, v in sorted(weights_map.items()):
            writer.writerow({'name': k, 'shape': v})

def data_csv():
    for k, v in weights_map.items():
        file_name = '-'.join(k.split('/'))
        file_source = 'output/' + file_name + '/' + file_name + '.ckpt'
        print(file_source)
        weight = restore(file_source, k, v)
        print("size: ", v, weight[0])
        var_map.update({k: weight})
        csv_name = 'csv_output/' + file_name + '.csv' 
        weight_na = np.asarray(weight)
        weight_na.tofile(csv_name, sep=',')


def main(argv):
    # do_inspection_first
    with open("layer_names.txt", 'r') as f:
        for line in f:
            line = line[:-1]
            layer_name = line.split(',')[0]
            layer_size = line.split(',')[1:]
            l_name = '-'.join(layer_name.split('/'))
            if layer_name != 'global_step':
                temp_list = [int(i) for i in layer_size if (is_number(i) and i != '')]
                print(temp_list)
                weights_map.update({l_name: temp_list})
        
        format_csv()
        #data_csv()
   
if __name__ == '__main__':
    main(sys.argv)
