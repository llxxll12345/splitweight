import numpy as np
import sys
import os

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags


def get_layer_names(file_name, do_print=True):
    # get a certain tensor myreader.get_tensor(name)
    res = []
    try:
        myreader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var2shape = myreader.get_variable_to_shape_map()
        for k in sorted(var2shape):
            if do_print == True:
                print(k + "," + str(myreader.get_tensor(k).shape) + "," + \
                        str(myreader.get_tensor(k).dtype))
            res.append(k)
        return res
    except Exception as e:
        print(str(e))


def my_drop(input_file, out_file, layers=None, to_be_dropped=False, percent=1.0, mode=0):
    '''
        to_be_dropped means that whether the layers are the ones to be dropped or to be remained
        to_be_dropped = True -> layers are to be dropped
        to_be_dropped = False -> layers not to be dropped

        mode indicates whether the dropped layers are stored(0) or layers not dropped(1) are stored
    '''
    dropped = {}
    # read from the checkpoint file
    myreader = pywrap_tensorflow.NewCheckpointReader(input_file)
    # get variable to shape map
    var2shape = myreader.get_variable_to_shape_map()
    target = layers
    names = get_layer_names(input_file, do_print=False)

    # for each key
    for k in sorted(var2shape):
        # wieghts of this layer
        weights = myreader.get_tensor(k)
        # get the indices of the non-zero entries mapped to each key
        indice = np.transpose(np.nonzero(myreader.get_tensor(k)))
        # get the shape of the tensor mapped to
        shape = myreader.get_tensor(k).shape
        #print(weights)

        if (k in target and to_be_dropped == True) or \
                (k not in target and to_be_dropped == False):
            #print("name of tensor to be dropped: ", )
            #print("tensor_name: ", k, " shape: ", shape)
            #print("drop percentage: ", percent)

            if percent == 1.0:
                 # percent == 1.0 drop all of the weights in the layer
                if mode == 0:
                    dropped[k] = tf.Variable(tf.zeros(shape))
                #print("Set the entire layer to zero.")
            elif percent == 0.0:
                dropped[k] = tf.Variable(tf.constant(weights))
                #print("Keep the entire layer.")
            else:
                if mode == 0:
                    # flatten the weights into one_dimension vector
                    flatten_weights = weights.flatten()
                    # number of variables to be set zero
                    zero_number = int(percent * len(flatten_weights))
                    # drop the first n weights
                    zero_indice = range(zero_number)
                    flatten_weights[zero_indice] = 0.0

                    # reshape the flattened weights to its original shape
                    new_weights = flatten_weights.reshape(shape)
                    print(new_weights)

                    #dropped[k + "_idx"] = tf.Variable(tf.constant(indice))
                    dropped[k + "_shape"] = tf.Variable(tf.constant(shape))
                else:
                    continue
        else:
            # layers to be kept
            print("Keeping: ", k)
            #dropped[k] = tf.Variable(tf.constant(weights))
            #X = tf.Variable(np.zeros(shape), dtype=tf.float32)
            #X = tf.get_variable(k, initializer = tf.zeros_initializer, shape=shape, dtype=tf.float32)
            #with tf.Session() as sess:
                #sess.run(tf.global_variables_initializer())
                #X.assign(weights).eval()
                #print("Evaled change.")
            dropped[k] = tf.Variable(tf.constant(weights))
            dropped[k + "_shape"] = tf.Variable(tf.constant(shape))
    save_data(out_file, dropped)

def save_data(out_file, dropped):
    with tf.Session() as sess:
        # Initialize new variables in a sparse form (Initialize all variables)
        for var in tf.global_variables():
            if tf.is_variable_initialized(var).eval() == False:
                sess.run(tf.variables_initializer([var]))
        final_saver = tf.train.Saver(dropped)
        #if not os.path.exists(out_file[7:]):
        #    os.makedirs(out_file[7:])
        print("Saving model to  " + out_file)
        final_saver.save(sess, out_file + '/'+ out_file[7:] + '.ckpt')


def keep_one(layer_name, infile, outfile):
    my_drop(infile, outfile, layer_name, False, percent=1.0, mode=1)


def main(argv):
    in_file_name = argv[1]
    #out_file_name= argv[2]
    '''
    names = get_layer_names(in_file_name)
    for name in names:
        keep_one(name, in_file_name, 'output/' + '-'.join(name.split('/')))
    '''
    get_layer_names(in_file_name)
    names = [['name: ', 'vgg_16/fc6/weights'],
            ['name: ', 'vgg_16/fc7/biases'],
            ['name: ', 'vgg_16/fc7/weights'],
            ['name: ', 'vgg_16/fc8/biases'],
            ['name: ', 'vgg_16/fc8/weights'],
            ['name: ', 'vgg_16/mean_rgb']]
    for name in names:
        keep_one(name[1], in_file_name, 'output/' + '-'.join(name[1].split('/')))

if __name__ == '__main__':
    main(sys.argv)
