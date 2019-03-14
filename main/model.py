"""
    This module defines different blocks in the VGG16 neural network.
"""
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, Input
from keras.backend import bias_add
import csv
import numpy as np
import pandas as pd
import cv2
import glob

shape_list = {}
data_list = {}

class VGG16:
    def __init__(self, data):
        self.data = data

    def set_weights_for(self, layer, name):
        layer.set_weights([self.data[name + '-weights'], self.data[name + '-biases']])
        #bias_add(layer, self.data[name + '-biases'])

    def block12345(self):
        """ Defines all Conv layer in VGG16. """
        image = Input(shape=(224, 224, 3))

        # block 1
        # Keras layer type
        layer_c11 = Conv2D(64, (3, 3), activation='relu', padding='same')
        # Tensorflow tensor type
        encoded_c11 = layer_c11(image)
 
        layer_c12 = Conv2D(64, (3, 3), activation='relu', padding='same')
        encoded_c12 = layer_c12(encoded_c11)

        layer_m1 = MaxPooling2D((2, 2), strides=(2, 2))(encoded_c12)

        
        # block 2
        layer_c21 = Conv2D(128, (3, 3), activation='relu', padding='same')
        encoded_c21 = layer_c21(layer_m1)
        
        layer_c22 = Conv2D(128, (3, 3), activation='relu', padding='same')
        encoded_c22 = layer_c22(encoded_c21)
        
        layer_m2 = MaxPooling2D((2, 2), strides=(2, 2))(encoded_c22)

        # block 3
        layer_c31 = Conv2D(256, (3, 3), activation='relu', padding='same')
        encoded_c31 = layer_c31(layer_m2)
        
        layer_c32 = Conv2D(256, (3, 3), activation='relu', padding='same')
        encoded_c32 = layer_c32(encoded_c31)
        
        layer_c33 = Conv2D(256, (3, 3), activation='relu', padding='same')
        encoded_c33 = layer_c33(encoded_c32)

        layer_m3 = MaxPooling2D((2, 2), strides=(2, 2))(encoded_c33)

        # block 4
        layer_c41 = Conv2D(512, (3, 3), activation='relu', padding='same')
        encoded_c41 = layer_c41(layer_m3)
        
        layer_c42 = Conv2D(512, (3, 3), activation='relu', padding='same')
        encoded_c42 = layer_c42(encoded_c41)

        layer_c43 = Conv2D(512, (3, 3), activation='relu', padding='same')
        encoded_c43 = layer_c43(encoded_c42)

        layer_m4 = MaxPooling2D((2, 2), strides=(2, 2))(encoded_c43)

        # block 5
        layer_c51 = Conv2D(512, (3, 3), activation='relu', padding='same')
        encoded_c51 = layer_c51(layer_m4)
        
        layer_c52 = Conv2D(512, (3, 3), activation='relu', padding='same')
        encoded_c52 = layer_c52(encoded_c51)

        layer_c53 = Conv2D(512, (3, 3), activation='relu', padding='same')
        encoded_c53 = layer_c53(encoded_c52)
       
        layer_m4 = MaxPooling2D((2, 2), strides=(2, 2))(encoded_c53)

        layer_output = Flatten()(layer_m4)

        # output shape = (1, 25088)
        model = Model(inputs=image, outputs=layer_output)
        self.set_weights_for(layer_c11, 'vgg_16-conv1-conv1_1')
        self.set_weights_for(layer_c12, 'vgg_16-conv1-conv1_2')
        self.set_weights_for(layer_c21, 'vgg_16-conv2-conv2_1')
        self.set_weights_for(layer_c22, 'vgg_16-conv2-conv2_2')
        self.set_weights_for(layer_c31, 'vgg_16-conv3-conv3_1')
        self.set_weights_for(layer_c32, 'vgg_16-conv3-conv3_2')
        self.set_weights_for(layer_c33, 'vgg_16-conv3-conv3_3')
        self.set_weights_for(layer_c41, 'vgg_16-conv4-conv4_1')
        self.set_weights_for(layer_c42, 'vgg_16-conv4-conv4_2')
        self.set_weights_for(layer_c43, 'vgg_16-conv4-conv4_3')
        self.set_weights_for(layer_c51, 'vgg_16-conv5-conv5_1')
        self.set_weights_for(layer_c52, 'vgg_16-conv5-conv5_2')
        self.set_weights_for(layer_c53, 'vgg_16-conv5-conv5_3')
        return model


    def fc1(self):
        """ Block 6 is the first fully connected layer in VGG16. """
        image = Input(shape=(25088,))
        fc6 = Dense(4096, activation='relu')
        encoded_fc6 = fc6(image)
        model = Model(inputs = image, outputs = encoded_fc6)
        self.set_weights_for(fc6, 'vgg_16-fc6')
        return model


    def fc2(self):
        """ Block 7 is the last two fully connected layer. """
        image = Input(shape=(4096,))
        fc7 = Dense(4096, activation='relu')
        encoded_fc7 = fc7(image)

        fc8 = Dense(1000, activation='softmax')
        encoded_fc8 = fc8(image)
        model = Model(inputs = image, outputs = encoded_fc8)
        self.set_weights_for(fc7, 'vgg_16-fc7')
        self.set_weights_for(fc8, 'vgg_16-fc8')
        return model

'''
def reconstruct():
    with open('csv_output/format.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            s_list = [int(i) for i in row['shape'][1:-1].split(', ')]
            shape_list.update({name: s_list})
            print(s_list)

            input_name = 'csv_output/' + name + '.csv'
            #reader = csv.reader(open(input_name, "r"), delimiter=",")
            #x = [float(i) for i in list(reader)[0]]
            #print(len(x))
            
            #arr = np.array(x).astype("float")
            x = pd.read_csv(input_name)
            #print("describe:", x.describe())
            #arr = x.reshape(s_list)
            #data_list.update({name: arr})
'''

def reconstruct():
    file_list = glob.glob("h5_output/*.h5")
    with open('csv_output/format.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            s_list = [int(i) for i in row['shape'][1:-1].split(', ')]
            shape_list.update({name: s_list})
            print(s_list)

    for file_name in file_list:
        df = pd.read_hdf(file_name)
        layer_name = file_name[10:-3]
        print(len(df[layer_name]))
        reshaped = np.array(df[layer_name].values).reshape(shape_list[layer_name])
        data_list.update({layer_name: reshaped})



def test():
    reconstruct()
    big_model = VGG16(data_list)
    block_model = big_model.block12345()
    fc1m = big_model.fc1()
    fc2m = big_model.fc2()


    img = cv2.imread("test.jpeg")
    cv2.imshow("test.jpeg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    print(resized.shape)
    resized = np.array(resized)
    resized = resized.reshape([1, 224, 224, 3])
    result1 = block_model.predict(resized)
    print(result1.shape)
    result2 = fc1m.predict(result1)
    print(result2.shape)
    result3 = fc2m.predict(result2)
    print(result3.shape)

if __name__ == '__main__':
    test()