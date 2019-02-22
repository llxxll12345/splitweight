"""
    This module defines different blocks in the VGG16 neural network.
"""
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, Input
from keras.backend import bias_add

class VGG16:
    def __init__(self, data):
        self.data = data

    def set_weights_for(self, layer, name):
        layer.set_weights(self.data[name + '-weights'])
        bias_add(layer, self.data[name + '-biases'])

    def block12345(self):
        """ Defines all Conv layer in VGG16. """
        image = Input(shape=(224, 224, 3))

        # block 1
        layer = Conv2D(64, (3, 3), activation='relu', padding='same')(image)
        self.set_weights_for(layer, 'vgg_16-conv1-conv1_1')

        layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv1-conv1_2')
        layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

        # block 2
        layer = Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv2-conv2_1')
        layer = Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv2-conv2_2')
        layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

        # block 3
        layer = Conv2D(256, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv3-conv3_1')
        layer = Conv2D(256, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv3-conv3_2')
        layer = Conv2D(256, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv3-conv3_3')
        layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

        # block 4
        layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv4-conv4_1')
        layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv4-conv4_2')
        layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv4-conv4_3')
        layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

        # block 5
        layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv5-conv5_1')
        layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv5-conv5_2')
        layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
        self.set_weights_for(layer, 'vgg_16-conv5-conv5_3')
        layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

        layer = Flatten()(layer)
        model = Model(image, layer)
        return model


    def fc1(self):
        """ Block 6 is the first fully connected layer in VGG16. """
        image = Input(shape=(25088,))
        layer = Dense(2048, activation='relu')(image)
        self.set_weights_for(layer, 'vgg_16-fc6')
        model = Model(image, layer)
        return model


    def fc2(self):
        """ Block 7 is the last two fully connected layer. """
        image = Input(shape=(4096,))
        layer = Dense(4096, activation='relu')(image)
        self.set_weights_for(layer, 'vgg_16-fc7')
        layer = Dense(1000, activation='softmax')(layer)
        self.set_weights_for(layer, 'vgg_16-fc8')
        model = Model(image, layer)
        return model

    
