import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(ConvBlock, self).__init__()
        self.cnn = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor):
        x = self.cnn(input_tensor)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x

class NetBlock(layers.Layer):
    def __init__(self, filters, strides=1, pooling=True, learnable=False):
        super(NetBlock, self).__init__()
        self.cnn1 = ConvBlock(filters=filters, kernel_size=3, strides=strides)
        self.cnn2 = ConvBlock(filters=filters, kernel_size=3, strides=strides)
        self.cnn3 = ConvBlock(filters=filters, kernel_size=1, strides=strides**2)
        self.pool = pooling
        self.subsample = layers.MaxPooling2D(pool_size=3, strides=2)

        if learnable:
            self.subsample = layers.Conv2D(filters=filters, kernel_size=5, strides=2)

    def call(self, input_tensor):
        x1 = self.cnn1(input_tensor)
        x1 = self.cnn2(x1)

        x2 = self.cnn3(input_tensor)

        x = (x1 + x2) / 2

        if self.pool:
            x = self.subsample(x)
        return x

class Net(tf.keras.Model):
    def __init__(self, channels=[16, 32, 48, 64, 80], learnable=False):
        super(Net, self).__init__()
        self.block1 = NetBlock(filters=channels[0], strides=2, learnable=learnable)
        self.block2 = NetBlock(filters=channels[1], strides=1, learnable=learnable)
        self.block3 = NetBlock(filters=channels[2], strides=1, learnable=learnable)
        self.block4 = NetBlock(filters=channels[3], strides=1, learnable=learnable)
        self.block5 = NetBlock(filters=channels[4], strides=1, pooling=False, learnable=learnable)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(2, activation='softmax')
    
    def call(self, input_tensor):
        x = self.block1(input_tensor)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = self.fc(x)
        return x
    
    def model(self):
        x = tf.keras.Input(shape=(512, 512, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    



