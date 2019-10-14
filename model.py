import tensorflow as tf

class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__(name='UNet')

        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        ])
        self.block3 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')
        ])
        self.block4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')
        ])
        self.block5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same'),
        ])

    def call(self, inputs):
        block1 = self.block1(inputs)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        concat = tf.concat([block2, block3], axis=3)
        block4 = self.block4(concat)
        concat = tf.concat([block1, block4], axis=3)
        block5 = self.block5(concat)
        return block5

