import os

import tensorflow as tf
import numpy as np

from PIL import Image

def list_dir(folder, excluded=[b'.DS_Store']):
    files = [item for item in os.listdir(folder) if item not in excluded]
    files.sort()
    return files

def load_img(filename, img_size, one_hot=False, num_classes=None):
    img = Image.open(filename)
    img.load()
    img = img.resize(img_size) 
    data = np.asarray(img, dtype="int32")
    if one_hot:
        if num_classes is None:
            raise Exception('When using one hot encoding the number of classes should be given')
        data = tf.convert_to_tensor(data)
        return tf.keras.utils.to_categorical(data, num_classes=num_classes)
    else:
        return data[:, :, :3]

def data_loader(folder, img_size, num_classes):
    train_input = list_dir(os.path.join(folder, b'input'))
    train_labels = list_dir(os.path.join(folder, b'labels'))
    for input, label in zip(train_input, train_labels):
        input_fname = os.path.join(folder, b'input', input)
        label_fname = os.path.join(folder, b'labels', label)
        input_img = load_img(input_fname, img_size)
        label_img = load_img(label_fname, img_size, one_hot=True, num_classes=num_classes)
        yield (input_img, label_img)
    
def get_data(folder, num_classes):
    img_size = (256, 128)
    ds = tf.data.Dataset.from_generator(
        data_loader, args=[folder, img_size, num_classes], 
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([128, 256, 3]), tf.TensorShape([128, 256, num_classes]))
    )
    return ds