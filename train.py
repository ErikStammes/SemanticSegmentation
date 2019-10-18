from model import UNet
from dataloader import get_data

import tensorflow as tf
import matplotlib.pyplot as plt

from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 32, "Batch size used during training")
flags.DEFINE_string("train_data", None, "Folder where training data is located")
flags.DEFINE_integer("num_classes", 35, "Number of classes in the dataset")

flags.mark_flag_as_required("train_data")

def plot_result(filename, image, label, prediction):
    plt.figure()
    plt.subplot('131')
    plt.imshow(tf.keras.preprocessing.image.array_to_img(tf.squeeze(image)))
    plt.axis('off')
    plt.subplot('132')
    plt.imshow(tf.keras.preprocessing.image.array_to_img(make_output_plottable(label)))
    plt.axis('off')
    plt.subplot('133')
    plt.imshow(tf.keras.preprocessing.image.array_to_img(make_output_plottable(prediction)))
    plt.axis('off')
    plt.savefig(filename)

def make_output_plottable(output):
    output = tf.squeeze(output)
    output = tf.math.argmax(output, axis=-1)
    output = tf.expand_dims(output, -1)
    return output


def main(argv):
    data = get_data(FLAGS.train_data, FLAGS.num_classes)
    train_data = data.batch(16, drop_remainder=True)
    
    model = UNet(num_classes = FLAGS.num_classes)                                             
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')
    model.fit(train_data, epochs=25)

    for index, (image, label) in enumerate(data.batch(1).take(5)):
        prediction = model.predict(image)
        plot_result(f'results/{index}.png', image, label, prediction)
        

if __name__ == '__main__':
  app.run(main)