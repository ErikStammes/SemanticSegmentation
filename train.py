from model import UNet
from dataloader import get_data

import tensorflow as tf

from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 32, "Batch size used during training")
flags.DEFINE_string("train_data", None, "Folder where training data is located")

flags.mark_flag_as_required("train_data")

def main(argv):
    train_data = get_data(FLAGS.train_data)
    
    model = UNet()                                             
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')
    model.fit(train_data, epochs=5)

if __name__ == '__main__':
  app.run(main)