import os
import tensorflow as tf
from termcolor import colored


TF_KERAS = os.environ.get("TF_KERAS", '0') == '1'
TF_EAGER = tf.executing_eagerly()
TF_22 = bool(float(tf.__version__[:3]) >= 2.2)

WARN = colored("WARNING:", 'red')
NOTE = colored("NOTE:", 'blue')


if TF_KERAS:
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Layer
else:
    from keras import backend as K
    from keras.layers import Layer
