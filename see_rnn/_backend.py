import os
from termcolor import colored


TF_KERAS = os.environ.get("TF_KERAS", '1') == '1'
WARN = colored("WARNING:", 'red')
NOTE = colored("NOTE:", 'blue')

try:
    if TF_KERAS:
        from tensorflow.python.keras import backend as K
        from tensorflow.keras.layers import Layer
        from tensorflow.keras.models import Model
    else:
        from keras import backend as K
        from keras.layers import Layer
        from keras.models import Model
except:
    print("WARNING: failed to import TensorFlow or Keras; functionality "
          "is restricted to see_rnn.visuals_gen")
    K, Layer, Model = None, None, None
