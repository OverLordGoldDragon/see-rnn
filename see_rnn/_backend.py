import os
from termcolor import colored


TF_KERAS = os.environ.get("TF_KERAS", '0') == '1'

WARN = colored("WARNING:", 'red')
NOTE = colored("NOTE:", 'blue')


if TF_KERAS:
    from tensorflow.python.keras import backend as K
    from tensorflow.keras.layers import Layer
else:
    from keras import backend as K
    from keras.layers import Layer
