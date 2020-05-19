import os
import tensorflow as tf
from termcolor import colored


TF_KERAS = os.environ.get("TF_KERAS", '0') == '1'
TF_22 = bool(float(tf.__version__[:3]) >= 2.2)

WARN = colored("WARNING:", 'red')
NOTE = colored("NOTE:", 'blue')


if TF_KERAS:
    import tensorflow.keras.backend as K
    if not TF_22:
        print(NOTE, "`sample_weights` & `learning_phase` not yet supported "
              "for `TF_KERAS`, and will be ignored (%s.py)" % __name__
              + "(TF 2.2+ overcomes this)")
else:
    import keras.backend as K
