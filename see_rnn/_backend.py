import os
from termcolor import colored


TF_KERAS = os.environ.get("TF_KERAS", '0') == '1'
WARN = colored("WARNING:", 'red')
NOTE = colored("NOTE:", 'blue')


if TF_KERAS:
    import tensorflow.keras.backend as K
    print(NOTE, "`sample_weights` & `learning_phase` not yet supported "
          "for `TF_KERAS`, and will be ignored (%s.py)" % __name__)
else:
    import keras.backend as K
