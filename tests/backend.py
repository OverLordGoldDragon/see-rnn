import os
import tempfile
import contextlib
import shutil
import tensorflow as tf
import numpy as np
import random
from termcolor import colored


#### Environment configs ######################################################
# for testing locally
os.environ['TF_KERAS'] = os.environ.get("TF_KERAS", '1')
os.environ['TF_EAGER'] = os.environ.get("TF_EAGER", '1')

TF_KERAS = bool(os.environ['TF_KERAS'] == '1')
TF_EAGER = bool(os.environ['TF_EAGER'] == '1')
TF_2 = bool(tf.__version__[0] == '2')

if TF_2:
    USING_GPU = bool(tf.config.list_logical_devices('GPU') != [])
else:
    USING_GPU = bool(tf.config.experimental.list_logical_devices('GPU') != [])

if not TF_EAGER:
    tf.compat.v1.disable_eager_execution()
elif not TF_2:
    raise Exception("see-rnn does not support TF1 in Eager execution")

print(("{}\nTF version: {}\nTF uses {}\nTF executing in {} mode\n"
       "TF_KERAS = {}\n{}\n").format("=" * 80,
                                     tf.__version__,
                                     "GPU"   if USING_GPU else "CPU",
                                     "Eager" if TF_EAGER  else "Graph",
                                     "1"     if TF_KERAS  else "0",
                                     "=" * 80))

#### Imports + Funcs ##########################################################
if TF_KERAS:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Input, LSTM, GRU, TimeDistributed, Dense
    from tensorflow.keras.layers import SimpleRNN, Bidirectional, concatenate
    from tensorflow.keras.layers import Activation, Conv1D
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    if USING_GPU:
        from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
else:
    import keras.backend as K
    from keras.layers import Input, LSTM, GRU, TimeDistributed, Dense
    from keras.layers import SimpleRNN, Bidirectional, concatenate
    from keras.layers import Activation, Conv1D
    from keras.models import Model
    from keras.regularizers import l1, l2, l1_l2
    if USING_GPU:
        from keras.layers import CuDNNLSTM, CuDNNGRU

if not USING_GPU:
    CuDNNLSTM, CuDNNGRU = None, None


WARN = colored("WARNING:", 'red')

@contextlib.contextmanager
def tempdir(dirpath=None):
    if dirpath is not None and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        os.mkdir(dirpath)
    elif dirpath is None:
        dirpath = tempfile.mkdtemp()
    else:
        os.mkdir(dirpath)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


def make_model(rnn_layer, batch_shape, units=6, bidirectional=False,
               use_bias=True, activation='tanh', recurrent_dropout=0,
               include_dense=False, IMPORTS={}, new_imports={}):
    Input         = IMPORTS['Input']
    Bidirectional = IMPORTS['Bidirectional']
    Model         = IMPORTS['Model']
    if new_imports != {}:
        Input         = new_imports['Input']
        Bidirectional = new_imports['Bidirectional']
        Model         = new_imports['Model']

    kw = {}
    if not use_bias:
        kw['use_bias'] = False     # for CuDNN or misc case
    if activation == 'relu':
        kw['activation'] = 'relu'  # for nan detection
        kw['recurrent_dropout'] = recurrent_dropout
    kw.update(dict(kernel_regularizer=l1_l2(1e-4),
                   recurrent_regularizer=l1_l2(1e-4),
                   bias_regularizer=l1_l2(1e-4)))

    ipt = Input(batch_shape=batch_shape)
    if bidirectional:
        x = Bidirectional(rnn_layer(units, return_sequences=True, **kw))(ipt)
    else:
        x = rnn_layer(units, return_sequences=True, **kw)(ipt)
    if include_dense:
        x = TimeDistributed(Dense(units, bias_regularizer=l1_l2(1e-4)))(x)
    out = rnn_layer(units, return_sequences=False)(x)

    model = Model(ipt, out)
    model.compile('adam', 'mse')
    return model


def make_data(batch_shape, units):
    return (np.random.randn(*batch_shape),
            np.random.uniform(-1, 1, (batch_shape[0], units)),
            np.random.uniform(0, 2, batch_shape[0]))


def train_model(model, iterations):
    batch_shape = K.int_shape(model.input)
    units = model.layers[2].units
    x, y, sw = make_data(batch_shape, units)

    for i in range(iterations):
        model.train_on_batch(x, y, sw)
        print(end='.')  # progbar
        if i % 40 == 0:
            x, y, sw = make_data(batch_shape, units)


def _make_nonrnn_model():
    if os.environ.get("TF_KERAS", '0') == '1':
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
    else:
        from keras.layers import Input, Dense
        from keras.models import Model
    ipt = Input((16,))
    out = Dense(16)(ipt)
    model = Model(ipt, out)
    model.compile('adam', 'mse')
    return model


def reset_seeds(reset_graph_with_backend=None, verbose=1):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        if verbose:
            print("KERAS AND TENSORFLOW GRAPHS RESET")

    np.random.seed(1)
    random.seed(2)
    tf.compat.v1.set_random_seed(3)
    if verbose:
        print("RANDOM SEEDS RESET")


def pass_on_error(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except BaseException as e:
        print("Task Failed Successfully:", e)
