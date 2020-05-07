import os
import tempfile
import contextlib
import shutil
import matplotlib
import tensorflow as tf

if not os.environ.get('IS_MAIN', '0') == '1':
    matplotlib.use('template')  # suppress figures for spyder unit testing

TF_KERAS = bool(os.environ.get("TF_KERAS", '0') == '1')
USING_GPU = bool(tf.config.experimental.list_physical_devices('GPU') != [])

if TF_KERAS:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Input, LSTM, GRU, TimeDistributed, Dense
    from tensorflow.keras.layers import SimpleRNN, Bidirectional
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    if USING_GPU:
        from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
else:
    import keras.backend as K
    from keras.layers import Input, LSTM, GRU, TimeDistributed, Dense
    from keras.layers import SimpleRNN, Bidirectional
    from keras.models import Model
    from keras.regularizers import l1, l2, l1_l2
    if USING_GPU:
        from keras.layers import CuDNNLSTM, CuDNNGRU


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
