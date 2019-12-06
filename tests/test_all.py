import numpy as np
import random
import tensorflow as tf
import os
from termcolor import cprint, colored

from . import K
from . import Input, LSTM, GRU, SimpleRNN, Bidirectional
from . import Model
from see_rnn import get_layer_gradients, show_features_0D
from see_rnn import show_features_1D, show_features_2D, rnn_summary
from see_rnn import rnn_heatmap, rnn_histogram


USING_GPU = bool(tf.config.experimental.list_physical_devices('GPU') != [])

if USING_GPU:
    from . import CuDNNLSTM, CuDNNGRU
    print("TF uses GPU")

print("TF version: %s" % tf.__version__)
TF_KERAS = bool(os.environ.get("TF_KERAS", '0') == '1')
TF_EAGER = bool(os.environ.get("TF_EAGER", '0') == '1')
TF_2 = (tf.__version__[0] == '2')

warn_str = colored("WARNING: ", 'red')

if TF_EAGER:
    if not TF_2:
        tf.enable_eager_execution()
    print("TF running eagerly")
else:
    if TF_2:
        tf.compat.v1.disable_eager_execution()
    print("TF running in graph mode")
if TF_2 and not TF_KERAS:
    print(warn_str + "LSTM, CuDNNLSTM, and CuDNNGRU imported `from keras` "
          + "are not supported in TF2 Graph execution, and will be skipped")


def test_all():
    units = 6
    batch_shape = (8, 100, 2*units)
    iterations = 20

    kwargs1 = dict(batch_shape=batch_shape, units=units, bidirectional=False)
    kwargs2 = dict(batch_shape=batch_shape, units=units, bidirectional=True)

    if TF_2 and not TF_KERAS:
        rnn_layers = GRU, SimpleRNN
    else:
        rnn_layers = LSTM, GRU, SimpleRNN
        if USING_GPU:
            rnn_layers += CuDNNLSTM, CuDNNGRU

    model_names = [layer.__name__ for layer in rnn_layers]
    model_names = [(prefix + name) for prefix in ("uni-", "bi-")
                   for name in model_names]

    configs = [dict(rnn_layer=rnn_layer, **kwargs)
               for kwargs in (kwargs1, kwargs2) for rnn_layer in rnn_layers]

    tests_ran = 0
    for config, model_name in zip(configs, model_names):
        reset_seeds(reset_graph_with_backend=K)
        model = make_model(**config)
        train_model(model, iterations, batch_shape, units)

        rnn_summary(model.layers[1])
        _test_activations_gradients(model)
        _test_weights_gradients(model)
        _test_weights(model)

        tests_ran += 1
        cprint("\n<< %s TESTED >>\n" % model_name, 'green')
    assert tests_ran == len(configs)
    cprint("\n<< ALL MODELS TESTED >>\n", 'green')


def _test_activations_gradients(model):
    x, y = make_data(K.int_shape(model.input), model.layers[2].units)
    name = model.layers[1].name
    grads_all  = get_layer_gradients(model, x, y, layer_name=name, mode='activations')
    grads_last = get_layer_gradients(model, x, y, layer_idx=2,     mode='activations')

    kwargs1 = dict(n_rows=None, show_xy_ticks=[0, 0], show_borders=True,
                   max_timesteps=50, show_title='grads')
    kwargs2 = dict(n_rows=2,    show_xy_ticks=[1, 1], show_borders=False,
                   max_timesteps=None)

    show_features_1D(grads_all[0], **kwargs1)
    show_features_1D(grads_all,    **kwargs2)
    show_features_2D(grads_all[0], norm=(-.01, .01), show_colorbar=True, **kwargs1)
    show_features_2D(grads_all,    norm=None,        reflect_half=True,  **kwargs2)
    show_features_0D(grads_last,   marker='o', color=None, show_title='grads')
    show_features_0D(grads_last,   marker='x', color='blue')
    print('\n')  # improve separation


def _test_weights_gradients(model):
    x, y = make_data(K.int_shape(model.input), model.layers[2].units)
    name = model.layers[1].name
    kws = dict(input_data=x, labels=y, mode='grads')

    rnn_histogram(model, layer_name=name, bins=100, **kws)
    rnn_heatmap(model,   layer_name=name,           **kws)


def _test_weights(model):
    name = model.layers[1].name
    rnn_histogram(model, layer_name=name, mode='weights', bins=100)
    rnn_heatmap(model,   layer_name=name, mode='weights')


def test_misc():  # misc tests to improve coverage %
    units = 6
    batch_shape = (8, 100, 2*units)

    model = make_model(GRU, batch_shape)

    x, y = make_data(batch_shape, units)
    grads = get_layer_gradients(model, x, y, layer_idx=1)
    grads_4D = np.expand_dims(grads, -1)
    _grads = np.transpose(grads, (2, 1, 0))

    show_features_2D(_grads, n_rows=1.5, channel_axis=0)
    show_features_2D(_grads[:, :, 0],    channel_axis=0)

    from see_rnn.inspect_gen import _make_grads_fn, get_layer

    def _pass_on_error(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            pass

    _pass_on_error(show_features_0D, grads)
    _pass_on_error(show_features_0D, grads_4D)
    _pass_on_error(show_features_1D, grads_4D)
    _pass_on_error(show_features_2D, grads_4D)
    _pass_on_error(show_features_2D, grads, channel_axis=1)
    _pass_on_error(get_layer_gradients, model, x, y, 1, mode='cactus')
    _pass_on_error(get_layer_gradients, model, x, y, 1, 'gru', model.layers[1])
    _pass_on_error(_make_grads_fn, model, model.layers[1], mode='banana')
    _pass_on_error(get_layer, model)

    get_layer(model, layer_name='gru')

    from importlib import reload

    from see_rnn import inspect_gen, inspect_rnn
    for flag in ['True', 'False']:
        os.environ['TF_KERAS'] = flag
        reload(inspect_gen)
        reload(inspect_rnn)
        from see_rnn.inspect_rnn import get_layer_gradients as grg
        from see_rnn.inspect_rnn import rnn_summary as rs

        _pass_on_error(grg, model, x, y, 1)
        rs(model.layers[1])


def make_model(rnn_layer, batch_shape, units=6, bidirectional=False):
    ipt = Input(batch_shape=batch_shape)
    if bidirectional:
        x = Bidirectional(rnn_layer(units, return_sequences=True))(ipt)
    else:
        x = rnn_layer(units, return_sequences=True)(ipt)
    out = rnn_layer(units, return_sequences=False)(x)
    model = Model(ipt, out)
    model.compile('adam', 'mse')
    return model


def make_data(batch_shape, units):
    return (np.random.randn(*batch_shape),
            np.random.uniform(-1, 1, (batch_shape[0], units)))


def train_model(model, iterations, batch_shape, units):
    x, y = make_data(batch_shape, units)
    for i in range(iterations):
        model.train_on_batch(x, y)
        print(end='.')  # progbar
        if i % 40 == 0:
            x, y = make_data(batch_shape, units)


def reset_seeds(reset_graph_with_backend=None, verbose=1):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        if verbose:
            print("KERAS AND TENSORFLOW GRAPHS RESET")

    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    if verbose:
        print("RANDOM SEEDS RESET")
