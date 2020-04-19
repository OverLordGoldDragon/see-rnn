import numpy as np
import random
import tensorflow as tf
import os
from termcolor import cprint, colored

from . import K
from . import Input, LSTM, GRU, SimpleRNN, Bidirectional
from . import Model
from see_rnn import get_layer_gradients, get_layer_outputs, get_rnn_weights
from see_rnn import weights_norm
from see_rnn import features_0D, features_1D, features_2D
from see_rnn import features_hist, features_hist_v2
from see_rnn import rnn_summary
from see_rnn import rnn_heatmap, rnn_histogram


IMPORTS = dict(K=K, Input=Input, GRU=GRU,
               Bidirectional=Bidirectional, Model=Model)
USING_GPU = bool(tf.config.experimental.list_physical_devices('GPU') != [])

if USING_GPU:
    from . import CuDNNLSTM, CuDNNGRU
    print("TF uses GPU")

print("TF version: %s" % tf.__version__)
TF_KERAS = bool(os.environ.get("TF_KERAS", '0') == '1')
TF_EAGER = bool(os.environ.get("TF_EAGER", '0') == '1')
TF_2 = (tf.__version__[0] == '2')

WARN = colored("WARNING:", 'red')

if TF_EAGER:
    if not TF_2:
        tf.enable_eager_execution()
    print("TF running eagerly")
else:
    if TF_2:
        tf.compat.v1.disable_eager_execution()
    print("TF running in graph mode")
if TF_2 and not TF_KERAS:
    print(WARN, "LSTM, CuDNNLSTM, and CuDNNGRU imported `from keras` "
          + "are not supported in TF2, and will be skipped")


def test_main():
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
        train_model(model, iterations)

        rnn_summary(model.layers[1])
        _test_outputs(model)
        _test_outputs_gradients(model)
        _test_weights_gradients(model)
        _test_weights(model)

        tests_ran += 1
        cprint("\n<< %s TESTED >>\n" % model_name, 'green')
    assert tests_ran == len(configs)
    cprint("\n<< ALL MODELS TESTED >>\n", 'green')


def _test_outputs(model):
    x, _ = make_data(K.int_shape(model.input), model.layers[2].units)
    outs = get_layer_outputs(model, x, layer_idx=1)
    features_1D(outs[:1], show_y_zero=True)
    features_1D(outs[0])
    features_2D(outs)


def _test_outputs_gradients(model):
    x, y = make_data(K.int_shape(model.input), model.layers[2].units)
    name = model.layers[1].name
    grads_all  = get_layer_gradients(model, x, y, layer_name=name, mode='outputs')
    grads_last = get_layer_gradients(model, x, y, layer_idx=2,     mode='outputs')

    kwargs1 = dict(n_rows=None, show_xy_ticks=[0, 0], show_borders=True,
                   max_timesteps=50, title_mode='grads')
    kwargs2 = dict(n_rows=2,    show_xy_ticks=[1, 1], show_borders=False,
                   max_timesteps=None)

    features_1D(grads_all[0],  **kwargs1)
    features_1D(grads_all[:1], **kwargs1)
    features_1D(grads_all,     **kwargs2)
    features_2D(grads_all[0], norm=(-.01, .01), show_colorbar=True, **kwargs1)
    features_2D(grads_all,    norm=None,        reflect_half=True,  **kwargs2)
    features_0D(grads_last,   marker='o', color=None, title_mode='grads')
    features_0D(grads_last,   marker='x', color='blue', ylims=(-.1, .1))
    features_hist(grads_all, bins=100, xlims=(-.01, .01), title="Outs hists")
    features_hist(grads_all, bins=100, n_rows=4)
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


def _test_prefetched_data(model):
    weights = get_rnn_weights(model, layer_idx=1)
    rnn_histogram(model, layer_idx=1, data=weights)
    rnn_heatmap(model,   layer_idx=1, data=weights)


def make_model(rnn_layer, batch_shape, units=6, bidirectional=False,
               use_bias=True, activation='tanh', recurrent_dropout=0,
               new_imports={}):
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

    ipt = Input(batch_shape=batch_shape)
    if bidirectional:
        x = Bidirectional(rnn_layer(units, return_sequences=True, **kw))(ipt)
    else:
        x = rnn_layer(units, return_sequences=True, **kw)(ipt)
    out = rnn_layer(units, return_sequences=False)(x)

    model = Model(ipt, out)
    model.compile('adam', 'mse')
    return model


def make_data(batch_shape, units):
    return (np.random.randn(*batch_shape),
            np.random.uniform(-1, 1, (batch_shape[0], units)))


def train_model(model, iterations):
    batch_shape = K.int_shape(model.input)
    units = model.layers[2].units
    x, y = make_data(batch_shape, units)

    for i in range(iterations):
        model.train_on_batch(x, y)
        print(end='.')  # progbar
        if i % 40 == 0:
            x, y = make_data(batch_shape, units)


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
        pass


def test_errors():  # test Exception cases
    units = 6
    batch_shape = (8, 100, 2*units)

    reset_seeds(reset_graph_with_backend=K)
    model = make_model(GRU, batch_shape, activation='relu',
                       recurrent_dropout=0.3)
    x, y = make_data(batch_shape, units)
    model.train_on_batch(x, y)

    grads = get_layer_gradients(model, x, y, layer_idx=1)
    grads_4D = np.expand_dims(grads, -1)

    from see_rnn.inspect_gen import get_layer, _make_grads_fn

    pass_on_error(features_0D, grads)
    pass_on_error(features_0D, grads_4D)
    pass_on_error(features_1D, grads_4D)
    pass_on_error(features_2D, grads_4D)
    pass_on_error(features_2D, grads)
    pass_on_error(get_layer_gradients, model, x, y, layer_idx=1, mode='cactus')
    pass_on_error(get_layer_gradients, model, x, y, layer_idx=1,
                   layer_name='gru', layer=model.layers[1])
    pass_on_error(_make_grads_fn, model, model.layers[1], mode='banana')
    pass_on_error(features_hist, grads[:, :4, :3], po='tato')
    pass_on_error(features_hist_v2, grads[:, :4, :3], po='tato')
    pass_on_error(get_layer, model)
    pass_on_error(get_layer, model, layer_name='capsule')
    pass_on_error(rnn_heatmap, model, layer_idx=1, input_data=x, labels=y,
                   mode='coffee')
    pass_on_error(rnn_heatmap, model, layer_idx=1, norm=(0, 1, 2))
    pass_on_error(rnn_heatmap, model, layer_idx=1, mode='grads')
    pass_on_error(rnn_histogram, model, layer_idx=1, norm=None)
    pass_on_error(rnn_heatmap, model, layer_index=9001)
    pass_on_error(features_0D, grads, cake='lie')
    pass_on_error(features_1D, grads, pup='not just any')
    pass_on_error(features_2D, grads, true=False)
    outs = get_layer_outputs(model, x, layer_idx=1)
    pass_on_error(rnn_histogram, model, layer_idx=1, data=outs)
    pass_on_error(rnn_histogram, model, layer_idx=1, data=[1])
    pass_on_error(rnn_histogram, model, layer_idx=1, data=[[1]])
    pass_on_error(features_hist, grads, co='vid')

    cprint("\n<< EXCEPTION TESTS PASSED >>\n", 'green')
    assert True


def test_misc():  # test miscellaneous functionalities
    units = 6
    batch_shape = (8, 100, 2*units)

    reset_seeds(reset_graph_with_backend=K)
    model = make_model(GRU, batch_shape, activation='relu',
                       recurrent_dropout=0.3)
    x, y = make_data(batch_shape, units)
    model.train_on_batch(x, y)

    # problem with TF2.0/2.1 graph in K.get_weights()
    pass_on_error(weights_norm, model, 'gru', omit_weight_names='bias',
                   verbose=1)
    pass_on_error(weights_norm, model, 'gru')

    grads = get_layer_gradients(model, x, y, layer_idx=1)

    features_1D(grads,    subplot_samples=True, tight=True, borderwidth=2)
    features_1D(grads[0], subplot_samples=True)
    features_2D(grads.T, n_rows=1.5, tight=True, borderwidth=2)
    features_2D(grads.T[:, :, 0])
    features_hist(grads, show_borders=False, borderwidth=1,
                  show_xy_ticks=[0, 0], title="grads")
    features_hist_v2(grads[:, :4, :3], show_borders=False, xlims=(-.01, .01),
                     ylim=100, borderwidth=1, show_xy_ticks=[0, 0],
                     side_annot='row', title="Grads")
    rnn_histogram(model, layer_idx=1, show_xy_ticks=[0, 0], equate_axes=2)
    rnn_heatmap(model, layer_idx=1, cmap=None, normalize=True, show_borders=False)
    rnn_heatmap(model, layer_idx=1, cmap=None, norm='auto', absolute_value=True)
    rnn_heatmap(model, layer_idx=1, norm=None)
    rnn_heatmap(model, layer_idx=1, norm=(-.004, .004))

    from see_rnn.inspect_gen import get_layer, _detect_nans

    get_layer(model, layer_name='gru')
    get_rnn_weights(model, layer_idx=1, concat_gates=False, as_tensors=True)
    rnn_heatmap(model, layer_idx=1, input_data=x, labels=y, mode='weights')
    _test_prefetched_data(model)

    # test NaN detection
    nan_txt = _detect_nans(np.array([1]*9999 + [np.nan])).replace('\n', ' ')
    print(nan_txt)  # case: print as quantity

    K.set_value(model.optimizer.lr, 1e12)
    train_model(model, iterations=10)
    rnn_histogram(model, layer_idx=1)
    rnn_heatmap(model, layer_idx=1)

    del model
    reset_seeds(reset_graph_with_backend=K)

    # test SimpleRNN & other
    _model = make_model(SimpleRNN, batch_shape, units=128, use_bias=False)
    train_model(_model, iterations=1)  # TF2-Keras-Graph bug workaround
    rnn_histogram(_model, layer_idx=1)  # test _pretty_hist
    K.set_value(_model.optimizer.lr, 1e50)  # SimpleRNNs seem ridiculously robust
    train_model(_model, iterations=20)
    rnn_heatmap(_model, layer_idx=1)
    data = get_rnn_weights(_model, layer_idx=1)
    rnn_heatmap(_model, layer_idx=1, input_data=x, labels=y, data=data)
    os.environ["TF_KERAS"] = '0'
    get_rnn_weights(_model, layer_idx=1, concat_gates=False)
    del _model

    assert True
    cprint("\n<< MISC TESTS PASSED >>\n", 'green')


def test_envs():  # pseudo-tests for coverage for different env flags
    reset_seeds(reset_graph_with_backend=K)
    units = 6
    batch_shape = (8, 100, 2*units)
    x, y = make_data(batch_shape, units)

    from importlib import reload

    from see_rnn import inspect_gen, inspect_rnn, utils
    for flag in ['1', '0']:
        os.environ["TF_KERAS"] = flag
        TF_KERAS = os.environ["TF_KERAS"] == '1'
        reload(inspect_gen)
        reload(inspect_rnn)
        reload(utils)
        from see_rnn.inspect_gen import get_layer_gradients as glg
        from see_rnn.inspect_rnn import rnn_summary as rs
        from see_rnn.utils import _validate_rnn_type as _vrt

        reset_seeds(reset_graph_with_backend=K)
        if TF_KERAS:
            from tensorflow.keras.layers import Input, Bidirectional
            from tensorflow.keras.layers import GRU as _GRU
            from tensorflow.keras.models import Model
            import tensorflow.keras.backend as _K
        else:
            from keras.layers import Input, Bidirectional
            from keras.layers import GRU as _GRU
            from keras.models import Model
            import keras.backend as _K

        reset_seeds(reset_graph_with_backend=_K)
        new_imports = dict(Input=Input, Bidirectional=Bidirectional,
                           Model=Model)
        model = make_model(_GRU, batch_shape, new_imports=new_imports)

        glg(model, x, y, layer_idx=1)
        pass_on_error(glg, model, x, y, 1)
        rs(model.layers[1])

        from see_rnn.inspect_rnn import get_rnn_weights as grw
        grw(model, layer_idx=1, concat_gates=False, as_tensors=True)
        grw(model, layer_idx=1, concat_gates=False, as_tensors=False)
        _test_outputs(model)
        setattr(model.layers[2].cell, 'get_weights', None)
        get_rnn_weights(model, layer_idx=2, concat_gates=True, as_tensors=False)

        _model = _make_nonrnn_model()
        pass_on_error(_vrt, _model.layers[1])
        del model, _model

    assert True
    cprint("\n<< ENV TESTS PASSED >>\n", 'green')
