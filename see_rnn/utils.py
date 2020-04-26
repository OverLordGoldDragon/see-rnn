import numpy as np
from pathlib import Path
from ._backend import K, WARN, NOTE


def _validate_args(name, idx, layer):
    def _ensure_list(name, idx, layer):
        # if None, leave as-is
        name, idx, layer = [[x] if not isinstance(x, (list, type(None))) else x
                            for x in (name, idx, layer)]
        # ensure external lists unaffected
        name, idx, layer = [x.copy() if isinstance(x, list) else x
                            for x in (name, idx, layer)]
        return name, idx, layer

    def _one_requested(name, idx, layer):
        return len(layer or name or idx) == 1  # give `layer` precedence

    find_layer = idx is not None or name is not None
    if find_layer and layer is not None:
        print(WARN, "`layer` will override `idx` & `name`")

    if layer is None:
        no_info  = idx is None and name is None
        too_much_info = idx is not None and name is not None
        if no_info or too_much_info:
            raise Exception("supply one (and only one) of `idx`, `name`")

    name, idx, layer = _ensure_list(name, idx, layer)
    return name, idx, layer, _one_requested(name, idx, layer)


def _process_rnn_args(model, name, idx, layer, input_data, labels,
                      mode, data=None, norm=None):
    """Helper method to validate `input_data` & `labels` dims, layer info args,
       `mode` arg, and fetch various pertinent RNN attributes.
    """

    from .inspect_gen import get_layer, get_gradients
    from .inspect_rnn import get_rnn_weights

    def _validate_args_(name, idx, layer, input_data, labels,
                        mode, norm, data):
        _validate_args(name, idx, layer)

        if data is not None:
            got_inputs = (input_data is not None) or (labels is not None)
            if got_inputs:
                print(NOTE, "`data` will override `input_data`, `labels`, "
                      "and `mode`")
            if not isinstance(data, list):
                raise Exception("`data` must be a list of kernel & gate matrices")

            if not (isinstance(data[0], np.ndarray) or isinstance(data[0], list)):
                raise Exception("`data` list elements must be numpy arrays "
                                + "or lists")
            elif isinstance(data[0], list):
                if not isinstance(data[0][0], np.ndarray):
                    raise Exception("`data` list elements' elements must be "
                                    + "numpy arrays")

        if mode not in ['weights', 'grads']:
            raise Exception("`mode` must be one of: 'weights', 'grads'")
        if mode == 'grads' and (input_data is None or labels is None):
            raise Exception("must supply input_data and labels for mode=='grads'")
        if mode == 'weights' and (input_data is not None or labels is not None):
            print(NOTE, "`input_data` and `labels will` be ignored for "
                  "`mode`=='weights'")

        is_iter = (isinstance(norm, list) or isinstance(norm, tuple) or
                   isinstance(norm, np.ndarray))
        is_iter_len2 = is_iter and len(norm)==2
        if (norm is not None) and (norm != 'auto') and not is_iter_len2:
            raise Exception("`norm` must be None, 'auto' or iterable ( "
                            + "list, tuple, np.ndarray) of length 2")

    _validate_args_(name, idx, layer, input_data, labels, mode, norm, data)
    if layer is None:
        layer = get_layer(model, name, idx)
    rnn_type = _validate_rnn_type(layer, return_value=True)

    gate_names = _rnn_gate_names(rnn_type)
    n_gates  = len(gate_names)
    is_bidir   = hasattr(layer, 'backward_layer')
    rnn_dim    = layer.layer.units if is_bidir else layer.units
    direction_names = ['FORWARD', 'BACKWARD'] if is_bidir else [[]]
    if 'CuDNN' in rnn_type:
        uses_bias = True
    else:
        uses_bias  = layer.layer.use_bias if is_bidir else layer.use_bias

    if data is None:
        if mode=='weights':
            data = get_rnn_weights(model, name, idx, as_tensors=False,
                                   concat_gates=True)
        else:
            data = get_gradients(model, input_data, labels,
                                 layer=layer, mode='weights')

    rnn_info = dict(rnn_type=rnn_type, gate_names=gate_names,
                    n_gates=n_gates, is_bidir=is_bidir,
                    rnn_dim=rnn_dim, uses_bias=uses_bias,
                    direction_names=direction_names)
    return data, rnn_info


def _validate_rnn_type(rnn_layer, return_value=False):
    if hasattr(rnn_layer, 'backward_layer'):
        rnn_type = type(rnn_layer.layer).__name__
    else:
        rnn_type = type(rnn_layer).__name__

    supported_rnns = ['LSTM', 'GRU', 'CuDNNLSTM', 'CuDNNGRU',
                      'SimpleRNN', 'IndRNN']
    if rnn_type not in supported_rnns:
        raise Exception("unsupported RNN type `%s` - must be one of: %s" % (
                        rnn_type, ', '.join(supported_rnns)))
    if return_value:
        return rnn_type


def _rnn_gate_names(rnn_type):
    return {'LSTM':      ['INPUT',  'FORGET', 'CELL', 'OUTPUT'],
            'GRU':       ['UPDATE', 'RESET',  'NEW'],
            'CuDNNLSTM': ['INPUT',  'FORGET', 'CELL', 'OUTPUT'],
            'CuDNNGRU':  ['UPDATE', 'RESET',  'NEW'],
            'SimpleRNN': [''],
            'IndRNN':    [''],
            }[rnn_type]


def K_eval(x, backend=K):
    """Workaround to TF2.0/2.1-Graph's buggy tensor evaluation"""
    K = backend
    try:
        return K.get_value(K.to_dense(x))
    except Exception as e:
        eval_fn = K.function([], [x])
        return eval_fn([])[0]


def _filter_duplicates_by_keys(keys, *data):
    collected = []
    for k in keys:
        if k in collected:
            for i in range(len(data)):
                data[i].pop(keys.index(k))
            keys.pop(keys.index(k))
            collected.append(k)
    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]
    return keys, data

def _save_rnn_fig(figs, savepath, kwargs):
    if len(figs) == 1:
        figs[0].savefig(savepath)
        return

    _dir = str(Path(savepath).parent)
    ext = Path(savepath).suffix
    basename = Path(savepath).stem
    names = [basename + '_0', basename + '_1']

    for fig, name in zip(figs, names):
        fig.savefig(Path(_dir).joinpath(name, ext), **kwargs)
