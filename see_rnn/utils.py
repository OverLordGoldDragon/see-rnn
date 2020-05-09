import numpy as np
from copy import deepcopy
from pathlib import Path

from ._backend import WARN, NOTE


def _kw_from_configs(configs, defaults):
    def _fill_absent_defaults(kw, defaults):
        # override `defaults`, but keep those not in `configs`
        for name, _dict in defaults.items():
            if name not in kw:
                kw[name] = _dict
            else:
                for k, v in _dict.items():
                    if k not in kw[name]:
                        kw[name][k] = v
        return kw

    configs = configs or {}
    configs = deepcopy(configs)  # ensure external dict unchanged
    for key in configs:
        if key not in defaults:
            raise ValueError(f"unexpected `configs` key: {key}; "
                             "supported are: %s" % ', '.join(list(defaults)))

    kw = deepcopy(configs)  # ensure external dict unchanged
    # override `defaults`, but keep those not in `configs`
    kw = _fill_absent_defaults(configs, defaults)
    return kw


def _validate_args(_id, layer=None):
    def _ensure_list(_id, layer):
        # if None, leave as-is
        _ids, layer = [[x] if not isinstance(x, (list, type(None))) else x
                       for x in (_id, layer)]
        # ensure external lists unaffected
        _ids, layer = [x.copy() if isinstance(x, list) else x
                       for x in (_ids, layer)]
        return _ids, layer

    def _ids_to_names_and_idxs(_ids):
        names, idxs = [], []
        for _id in _ids:
            if not isinstance(_id, (str, int, tuple)):
                tp = type(_id).__name__
                raise ValueError("unsupported _id list element type: %s" % tp
                                 + "; supported are: str, int, tuple")
            if isinstance(_id, str):
                names.append(_id)
            else:
                if isinstance(_id, int):
                    idxs.append(_id)
                else:
                    assert all(isinstance(x, int) for x in _id)
                    idxs.append(_id)
        return names or None, idxs or None

    def _one_requested(_ids, layer):
        return len(layer or _ids) == 1  # give `layer` precedence

    if (_id is not None and layer is not None):
        print(WARN, "`layer` will override `_id`")

    _ids, layer = _ensure_list(_id, layer)
    if _ids is None:
        names, idxs = None, None
    else:
        names, idxs = _ids_to_names_and_idxs(_ids)
    return names, idxs, layer, _one_requested(_ids, layer)


def _process_rnn_args(model, _id, layer, input_data, labels, mode,
                      data=None, norm=None):
    """Helper method to validate `input_data` & `labels` dims, layer info args,
       `mode` arg, and fetch various pertinent RNN attributes.
    """

    from .inspect_gen import get_layer, get_gradients
    from .inspect_rnn import get_rnn_weights

    def _validate_args_(_id, layer, input_data, labels, mode, norm, data):
        _validate_args(_id, layer)

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

    _validate_args_(_id, layer, input_data, labels, mode, norm, data)
    if layer is None:
        layer = get_layer(model, _id)
    rnn_type = _validate_rnn_type(layer, return_value=True)

    gate_names = _rnn_gate_names(rnn_type)
    n_gates  = len(gate_names)
    is_bidir = hasattr(layer, 'backward_layer')
    rnn_dim  = layer.layer.units if is_bidir else layer.units
    direction_names = ['FORWARD', 'BACKWARD'] if is_bidir else [[]]
    if 'CuDNN' in rnn_type:
        uses_bias = True
    else:
        uses_bias  = layer.layer.use_bias if is_bidir else layer.use_bias

    if data is None:
        if mode=='weights':
            data = get_rnn_weights(model, _id, as_tensors=False,
                                   concat_gates=True)
        else:
            data = get_gradients(model, None, input_data, labels,
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


def _filter_duplicates_by_keys(keys, *data):
    def _second_index(ls, k):
        return [i for i, x in enumerate(ls) if x == k][1]

    collected = []
    for k in keys:
        if k in collected:
            for i in range(len(data)):
                data[i].pop(_second_index(keys, k))
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
