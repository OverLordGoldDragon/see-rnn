from termcolor import colored

note_str = colored("NOTE: ", 'blue')
warn_str = colored("WARNING: ", 'red')


def _validate_args(model, layer_idx, layer_name, layer):
    find_layer = layer_idx is not None or layer_name is not None
    if find_layer and layer is not None:
        print(warn_str + "`layer` will override `layer_idx` & `layer_name`")

    no_info  = layer_idx is None and layer_name is None
    too_much_info = layer_idx is not None and layer_name is not None
    if find_layer and (no_info or too_much_info):
        raise Exception("must supply one (and only one) of "
                        "`layer_idx`, `layer_name`")


def _process_rnn_args(model, layer_name, layer_idx, layer,
                      input_data, labels, mode):
    """Helper method to validate `input_data` & `labels` dims, layer info args,
       `mode` arg, and fetch various pertinent RNN attributes.
    """

    from .inspect_gen import get_layer, get_layer_gradients
    from .inspect_rnn import get_rnn_weights

    def _validate_args_(model, layer_idx, layer_name, layer, mode,
                        input_data, labels):
        _validate_args(model, layer_idx, layer_name, layer)
        if mode not in ['weights', 'grads']:
            raise Exception("`mode` must be one of: 'weights', 'grads'")
        if mode == 'grads' and (input_data is None or labels is None):
            raise Exception("must supply input_data and labels for mode=='grads'")
        if mode == 'weights' and (input_data is not None or labels is not None):
            print(note_str + "input_data and labels will be ignored for "
                  + "mode=='weights'")

    _validate_args_(model, layer_idx, layer_name, layer, mode, input_data, labels)

    if layer is None:
        layer = get_layer(model, layer_idx, layer_name)
    rnn_type = _validate_rnn_type(layer, return_value=True)

    gate_names = _rnn_gate_names(rnn_type)
    num_gates  = len(gate_names)
    is_bidir   = hasattr(layer, 'backward_layer')
    rnn_dim    = layer.layer.units if is_bidir else layer.units
    direction_names = ['FORWARD', 'BACKWARD'] if is_bidir else [[]]
    if 'CuDNN' in rnn_type:
        uses_bias = True
    else:
        uses_bias  = layer.layer.use_bias if is_bidir else layer.use_bias

    if mode=='weights':
        data = get_rnn_weights(model, layer_idx, layer_name,
                               as_tensors=False, concat_gates=True)
    else:
        data = get_layer_gradients(model, input_data, labels,
                                   layer=layer, mode='weights')

    rnn_info = dict(rnn_type=rnn_type, gate_names=gate_names,
                    num_gates=num_gates, is_bidir=is_bidir,
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
