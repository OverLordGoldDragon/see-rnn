from .inspect_gen import get_layer
from .utils import _validate_args, _validate_rnn_type
from ._backend import K, TF_KERAS, WARN


def get_rnn_weights(model, _id, layer=None, as_tensors=False, concat_gates=True):
    """Retrievers RNN layer weights.

    Arguments:
        model: keras.Model/tf.keras.Model.
        idx: int. Index of layer to fetch, via model.layers[idx].
        name: str. Name of layer (can be substring) to be fetched. Returns
              earliest match if multiple found.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `idx` and `name`.
        as_tensors: If True, returns weight tensors instead of array values.
               NOTE: in Eager, both are returned.
        concat_gates: If True, returns kernel weights are signle concatenated
               matrices, instead of individual per-gate weight lists.
    """

    names, idxs, *_ = _validate_args(_id, layer)
    name = names[0] if names is not None else None
    idx  = idxs[0]  if idxs  is not None else None

    if layer is None:
        layer = get_layer(model, name or idx)
    rnn_type = _validate_rnn_type(layer, return_value=True)
    IS_CUDNN = 'CuDNN' in rnn_type

    if hasattr(layer, 'backward_layer'):
        l = layer
        forward_cell  = l.forward_layer  if IS_CUDNN else l.forward_layer.cell
        backward_cell = l.backward_layer if IS_CUDNN else l.backward_layer.cell

        forward_cell_weights  = _get_cell_weights(forward_cell,  as_tensors,
                                                  concat_gates)
        backward_cell_weights = _get_cell_weights(backward_cell, as_tensors,
                                                  concat_gates)
        return forward_cell_weights + backward_cell_weights
    else:
        cell = layer if IS_CUDNN else layer.cell
        return _get_cell_weights(cell, as_tensors, concat_gates)


def _get_cell_weights(rnn_cell, as_tensors=True, concat_gates=True):
    """Retrieves RNN layer weights from their cell(s).
    NOTE: if CuDNNLSTM or CuDNNGRU cell, `rnn_cell` must be the layer instead,
          where non-CuDNN cell attributes are stored.
    """
    def _get_cell_info(rnn_cell):
        rnn_type = type(rnn_cell).__name__.replace('Cell', '')

        if rnn_type in ['SimpleRNN', 'IndRNN']:
            gate_names = ['']
        elif rnn_type in ['LSTM', 'CuDNNLSTM']:
            gate_names = ['i', 'f', 'c', 'o']
        elif rnn_type in ['GRU',  'CuDNNGRU']:
            gate_names = ['z', 'r', 'h']

        if ('CuDNN' in rnn_type) or rnn_cell.use_bias:
            kernel_types = ['kernel', 'recurrent_kernel', 'bias']
        else:
            kernel_types = ['kernel', 'recurrent_kernel']

        return rnn_type, gate_names, kernel_types

    rnn_type, gate_names, kernel_types = _get_cell_info(rnn_cell)

    if TF_KERAS and not concat_gates:
        print(WARN, "getting weights per-gate not supported for tf.keras "
              "implementations; fetching per concat_gates==True instead")
        concat_gates = True
    if not concat_gates and not (hasattr(rnn_cell, 'kernel_i') or
                                 hasattr(rnn_cell, 'kernel_z')):
        print(WARN, rnn_type + " is not a gated RNN; fetching per "
              "concat_gates==True instead")
        concat_gates = True

    if concat_gates:
        if as_tensors:
            return [getattr(rnn_cell, w_type) for w_type in kernel_types]
        try:
            return rnn_cell.get_weights()
        except:
            return K.batch_get_value(rnn_cell.weights)

    if 'GRU' in rnn_type:
        kernel_types = ['kernel', 'recurrent_kernel', 'input_bias']
    rnn_weights = []
    for w_type in kernel_types:
        rnn_weights.append([])
        for g_name in gate_names:
            rnn_weights[-1].append(getattr(rnn_cell, w_type + '_' + g_name))

    if as_tensors:
        return rnn_weights
    else:
        for weight_idx in range(len(rnn_weights)):
            for gate_idx in range(len(rnn_weights[weight_idx])):
                rnn_weights[weight_idx][gate_idx] = K.eval(
                    rnn_weights[weight_idx][gate_idx])
        return rnn_weights


def rnn_summary(layer):
    """Prints passed RNN layer's weights, and if applicable, gates information
    NOTE: will not print gates information for tf.keras imports as they
    lack pertinent attributes.
    """

    if hasattr(layer, 'backward_layer'):
        rnn_cells = layer.forward_layer, layer.backward_layer
        IS_CUDNN = "CuDNN" in type(rnn_cells[0]).__name__
        if not IS_CUDNN:
            rnn_cells = [layer.cell for layer in rnn_cells]
    else:
        IS_CUDNN = "CuDNN" in type(layer).__name__
        rnn_cells = [layer] if IS_CUDNN else [layer.cell]

    for idx, rnn_cell in enumerate(rnn_cells):
        if len(rnn_cells) == 2:
            if idx == 0:
                print("// FORWARD LAYER")
            elif idx == 1:
                print("// BACKWARD LAYER")

        kernel_types = ['kernel', 'recurrent_kernel', 'bias']
        if type(layer).__name__ == 'GRU':
            kernel_types += ['input_bias']

        for kernel_type in kernel_types:
            weight_matrix = getattr(rnn_cell, kernel_type, None)
            if weight_matrix is not None:
                print(weight_matrix.name, "-- shape=%s" % weight_matrix.shape)

            if not TF_KERAS:
                [print(key, "-- shape=%s" % val.shape) for key, val in
                    rnn_cell.__dict__.items() if (kernel_type + '_' in key)
                    and (len(key) == len(kernel_type) + 2)]
                print()
