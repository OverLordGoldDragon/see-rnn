import os
import numpy as np
from .inspect_gen import _get_layer, _make_grads_fn
from termcolor import colored

TF_KERAS = os.environ.get("TF_KERAS", 'False') == 'True'
warn_str = colored("WARNING: ", 'red')
note_str = colored("NOTEL ", 'blue')


def get_rnn_gradients(model, input_data, labels, layer_idx=None,
                      layer_name=None, layer=None, mode='activations',
                      sample_weights=None, learning_phase=1):
    """Retrieves RNN layer gradients w.r.t. activations or weights.
    NOTE: gradients will be clipped if `clipvalue` or `clipnorm` were set.

    Arguments:
        model: keras.Model/tf.keras.Model.
        input_data: np.ndarray & supported formats(1). Data w.r.t. which loss is
               to be computed for the gradient.
        labels: np.ndarray & supported formats. Labels w.r.t. which loss is
               to be computed for the gradient.
        layer_idx: int. Index of layer to fetch, via model.layers[layer_idx].
        layer_name: str. Substring of name of layer to be fetched. Returns
               earliest match if multiple found.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `layer_idx` and `layer_name`.
        mode: str. One of: 'activations', 'weights'. If former, returns grads
               w.r.t. layer outputs(2) - else, w.r.t. layer trainable weights.
        sample_weights: np.ndarray & supported formats. `sample_weight` kwarg
               to model.fit(), etc., weighting individual sample losses.
        learning_phase: int/bool. If 1, uses model in train model - else,
               in inference mode.

    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats
    (2): not necessarily activations. If an Activation layer is used, returns
         the gradients of the target layer's PRE-activations instead. To then
         get grads w.r.t. activations, specify the Activation layer.
    """

    def _validate_args(model, layer_idx, layer_name, layer, mode):
        find_layer = layer_idx is not None or layer_name is not None
        if find_layer and layer is not None:
            print(warn_str + "`layer` will override `layer_idx` & `layer_name`")

        no_info  = layer_idx is None and layer_name is None
        too_much_info = layer_idx is not None and layer_name is not None
        if find_layer and (no_info or too_much_info):
            raise Exception("must supply one (and only one) of "
                            "`layer_idx`, `layer_name`")

        if mode not in ['activations', 'weights']:
            raise Exception("`mode` must be one of: 'activations', 'weights'")

    _validate_args(model, layer_idx, layer_name, layer, mode)
    if layer is None:
        layer = _get_layer(model, layer_idx, layer_name)

    grads_fn = _make_grads_fn(model, layer, mode)
    if TF_KERAS:
        grads = grads_fn([input_data, labels])
    else:
        sample_weights = sample_weights or np.ones(len(input_data))
        grads = grads_fn([input_data, sample_weights, labels, 1])

    while type(grads) == list:
        grads = grads[0]
    return grads


def rnn_summary(layer):
    """Prints passed RNN layer's weights, and if applicable, gates information
    NOTE: will not print gates information for tf.keras imports or CuDNN
          implementations, as they lack pertinent attributes.
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

        for key_attr in ['kernel', 'recurrent_kernel', 'bias']:
            weight_matrix = getattr(rnn_cell, key_attr, None)
            if weight_matrix is not None:
                print(weight_matrix.name, "-- shape=%s" % weight_matrix.shape)

            if not IS_CUDNN and not TF_KERAS:
                [print(key, "-- shape=%s" % val.shape) for key, val in
                    rnn_cell.__dict__.items() if (key_attr + '_' in key)
                    and (len(key) == len(key_attr) + 2)]
                print()
