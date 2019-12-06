import os
from termcolor import colored
import numpy as np
from .utils import _validate_args

TF_KERAS = os.environ.get("TF_KERAS", '0') == '1'
warn_str = colored("WARNING: ", 'red')
note_str = colored("NOTE: ", 'blue')


if TF_KERAS:
    import tensorflow.keras.backend as K
    print(note_str + "`sample_weights` & `learning_phase` not yet supported "
          + "for `TF_KERAS`, and will be ignored (%s.py)" % __name__)
else:
    import keras.backend as K


def get_layer_outputs(model, input_data, layer_name=None, layer_idx=None,
                      layer=None, learning_phase=0):
    """Retrieves layer outputs given input data and layer info.

    Arguments:
        model: keras.Model/tf.keras.Model.
        input_data: np.ndarray & supported formats(1). Data w.r.t. which loss is
               to be computed for the gradient. Only for mode=='grads'.
        labels: np.ndarray & supported formats. Labels w.r.t. which loss is
               to be computed for the gradient. Only for mode=='grads'
        layer_idx: int. Index of layer to fetch, via model.layers[layer_idx].
        layer_name: str. Substring of name of layer to be fetched. Returns
               earliest match if multiple found.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `layer_idx` and `layer_name`
    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats
    """

    _validate_args(model, layer_idx, layer_name, layer)
    layer = get_layer(model, layer_idx, layer_name)
    layers_fn = K.function([model.input, K.learning_phase()], [layer.output])
    return layers_fn([input_data, learning_phase])[0]


def get_layer_gradients(model, input_data, labels, layer_idx=None,
                        layer_name=None, layer=None, mode='activations',
                        sample_weights=None, learning_phase=0):
    """Retrieves layer gradients w.r.t. activations or weights.
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

    def _validate_args_(model, layer_idx, layer_name, layer, mode):
        _validate_args(model, layer_idx, layer_name, layer)
        if mode not in ['activations', 'weights']:
            raise Exception("`mode` must be one of: 'activations', 'weights'")

    _validate_args_(model, layer_idx, layer_name, layer, mode)
    if layer is None:
        layer = get_layer(model, layer_idx, layer_name)

    grads_fn = _make_grads_fn(model, layer, mode)
    if TF_KERAS:
        grads = grads_fn([input_data, labels])
    else:
        sample_weights = sample_weights or np.ones(len(input_data))
        grads = grads_fn([input_data, sample_weights, labels, 1])

    while isinstance(grads, list) and len(grads)==1:
        grads = grads[0]
    return grads


def get_layer(model, layer_idx=None, layer_name=None):
    """Returns layer by index or name.
    If multiple matches are found, returns earliest.
    """

    _validate_args(model, layer_idx, layer_name, layer=None)
    if layer_idx is not None:
        return model.layers[layer_idx]

    layer = [layer for layer in model.layers if layer_name in layer.name]
    if len(layer) > 1:
        print(warn_str + "multiple matching layer names found; "
              + "picking earliest")
    elif len(layer) == 0:
        raise Exception("no layers found w/ names matching "
                        + "substring: '%s'" % layer_name)
    return layer[0]


def _make_grads_fn(model, layer, mode='activations'):
    """Returns gradient computation function w.r.t. layer activations or weights.
    NOTE: gradients will be clipped if `clipnorm` or `clipvalue` were set.
    """

    if mode not in ['activations', 'weights']:
        raise Exception("`mode` must be one of: 'activations', 'weights'")

    params = layer.output if mode=='activations' else layer.trainable_weights
    grads = model.optimizer.get_gradients(model.total_loss, params)

    if TF_KERAS:
        inputs = [model.inputs[0], model._feed_targets[0]]
    else:
        inputs = [model.inputs[0], model.sample_weights[0],
                  model._feed_targets[0], K.learning_phase()]
    return K.function(inputs=inputs, outputs=grads)
