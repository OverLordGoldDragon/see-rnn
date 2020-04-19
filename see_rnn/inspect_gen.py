import os
import numpy as np

from termcolor import colored
from copy import deepcopy
from .utils import _validate_args


TF_KERAS = os.environ.get("TF_KERAS", '0') == '1'
WARN = colored("WARNING:", 'red')
NOTE = colored("NOTE:", 'blue')


if TF_KERAS:
    import tensorflow.keras.backend as K
    print(NOTE, "`sample_weights` & `learning_phase` not yet supported "
          "for `TF_KERAS`, and will be ignored (%s.py)" % __name__)
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

    _validate_args(layer_name, layer_idx, layer)
    layer = get_layer(model, layer_name, layer_idx)
    if TF_KERAS:
        layers_fn = K.function([model.input], [layer.output])
    else:
        layers_fn = K.function([model.input, K.learning_phase()], [layer.output])
    return layers_fn([input_data, learning_phase])[0]


def get_layer_gradients(model, input_data, labels, layer_name=None,
                        layer_idx=None, layer=None, mode='outputs',
                        sample_weights=None, learning_phase=0):
    """Retrieves layer gradients w.r.t. outputs or weights.
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
        mode: str. One of: 'outputs', 'weights'. If former, returns grads
               w.r.t. layer outputs(2) - else, w.r.t. layer trainable weights.
        sample_weights: np.ndarray & supported formats. `sample_weight` kwarg
               to model.fit(), etc., weighting individual sample losses.
        learning_phase: int/bool. If 1, uses model in train model - else,
               in inference mode.

    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats
    (2): "Outputs" are not necessarily activations. If an Activation layer is
         used, returns the gradients of the target layer's PRE-activations
         instead. To then get grads w.r.t. activations, specify the
         Activation layer.
    """

    def _validate_args_(layer_name, layer_idx, layer, mode):
        _validate_args(layer_name, layer_idx, layer)
        if mode not in ['outputs', 'weights']:
            raise Exception("`mode` must be one of: 'outputs', 'weights'")

    _validate_args_(layer_name, layer_idx, layer, mode)
    if layer is None:
        layer = get_layer(model, layer_name, layer_idx)

    grads_fn = _make_grads_fn(model, layer, mode)
    if TF_KERAS:
        grads = grads_fn([input_data, labels])
    else:
        sample_weights = sample_weights or np.ones(len(input_data))
        grads = grads_fn([input_data, sample_weights, labels, 1])

    while isinstance(grads, list) and len(grads)==1:
        grads = grads[0]
    return grads


def get_layer(model, layer_name=None, layer_idx=None):
    """Returns layer by index or name.
    If multiple matches are found, returns earliest.
    """
    _validate_args(layer_name, layer_idx, layer=None)
    if layer_idx is not None:
        return model.layers[layer_idx]

    layer = [layer for layer in model.layers if layer_name in layer.name]
    if len(layer) > 1:
        print(WARN, "multiple matching layer names found; picking earliest")
    elif len(layer) == 0:
        raise Exception("no layers found w/ names matching "
                        + "substring: '%s'" % layer_name)
    return layer[0]


def _make_grads_fn(model, layer, mode='outputs'):
    """Returns gradient computation function w.r.t. layer outputs or weights.
    NOTE: gradients will be clipped if `clipnorm` or `clipvalue` were set.
    """
    if mode not in ['outputs', 'weights']:
        raise Exception("`mode` must be one of: 'outputs', 'weights'")

    params = layer.output if mode=='outputs' else layer.trainable_weights
    grads = model.optimizer.get_gradients(model.total_loss, params)

    if TF_KERAS:
        inputs = [model.inputs[0], model._feed_targets[0]]
    else:
        inputs = [model.inputs[0], model.sample_weights[0],
                  model._feed_targets[0], K.learning_phase()]
    return K.function(inputs=inputs, outputs=grads)


def _detect_nans(data):
    data = np.array(data).ravel()
    perc_nans = 100 * np.sum(np.isnan(data)) / len(data)
    if perc_nans == 0:
        return None
    if perc_nans < 0.1:
        num_nans = (perc_nans / 100) * len(data)  # show as quantity
        txt = str(int(num_nans)) + '\nNaNs'
    else:
        txt = "%.1f" % perc_nans + "% \nNaNs"  # show as percent
    return txt


def weights_norm(model, names, _dict=None, stat_fns=(np.max, np.mean),
                 norm_fn=np.square, omit_weight_names=None, axis=-1, verbose=0):
    """Retrieves model layer weight matrix norms, as specified by `norm_fn`.

    Arguments:
        model: keras.Model/tf.keras.Model.
        names: str list. List of names (can be substring) of layers to fetch
               weights from.
        _dict: dict/None. If None, returns new dict. If dict, appends to it.
        stat_fns: functions list/tuple. Aggregate statistic to compute from
               normed weights.
        norm_fn: function. Norm transform to apply to weights. Ex:
              - np.square (l2 norm)
              - np.abs    (l1 norm)
        omit_weight_names: str list. List of names (can be substring) of weights
               to omit from fetching.
        axis: int. Axis w.r.t. which compute the norm (collapsing all others).
        verbose: int/bool, 0/1. 1/True: print norm stats, enumerated by layer
               indices. If `_dict` is not None, print last computed results.

    Returns:
        stats_all: dict. dict of lists containing layer weight norm statistics.
    """
    def _process_args(model, names, _dict, omit_weight_names):
        def _process_names(names, model):
            if isinstance(names, str):
                names = [names]
            fullnames = []
            for layer in model.layers:
                if any([name in layer.name.lower() for name in names]):
                    fullnames.append(layer.name)
            return fullnames

        names = _process_names(names, model)

        if omit_weight_names is None:
            omit_weight_names = []
        elif isinstance(omit_weight_names, str):
            omit_weight_names = [omit_weight_names]

        if _dict:
            stats_all = deepcopy(_dict)  # do not mutate original dict
        else:
            stats_all = {name: [[]] for name in names}
        return stats_all, names, omit_weight_names


    def _print_stats(stats_all, l_idx, l_name):
        def _unpack_layer_stats(stats_all, l_name):
            l_stats = stats_all[l_name]
            stats_flat = []
            for w_stats in l_stats:
                if isinstance(w_stats, list):
                    [stats_flat.extend(s) for s in w_stats]
                else:
                    stats_flat.append(w_stats)
            return stats_flat

        txt = "{} "
        for w_stats in stats_all[l_name]:
            txt += "{:.4f}, " * len(w_stats)
            txt = txt.rstrip(", ") + " -- "
        txt = txt.rstrip(" -- ")

        stats_flat = _unpack_layer_stats(stats_all, l_name)
        print(txt.format(l_idx, *stats_flat))


    def _get_layer_norm(stats_all, layer, norm_fn, stat_fns, axis=-1):
        def _compute_norm(w, norm_fn, axis=-1):
            axis = axis if axis != -1 else len(w.shape) - 1
            reduction_axes = tuple([ax for ax in range(len(w.shape))
                                    if ax != axis])
            return np.sqrt(np.sum(norm_fn(w), axis=reduction_axes))

        def _append(stats_all, l2_stats, w_idx, l_name):
            if len(stats_all[l_name]) < w_idx + 1:
                stats_all[l_name].append([])
            while len(stats_all[l_name][w_idx]) < len(l2_stats):
                stats_all[l_name][w_idx].append([])

            for stat_idx, stat in enumerate(l2_stats):
                stats_all[l_name][w_idx][stat_idx].append(stat)

        W = layer.get_weights()
        w_names = [w.name for w in layer.weights]
        l_name = layer.name

        for w_idx, (w, w_name) in enumerate(zip(W, w_names)):
            if any([to_omit in w_name for to_omit in omit_weight_names]):
                continue
            l2 = _compute_norm(w, norm_fn, axis)
            l2_stats = [fn(l2) for fn in stat_fns]
            _append(stats_all, l2_stats, w_idx, l_name)
        return stats_all

    stats_all, names, omit_weight_names = _process_args(
        model, names, _dict, omit_weight_names)

    for l_idx, layer in enumerate(model.layers):
        if layer.name in names:
            stats_all = _get_layer_norm(stats_all, layer, norm_fn,
                                        stat_fns, axis)
            if verbose:
                _print_stats(stats_all, l_idx, layer.name)
    return stats_all
