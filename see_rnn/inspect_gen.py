import numpy as np

from copy import deepcopy
from .utils import _validate_args
from ._backend import K, TF_KERAS


def get_outputs(model, _id, input_data, layer=None, learning_phase=0,
                as_dict=False):
    """Retrieves layer outputs given input data and layer info.

    Arguments:
        model: keras.Model/tf.keras.Model.
        _id: str/int/(list of str/int). int -> idx; str -> name
            idx: int. Index of layer to fetch, via model.layers[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
            list of str/int -> treat each str element as name, int as idx.
                      Ex: ['gru', 2] gets outputs of first layer with name
                      substring 'gru', then of layer w/ idx 2
        input_data: np.ndarray & supported formats(1). Data w.r.t. which loss is
               to be computed for the gradient. Only for mode=='grads'.
        layer: keras.Layer/tf.keras.Layer. Layer whose outputs to return.
               Overrides `_id`.
        learning_phase: bool. 1: use model in train mode
                              0: use model in inference mode
        as_dict: bool. True:  return output fullname-value pairs in a dict
                       False: return output values as list in order fetched

    Returns:
        Layer output values or name-value pairs (see `as_dict`).

    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats
    """
    def _get_outs_tensors(model, names, idxs, layers):
        if layers is None:
            _id = [x for var in (names, idxs) if var for x in var] or None
            layers = get_layer(model, _id)
        if not isinstance(layers, list):
            layers = [layers]
        return [l.output for l in layers]

    names, idxs, layers, one_requested = _validate_args(_id, layer)
    layer_outs = _get_outs_tensors(model, names, idxs, layers)

    if TF_KERAS:
        outs_fn = K.function([model.input], layer_outs)
    else:
        outs_fn = K.function([model.input, K.learning_phase()], layer_outs)

    outs = outs_fn([input_data, learning_phase])
    if as_dict:
        return {get_full_name(model, i): x for i, x in zip(names or idxs, outs)}
    return outs[0] if (one_requested and len(outs) == 1) else outs


def get_gradients(model, _id, input_data, labels, layer=None, mode='outputs',
                  sample_weights=None, learning_phase=0, as_dict=False):
    """Retrieves layer gradients w.r.t. outputs or weights.
    NOTE: gradients will be clipped if `clipvalue` or `clipnorm` were set.

    Arguments:
        model: keras.Model/tf.keras.Model.
        _id: str/int/(list of str/int). int -> idx; str -> name
            idx: int. Index of layer to fetch, via model.layers[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
            list of str/int -> treat each str element as name, int as idx.
                      Ex: ['gru', 2] gets gradients of first layer with name
                      substring 'gru', then of layer w/ idx 2
        input_data: np.ndarray & supported formats(1). Data w.r.t. which loss is
               to be computed for the gradient.
        labels: np.ndarray & supported formats. Labels w.r.t. which loss is
               to be computed for the gradient.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `idx` and `name`.
        mode: str. One of: 'outputs', 'weights'. If former, returns grads
               w.r.t. layer outputs(2) - else, w.r.t. layer trainable weights.
        sample_weights: np.ndarray & supported formats. `sample_weight` kwarg
               to model.fit(), etc., weighting individual sample losses.
        learning_phase: bool. 1: use model in train mode
                              0: use model in inference mode
        as_dict: bool. True:  return gradient fullname-value pairs in a dict
                       False: return gradient values as list in order fetched

    Returns:
        Layer weight gradients or gradient-value pairs (see `as_dict`).

    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats
    (2): "Outputs" are not necessarily activations. If an Activation layer is
         used, returns the gradients of the target layer's PRE-activations
         instead. To then get grads w.r.t. activations, specify the
         Activation layer.
    """

    def _validate_args_(_id, layer, mode):
        if mode not in ['outputs', 'weights']:
            raise Exception("`mode` must be one of: 'outputs', 'weights'")
        return _validate_args(_id, layer)

    names, idxs, layers, one_requested = _validate_args_(_id, layer, mode)
    _id = [x for var in (names, idxs) if var for x in var] or None

    if layers is None:
        layers = get_layer(model, _id)

    grads_fn = _make_grads_fn(model, layers, mode)
    if TF_KERAS:
        grads = grads_fn([input_data, labels])
    else:
        sample_weights = sample_weights or np.ones(len(input_data))
        grads = grads_fn([input_data, sample_weights, labels, 1])

    if as_dict:
        return {get_full_name(model, i): x for i, x in zip(names or idxs, grads)}
    return grads[0] if (one_requested and len(grads) == 1) else grads


def _make_grads_fn(model, layers, mode='outputs'):
    """Returns gradient computation function w.r.t. layer outputs or weights.
    NOTE: gradients will be clipped if `clipnorm` or `clipvalue` were set.
    """
    def _get_params(layers, mode):
        if not isinstance(layers, list):
            layers = [layers]
        if mode == 'outputs':
            return [l.output for l in layers]
        params = []
        _ = [params.extend(l.trainable_weights) for l in layers]
        return params

    if mode not in ('outputs', 'weights'):
        raise Exception("`mode` must be one of: 'outputs', 'weights'")

    params = _get_params(layers, mode)
    grads = model.optimizer.get_gradients(model.total_loss, params)

    if TF_KERAS:
        inputs = [model.inputs[0], model._feed_targets[0]]
    else:
        inputs = [model.inputs[0], model.sample_weights[0],
                  model._feed_targets[0], K.learning_phase()]
    return K.function(inputs=inputs, outputs=grads)


def get_layer(model, _id):
    """Returns layer by index or name.
    If multiple matches are found, returns earliest.
    """
    names, idxs, _, one_requested = _validate_args(_id)

    layers = []
    if idxs is not None:
        layers = [model.layers[i] for i in idxs]
        if names is None:
            return layers if len(layers) > 1 else layers[0]

    for layer in model.layers:
        for n in names:
            if (n in layer.name):
                layers.append(layer)
                _ = names.pop(names.index(n))  # get at most one per match
                break
            # note that above doesn't avoid duplicates, since `names` doesn't

    if len(layers) == 0:
        raise Exception("no layers found w/ names matching substring(s):",
                        ', '.join(names))
    return layers[0] if one_requested else layers


def get_full_name(model, _id):
    """Given full or partial (substring) layer name, or layer index, or list
    containing either, return complete layer name(s).

    Arguments:
        model: keras.Model / tf.keras.Model.
        _id: str/int/(list of str/int). int -> idx; str -> name
            idx: int. Index of layer to fetch, via model.layers[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
            list of str/int -> treat each str element as name, int as idx.
                      Ex: ['gru', 2] gets full names of first layer w/ name
                      substring 'gru', and of layer w/ idx 2.

    Returns:
        Full name of layer specified by `_id`.
    """
    names, idxs, _, one_requested = _validate_args(_id)

    fullnames = []
    if idxs is not None:
        fullnames = [model.layers[i].name for i in idxs]
        if names is None:
            return fullnames[0] if one_requested else fullnames

    for layer in model.layers:
        for n in names:
            if n in layer.name:
                fullnames.append(layer.name)
                _ = names.pop(names.index(n))  # get at most one per match
                break
            # note that above doesn't avoid duplicates, since `names` doesn't

    if len(fullnames) == 0:
        raise Exception(f"layer w/ identifier '{_id}' not found")
    return fullnames[0] if one_requested else fullnames


def get_weights(model, _id, as_dict=False):
    """Given full or partial (substring) weight name(s), return weight values
    (and corresponding names if as_list=False).

    Arguments:
        model: keras.Model / tf.keras.Model
        _id: str/int/tuple of int/(list of str/int/tuple of int).
                      int/tuple of int -> idx; str -> name
            idx: int/tuple of int. Index of layer weights to fetch.
                       int -> all weights of model.layer[idx]
                       tuple of int -> e.g. (idx, wi0, wi1) -> weights indexed
                       wi0, wi1, of model.layer[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
                       Can specify a weight (full or substring) in format
                       {name/weight_name}.
            list of str/int/tuple of int -> treat each str element as name,
                       int/tuple of int as idx. Ex: ['gru', 2, (3, 1, 2)] gets
                       weights of first layer with name substring 'gru', then all
                       weights of layer w/ idx 2, then weights w/ idxs 1 and 2 of
                       layer w/ idx 3.
        as_dict: bool. True:  return weight fullname-value pairs in a dict
                       False: return weight values as list in order fetched

    Returns:
        Layer weight values or name-value pairs (see `as_dict`).
    """
    def _get_weights_tensors(model, _id):
        def _get_by_idx(model, idx):
            if isinstance(idx, tuple) and len(idx) == 2:
                layer_idx, weight_idxs = idx
            else:
                layer_idx, weight_idxs = idx, None

            layer = model.get_layer(index=layer_idx)
            if weight_idxs is None:
                weight_idxs = list(range(len(layer.weights)))  # get all
            if not isinstance(weight_idxs, (tuple, list)):
                weight_idxs = [weight_idxs]

            return {w.name: w for i, w in enumerate(layer.weights)
                    if i in weight_idxs}

        def _get_by_name(model, name):
            # weight_name == weight part of the full weight name
            if len(name.split('/')) == 2:
                layer_name, weight_name = name.split('/')
            else:
                layer_name, weight_name = name.split('/')[0], None
            layer_name = get_full_name(model, layer_name)
            layer = model.get_layer(name=layer_name)

            if weight_name is not None:
                _weights = {}
                for w in layer.weights:
                    if weight_name in w.name:
                        _weights[w.name] = w
            else:
                _weights = {w.name: w for w in layer.weights}
            if len(_weights) == 0:
                raise Exception(f"weight w/ name '{name}' not found")
            return _weights

        if isinstance(_id, str):
            return _get_by_name(model, _id)
        else:
            return _get_by_idx(model, _id)

    weights = {}
    names, idxs, *_ = _validate_args(_id)
    _ids = [x for var in (names, idxs) if var for x in var] or None

    for _id in _ids:
        weights.update(_get_weights_tensors(model, _id))

    weights = {name: value for name, value in
               zip(weights, K.batch_get_value(list(weights.values())))}
    if as_dict:
        return weights
    weights = list(weights.values())
    return weights[0] if (len(_ids) == 1 and len(_ids) == 1) else weights


def _detect_nans(data):
    data = np.asarray(data).ravel()
    perc_nans = 100 * np.sum(np.isnan(data)) / len(data)
    if perc_nans == 0:
        return None
    if perc_nans < 0.1:
        num_nans = int((perc_nans / 100) * len(data))  # show as quantity
        txt = "{:d}% \nNaNs".format(num_nans)
    else:
        txt = "{:.1f}% \nNaNs".format(perc_nans)  # show as percent
    return txt


def weights_norm(model, _id, _dict=None, stat_fns=(np.max, np.mean),
                 norm_fn=np.square, omit_weight_names=None, axis=-1, verbose=0):
    """Retrieves model layer weight matrix norms, as specified by `norm_fn`.

    Arguments:
        model: keras.Model/tf.keras.Model.
        _id: str/int/tuple of int/(list of str/int/tuple of int).
                      int/tuple of int -> idx; str -> name
            idx: int/tuple of int. Index of layer weights to fetch.
                       int -> all weights of model.layer[idx]
                       tuple of int -> e.g. (idx, wi0, wi1) -> weights indexed
                       wi0, wi1, of model.layer[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
                       Can specify a weight (full or substring) in format
                       {name/weight_name}.
            list of str/int/tuple of int -> treat each str element as name,
                       int/tuple of int as idx.  Ex: ['gru', 2, (3, 1, 2)] gets
                       weights of first layer with name substring 'gru', then all
                       weights of layer w/ idx 2, then weights w/ idxs 1 and 2 of
                       layer w/ idx 3.
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
        stats_all: dict. dict of lists containing layer weight norm statistics,
               structured: stats_all[layer_fullname][weight_index][stat_index].
    """
    def _process_args(model, _id, _dict, omit_weight_names):
        def _get_names(model, _ids):
            _ids_normalized = []
            for _id in _ids:
                _ids_normalized.append(_id[0] if isinstance(_id, tuple) else _id)
            names = get_full_name(model, _ids_normalized)
            if not isinstance(names, list):
                names = [names]
            return names

        _ids = _id if isinstance(_id, list) else [_id]
        names = _get_names(model, _ids)

        if omit_weight_names is None:
            omit_weight_names = []
        elif isinstance(omit_weight_names, str):
            omit_weight_names = [omit_weight_names]

        if _dict:
            stats_all = deepcopy(_dict)  # do not mutate original dict
        else:
            stats_all = {name: [[]] for name in names}
        return _ids, stats_all, names, omit_weight_names

    def _print_stats(stats_all, l_idx, l_name):
        def _unpack_layer_stats(stats_all, l_name):
            l_stats = stats_all[l_name]
            stats_flat = []
            for w_stats in l_stats:
                if isinstance(w_stats, list):
                    _ = [stats_flat.append(stat[-1]) for stat in w_stats]
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

    def _get_layer_norm(stats_all, _id, model, l_name, norm_fn, stat_fns, axis):
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

        weights = get_weights(model, _id, as_dict=True)
        w_names, W = zip(*weights.items())

        for w_idx, (w, w_name) in enumerate(zip(W, w_names)):
            if any([to_omit in w_name for to_omit in omit_weight_names]):
                continue
            l2 = _compute_norm(w, norm_fn, axis)
            l2_stats = [fn(l2) for fn in stat_fns]
            _append(stats_all, l2_stats, w_idx, l_name)
        return stats_all

    _ids, stats_all, names, omit_weight_names = _process_args(model, _id, _dict,
                                                              omit_weight_names)
    for l_idx, layer in enumerate(model.layers):
        if layer.name in names:
            _id = _ids[names.index(layer.name)]
            stats_all = _get_layer_norm(stats_all, _id, model, layer.name,
                                        norm_fn, stat_fns, axis)
            if verbose:
                _print_stats(stats_all, l_idx, layer.name)
    return stats_all
