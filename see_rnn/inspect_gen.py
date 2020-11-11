import tensorflow as tf
import numpy as np

from copy import deepcopy
from .utils import _validate_args, _get_params, _layer_of_output
from ._backend import K, TF_KERAS, Model

if tf.executing_eagerly():
    from tensorflow.python.distribute import parameter_server_strategy
    from tensorflow.python.keras.engine import data_adapter
    from tensorflow.python.keras.mixed_precision.experimental import (
        loss_scale_optimizer as lso)


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
            '*': wildcard -> get outputs of all layers (except input) with
                      'output' attribute. Overrides `layer`.
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

    if _id != '*':
        names, idxs, layers, one_requested = _validate_args(_id, layer)
    else:
        # exclude input layer & non-output layers
        names = [l.name for l in model.layers
                 if (getattr(l, 'output', None) is not None and
                     'Input' not in getattr(l.__class__, '__name__', ''))]
        idxs, layers = None, None
        one_requested = len(_id) == 1

    if not isinstance(input_data, (list, tuple)):
        input_data = [input_data]
    layer_outs = _get_outs_tensors(model, names, idxs, layers)

    if tf.executing_eagerly():
        partial_model = Model(model.inputs, layer_outs)
        outs = partial_model(input_data, training=bool(learning_phase))
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        outs = [o.numpy() for o in outs]
    else:
        lp = K.symbolic_learning_phase() if TF_KERAS else K.learning_phase()
        outs_fn = K.function([*model.inputs, lp], layer_outs)
        outs = outs_fn([*input_data, bool(learning_phase)])

    if as_dict:
        return {get_full_name(model, i): x for i, x in zip(names or idxs, outs)}
    return outs[0] if (one_requested and len(outs) == 1) else outs


def get_gradients(model, _id, input_data, labels, sample_weight=None,
                  learning_phase=0, layer=None, mode='outputs',
                  params=None, as_dict=False):
    """Retrieves layer gradients w.r.t. outputs or weights.

    NOTE: gradients will be clipped if `clipvalue` or `clipnorm` were set.
    NOTE: in Graph execution, repeated calls to `get_gradients` can be expensive
          due to remaking the grads getter function; reuse parts of code to use
          `_make_grads_fn` ONCE to make `grads_fn`, and subsequently feed to
          `grads_fn` for potentially significant speedup.

    Arguments:
        model: keras.Model/tf.keras.Model.
        _id: str/int/(list of str/int). int -> idx; str -> name
        Ignored if `params` is not None.
            idx: int. Index of layer to fetch, via model.layers[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
            list of str/int -> treat each str element as name, int as idx.
                      Ex: ['gru', 2] gets gradients of first layer with name
                      substring 'gru', then of layer w/ idx 2
            '*': wildcard -> get outputs of all layers (except input) with:
                      - 'output'  attribute (mode == 'outputs')
                      - 'weights' attribute (mode == 'weights')
                      Overrides `layer`.
        input_data: np.ndarray & supported formats(1). Data w.r.t. which loss is
               to be computed for the gradient.
        labels: np.ndarray & supported formats. Labels w.r.t. which loss is
               to be computed for the gradient.
        sample_weight: np.ndarray & supported formats. `sample_weight` kwarg
               to model.fit(), etc., weighting individual sample losses.
        learning_phase: bool. 1: use model in train mode
                              0: use model in inference mode
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `idx` and `name`.
        mode: str. One of: 'outputs', 'weights'. If former, returns grads
               w.r.t. layer outputs(2) - else, w.r.t. layer trainable weights.
        params: outputs / weights or list of, or None. If not None, will
               ignore `_id`, `mode`, and `layer`.
        as_dict: bool. True:  return gradient fullname-value pairs in a dict
                       False: return gradient values as list in order fetched

    Returns:
        Layer gradients or gradient-value pairs (see `as_dict`); gradients for
        weights (if mode=='weights') or outputs (if mode=='outputs').

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

    def _get_info(model, _id, layer, mode):
        if _id == '*':
            # get all names for now, do validation in _get_params
            _id = [l.name for l in model.layers]
            names = _id
            idxs, layers = None, None
            one_requested = len(_id) == 1
        else:
            names, idxs, layers, one_requested = _validate_args_(_id, layer, mode)
            _id = [x for var in (names, idxs) if var for x in var] or None
        return _id, names, idxs, layers, one_requested

    if params is not None:
        params = _get_params(model, layer, params, mode, verbose=1)
        one_requested = len(params) == 1
    else:
        verbose = bool(_id != '*')
        _id, _, _, layers, one_requested = _get_info(model, _id, layer, mode)
        if layers is None and params is None:
            layers = get_layer(model, _id)
        params = _get_params(model, layers, mode=mode, verbose=verbose)

    if TF_KERAS and tf.executing_eagerly():
        grads = _get_grads_eager(model, input_data, labels, sample_weight,
                                 learning_phase, params=params)
    else:
        grads_fn = _make_grads_fn(model, params=params)
        if sample_weight is None:
            if TF_KERAS:
                sw = None
            elif isinstance(input_data, list):
                sw = np.ones(len(input_data[0]))
            else:
                sw = np.ones(len(input_data))
            if isinstance(input_data, list):
                sample_weight = []
                for x in input_data:
                    # extend to each input
                    if TF_KERAS:
                        sample_weight.append(sw)
                    else:
                        sample_weight.append(sw)
            else:
                sample_weight = [sw]
        ins = [input_data, labels, sample_weight]
        for i, data in enumerate(ins):
            if not isinstance(data, (list, tuple)):
                ins[i] = [data]
        ins = [x for data in ins for x in data]  # flatten list
        grads = grads_fn([*ins, bool(learning_phase)])

    if as_dict:
        return {p.name: x for p, x in zip(params, grads)}
    return grads[0] if (one_requested and len(grads) == 1) else grads


def _make_grads_fn(model, layers=None, params=None, mode='outputs'):
    """Returns gradient computation function w.r.t. layer outputs or weights.
    NOTE: gradients will be clipped if `clipnorm` or `clipvalue` were set.

    `params` can be layer weights or outputs; cannot supply along `layers`.
    `layers` and `mode` ignored if `params` is not None.

    NOTE: if `sample_weight_mode` in model.compile() is unspecified, or
    `train_on_batch` or `test_on_batch` was never called with a `sample_weight`
    input, then `model._feed_sample_weights == []`, and the resulting function
    will only have three inputs: `(x, y, learning_phase)`.
    """
    if TF_KERAS and tf.executing_eagerly():
        raise Exception("`_make_grads_fn` is unavailable in tf.keras "
                        "Eager execution")

    params = _get_params(model, layers, params, mode)
    grads = model.optimizer.get_gradients(model.total_loss, params)

    inputs = (model.inputs + model._feed_targets + model._feed_sample_weights
              + [K.learning_phase()])
    return K.function(inputs=inputs, outputs=grads)


def _get_grads_eager(model, input_data, labels, sample_weight=None,
                     learning_phase=0, layers=None, params=None, mode='outputs'):
    """Helper method to get gradients in Eager execution; reuses parts of
    Eager train loop code (tf.python.keras.engine.training.train_step) to ensure
    consistency with TensorFlow's training gradient computation.

    `layers` and `mode` ignored if `params` is not None.

    NOTE: do not interrupt, as layer.call of layers are temporarily modified,
    then later reverted IF not interrupted.
    """
    def _process_input_data(x, y, sample_weight):
        iterator = data_adapter.single_batch_iterator(model.distribute_strategy,
                                                      x, y, sample_weight,
                                                      class_weight=None)
        data = next(iterator)
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        return x, y, sample_weight

    def _watch_layer_outputs(layer, tape):
        """Make `layer` watchable by `tape` (tf.GradientTape()). After calling
           this function, layer output gradients can be obtained via:
               grads = tape.gradient(..., layer.output_cache)

           Does nothing if `layer.call_orig` is already defined (ensures no
           unintended composite definitions due to e.g. duplicate `params`).
        """
        def cache_and_watch_output(func):
            def wrap(*args, **kwargs):
                # store the output of `layer.call` internally
                layer.output_cache = func(*args, **kwargs)
                # watch this tensor
                tape.watch(layer.output_cache)
                # return the output to continue with the forward pass
                return layer.output_cache
            return wrap
        if not hasattr(layer, 'call_orig'):
            layer.call_orig = layer.call
            layer.call = cache_and_watch_output(layer.call)

    def _clip_and_scale_grads(strategy, tape, optimizer, loss, params):
        with tape:
            if isinstance(optimizer, lso.LossScaleOptimizer):
                loss = optimizer.get_scaled_loss(loss)

        _params = [(p if isinstance(p, tf.Variable)
                    else _layer_of_output(p).output_cache) for p in params]
        gradients = tape.gradient(loss, _params)

        aggregate_grads_outside_optimizer = (
            optimizer._HAS_AGGREGATE_GRAD and not isinstance(
                strategy.extended,
                parameter_server_strategy.ParameterServerStrategyExtended))

        if aggregate_grads_outside_optimizer:
            gradients = optimizer._aggregate_gradients(zip(gradients, _params))
        if isinstance(optimizer, lso.LossScaleOptimizer):
            gradients = optimizer.get_unscaled_gradients(gradients)

        gradients = optimizer._clip_gradients(gradients)
        return gradients

    if not tf.executing_eagerly():
        raise Exception("`_get_grads_eager` requires TF in Eager execution")

    params = _get_params(model, layers, params, mode)
    x, y, sample_weight = _process_input_data(input_data, labels, sample_weight)

    try:
        with tf.GradientTape() as tape:
            for p in params:
                if isinstance(p, tf.Tensor):
                    _watch_layer_outputs(_layer_of_output(p), tape)
            y_pred = model(x, training=bool(learning_phase))
            loss = model.compiled_loss(y, y_pred, sample_weight,
                                       regularization_losses=model.losses)
        gradients = _clip_and_scale_grads(model.distribute_strategy, tape,
                                          model.optimizer, loss, params)
    finally:
        # ensure layer.call is restored to original
        # not guaranteed; can fail if program is forcibly interrupted
        for p in params:
            if isinstance(p, tf.Tensor):
                layer = _layer_of_output(p)
                if hasattr(layer, 'call_orig'):
                    # may be False if `params` includes duplicates
                    layer.call = layer.call_orig
                    delattr(layer, 'call_orig')
                if hasattr(layer, 'output_cache'):
                    # may be False if `params` includeds duplicates
                    layer.output_cache = []  # ensures no potential memory leak
                    delattr(layer, 'output_cache')

    gradients = K.batch_get_value(gradients)  # evaluate gradient tensors
    return gradients


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


def get_weights(model, _id, omit_names=None, as_tensors=False, as_dict=False):
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
            '*': wildcard -> get weights of all layers with 'weights' attribute.
        omit_names: str/str list. List of names (can be substring) of weights
                                  to omit from fetching.
        as_tensors: bool. True:  return weight tensors.
                          False: return weight values.
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
            _weights = _get_by_name(model, _id)
        else:
            _weights = _get_by_idx(model, _id)

        w_names = list(_weights)
        for w_name in w_names:
            if any(to_omit in w_name for to_omit in omit_names):
                del _weights[w_name]
        return _weights

    if _id != '*':
        names, idxs, *_ = _validate_args(_id)
        _ids = [x for var in (names, idxs) if var for x in var] or None
    else:
        # exclude input layer & non-weight layers
        _ids = [l.name for l in model.layers[1:]
                if getattr(l, 'weights', None) not in (None, [])]
    if not isinstance(omit_names, list):
        omit_names = [omit_names] if omit_names else []

    weights = {}
    for _id in _ids:
        weights.update(_get_weights_tensors(model, _id))

    if not as_tensors:
        weights = {name: value for name, value in
                   zip(weights, K.batch_get_value(list(weights.values())))}
    if as_dict:
        return weights
    weights = list(weights.values())
    return weights[0] if (len(_ids) == 1 and len(weights) == 1) else weights


def detect_nans(data, include_inf=True):
    def _get_txt(perc, data, name):
        txt = ''
        if perc > 0:
            if perc < .1:
                num = int((perc / 100) * data.size)  # show as quantity
                txt = "{:d}% {}".format(num, name)
            else:
                txt = "{:.1f}% {}".format(perc, name)  # show as percent
        return txt

    data = np.asarray(data)
    perc_nans = 100 * np.sum(np.isnan(data)) / data.size

    if include_inf:
        perc_inf = 100 * np.sum(np.isinf(data)) / data.size
    else:
        perc_inf = 0
    if perc_nans == 0 and perc_inf == 0:
        return None

    nan_txt = _get_txt(perc_nans, data, 'NaN')
    inf_txt = _get_txt(perc_inf,  data, 'Inf')

    if nan_txt and inf_txt:
        txt = nan_txt + ', ' + inf_txt
    elif nan_txt:
        txt = nan_txt
    else:
        txt = inf_txt
    return txt


def weights_norm(model, _id, _dict=None, stat_fns=(np.max, np.mean),
                 norm_fn=(np.sqrt, np.square), omit_names=None, axis=-1,
                 verbose=0):
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
        norm_fn: inner function / (outer function, inner function). Norm
                 transform to apply to weights. Ex:
              - (np.sqrt, np.square) (l2 norm)
              - np.abs               (l1 norm)
              Computed as: `outer_fn(sum(inner_fn(x) for x in data))`.
        omit_names: str/str list. List of names (can be substring) of weights
                                  to omit from fetching.
        axis: int. Axis w.r.t. which compute the norm (collapsing all others).
        verbose: int/bool, 0/1. 1/True: print norm stats, enumerated by layer
               indices. If `_dict` is not None, print last computed results.
    Returns:
        stats_all: dict. dict of lists containing layer weight norm statistics,
               structured: stats_all[layer_fullname][weight_index][stat_index].
    Applied example: https://stackoverflow.com/q/61481921/10133797
    """
    def _process_args(model, _id, _dict):
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

        if _dict:
            stats_all = deepcopy(_dict)  # do not mutate original dict
        else:
            stats_all = {name: [[]] for name in names}
        return _ids, stats_all, names

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

    def _get_layer_norm(stats_all, _id, model, l_name, omit_names, norm_fn,
                        stat_fns, axis):
        def _compute_norm(w, norm_fn, axis=-1):
            axis = axis if axis != -1 else len(w.shape) - 1
            reduction_axes = tuple([ax for ax in range(len(w.shape))
                                    if ax != axis])
            if isinstance(norm_fn, (tuple, list)):
                outer_fn, inner_fn = norm_fn
                return outer_fn(np.sum(inner_fn(w), axis=reduction_axes))
            else:
                return np.sum(norm_fn(w), axis=reduction_axes)

        def _append(stats_all, l2_stats, w_idx, l_name):
            if len(stats_all[l_name]) < w_idx + 1:
                stats_all[l_name].append([])
            while len(stats_all[l_name][w_idx]) < len(l2_stats):
                stats_all[l_name][w_idx].append([])

            for stat_idx, stat in enumerate(l2_stats):
                stats_all[l_name][w_idx][stat_idx].append(stat)

        weights = list(get_weights(model, _id, omit_names, as_dict=True).values())

        for w_idx, w in enumerate(weights):
            l2 = _compute_norm(w, norm_fn, axis)
            l2_stats = [fn(l2) for fn in stat_fns]
            _append(stats_all, l2_stats, w_idx, l_name)
        return stats_all

    _ids, stats_all, names = _process_args(model, _id, _dict)
    for l_idx, layer in enumerate(model.layers):
        if layer.name in names:
            _id = _ids[names.index(layer.name)]
            stats_all = _get_layer_norm(stats_all, _id, model, layer.name,
                                        omit_names, norm_fn, stat_fns, axis)
            if verbose:
                _print_stats(stats_all, l_idx, layer.name)
    return stats_all


def get_weight_penalties(model):
    """Get l1, l2, and l1_l2 weight loss penalties from all model layers."""
    wp_dict = {}
    for layer in model.layers:
        layer_penalties = _get_layer_penalties(layer)
        if layer_penalties:
            for p in layer_penalties:
                weight_name, weight_penalty = p
                if not all(wp == 0 for wp in weight_penalty):
                    wp_dict.update({weight_name: weight_penalty})
    return wp_dict


def _get_layer_penalties(layer):
    """Get l1, l2, and l1_l2 weight loss penalties of `layer`."""
    def _rnn_penalties(layer):
        penalties = []
        if hasattr(layer, 'backward_layer'):
            for layer in [layer.forward_layer, layer.backward_layer]:
                penalties += _cell_penalties(layer.cell)
            return penalties
        else:
            return _cell_penalties(layer.cell)

    def _cell_penalties(cell):
        penalties = []
        for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
            _lambda = getattr(cell, weight_type + '_regularizer', None)

            if _lambda is not None:
                weight_name = cell.weights[weight_idx].name
                l1_l2 = (float(getattr(_lambda, 'l1', 0)),
                         float(getattr(_lambda, 'l2', 0)))
                penalties.append([weight_name, l1_l2])
        return penalties

    if hasattr(layer, 'cell') or \
      (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
        return _rnn_penalties(layer)
    elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
        layer = layer.layer

    penalties= []
    for weight_name in ['kernel', 'bias']:
        _lambda = getattr(layer, weight_name + '_regularizer', None)
        if _lambda is not None:
            l1_l2 = (float(_lambda.l1), float(_lambda.l2))
            penalties.append([getattr(layer, weight_name).name, l1_l2])
    return penalties


def weight_loss(model):
    """Compute l1, l2, and l1_l2 weight loss penalties of model layers.
    (e.g. set via `kernel_regularizer=l2(1e-4)`)"""
    weight_penalties = get_weight_penalties(model)

    penalized_weights = []
    ordered_penalties = []
    for w_name, (l1, l2) in weight_penalties.items():
        l_name = w_name.split('/')[0]
        layer = model.get_layer(name=l_name)
        for weight in layer.trainable_weights:
            if weight.name == w_name:
                penalized_weights.append(weight)
                ordered_penalties.append((l1, l2))
    penalized_weights = K.batch_get_value(penalized_weights)

    loss = 0
    for weight, (l1, l2) in zip(penalized_weights, ordered_penalties):
        if l1 != 0:
            loss += l1 * np.sum(np.abs(weight))
        if l2 != 0:
            loss += l2 * np.sum(np.square(weight))
    return loss
