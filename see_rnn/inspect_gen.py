import os
from termcolor import colored

TF_KERAS = os.environ.get("TF_KERAS", 'False') == 'True'
note_str = colored("NOTE: ", 'blue')
warn_str = colored("WARNING: ", 'red')

if TF_KERAS:
    import tensorflow.keras.backend as K
    print(note_str + "`sample_weights` & `learning_phase` not yet supported "
          + "for `TF_KERAS`, and will be ignored (%s.py)" % __name__)
else:
    import keras.backend as K


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


def _get_layer(model, layer_idx=None, layer_name=None):
    """Returns layer by index or name.
    If multiple matches are found, returns earliest.
    """

    if (layer_idx is None and layer_name is None) or \
       (layer_idx is not None and layer_name is not None):
        raise Exception('supply one (and only one) of `layer_idx`, `layer_name`')

    if layer_idx is not None:
        return model.layers[layer_idx]

    layer = [layer for layer in model.layers if layer_name in layer.name]
    if len(layer) > 1:
        print(warn_str + "multiple matching layer names found; "
              + "picking earliest")
    return layer[0]
