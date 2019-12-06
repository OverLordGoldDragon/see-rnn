import matplotlib.pyplot as plt
import numpy as np
from .utils import _process_rnn_args


def rnn_histogram(model, layer_name=None, layer_idx=None, layer=None,
                  input_data=None, labels=None, mode='weights', **kwargs):
    """Plots histogram grid of RNN weights/gradients by kernel, gate (if gated),
       and direction (if bidirectional). Also detects NaNs and shows on plots.

    Arguments:
        model: keras.Model/tf.keras.Model.
        layer_idx: int. Index of layer to fetch, via model.layers[layer_idx].
        layer_name: str. Substring of name of layer to be fetched. Returns
               earliest match if multiple found.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `layer_idx` and `layer_name`
        input_data: np.ndarray & supported formats(1). Data w.r.t. which loss is
               to be computed for the gradient. Only for mode=='grads'.
        labels: np.ndarray & supported formats. Labels w.r.t. which loss is
               to be computed for the gradient. Only for mode=='grads'
        mode: str. One of: 'weights', 'grads'. If former, plots layer weights -
              else, plots layer weights grads w.r.t. `input_data` & `labels`.
    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats

    kwargs:
        scale_width:   float. Scale width  of resulting plot by a factor.
        scale_height:  float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plots.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        equate_axes: int: 0, 1, 2. 0 --> auto-managed axes. 1 --> kernel &
                     recurrent subplots' x- & y-axes lims set to common value.
                     2 --> 1, but lims shared for forward & backward plots.
                     Bias plot lims never affected.
        bins: int. Pyplot `hist` kwarg: number of histogram bins per subplot.
    """

    scale_width   = kwargs.get('scale_width',  1)
    scale_height  = kwargs.get('scale_height', 1)
    show_borders  = kwargs.get('show_borders', False)
    show_xy_ticks = kwargs.get('show_xy_ticks',  [True, True])
    equate_axes   = kwargs.get('equate_axes', 1)
    bins          = kwargs.get('bins', 150)

    def _detect_nans(weight_data):
        perc_nans = 100 * np.sum(np.isnan(weight_data)) / len(weight_data)
        if perc_nans == 0:
            return None
        if perc_nans < 0.1:
            num_nans = (perc_nans / 100) * len(weight_data)  # show as quantity
            txt = str(int(num_nans)) + '\nNaNs'
        else:
            txt = "%.1f" % perc_nans + "% \nNaNs"  # show as percent
        return txt

    def _get_axes_extrema(axes):
        axes = np.array(axes)
        is_bidir = len(axes.shape)==3 and axes.shape[0]!=1
        x_new, y_new = [], []

        for direction_idx in range(1 + is_bidir):
            axis = np.array(axes[direction_idx]).T
            for type_idx in range(2):  # 2 = len(kernel_types)
                x_new += [np.max(np.abs([ax.get_xlim() for ax in axis[type_idx]]))]
                y_new += [np.max(np.abs([ax.get_ylim() for ax in axis[type_idx]]))]
        return max(x_new), max(y_new)

    def _set_axes_limits(axes, x_new, y_new):
        axes = np.array(axes)
        is_bidir = len(axes.shape)==3 and axes.shape[0]!=1

        for direction_idx in range(1 + is_bidir):
            axis = np.array(axes[direction_idx]).T
            for type_idx in range(2):
                for gate_idx in range(num_gates):
                    axis[type_idx][gate_idx].set_xlim(-x_new, x_new)
                    axis[type_idx][gate_idx].set_ylim(0,      y_new)

    def _unpack_rnn_info(rnn_info):
        d = rnn_info
        return (d['rnn_type'], d['gate_names'], d['num_gates'],
                d['rnn_dim'],  d['is_bidir'],   d['uses_bias'],
                d['direction_names'])

    data, rnn_info = _process_rnn_args(model, layer_name, layer_idx, layer,
                                       input_data, labels, mode)
    (rnn_type, gate_names, num_gates, rnn_dim,
     is_bidir, uses_bias, direction_names) = _unpack_rnn_info(rnn_info)
    gated_types  = ['LSTM', 'GRU']
    kernel_types = ['KERNEL', 'RECURRENT']

    axes = []
    for direction_idx, direction_name in enumerate(direction_names):
        share = (equate_axes >= 1)
        _, subplot_axes = plt.subplots(num_gates, 2, sharex=share, sharey=share)
        if num_gates == 1:
            subplot_axes = np.expand_dims(subplot_axes, 0)
        axes.append(subplot_axes)
        if direction_name != []:
            plt.suptitle(direction_name + ' LAYER', weight='bold', y=1.05)

        for type_idx, kernel_type in enumerate(kernel_types):
            for gate_idx in range(num_gates):
                ax = subplot_axes[gate_idx][type_idx]
                weight_idx = type_idx + direction_idx*(2 + uses_bias)
                weight_data = data[weight_idx]

                if rnn_type in gated_types:
                    start = gate_idx * rnn_dim
                    end   = start + rnn_dim
                    weight_data = weight_data[:, start:end]
                weight_data = weight_data.flatten()

                ax.hist(weight_data, bins=bins)

                if gate_idx == 0:
                    title = kernel_type + ' GATES' * (rnn_type in gated_types)
                    ax.set_title(title, weight='bold')
                if rnn_type in gated_types:
                    ax.annotate(gate_names[gate_idx], fontsize=12, weight='bold',
                                xy=(0.90, 0.93), xycoords='axes fraction')

                nan_txt = _detect_nans(weight_data)
                if nan_txt is not None:
                    ax.annotate(nan_txt, fontsize=12, weight='bold', color='red',
                                xy=(0.05, 0.63), xycoords='axes fraction')

                if not show_borders:
                    ax.set_frame_on(False)
                if not show_xy_ticks[0]:
                    ax.set_xticks([])
                if not show_xy_ticks[1]:
                    ax.set_yticks([])

        plt.tight_layout()
        plt.gcf().set_size_inches(11*scale_width, 4.5*scale_height)

        if uses_bias:  # does not equate axes
            plt.figure(figsize=(11*scale_width, 4.5*scale_height))
            plt.subplot(num_gates+1, 2, (2*num_gates+1, 2*num_gates+2))
            plt.hist(data[2 + direction_idx*3].flatten(), bins=bins)
            plt.box(on=None)
            plt.annotate('BIAS', fontsize=12, weight='bold',
                         xy=(0.90, 0.93), xycoords='axes fraction')
            plt.tight_layout()
            plt.gcf().set_size_inches(11*scale_width, 4.5*scale_height)
            if not show_xy_ticks[0]:
                plt.gca().set_xticks([])
            if not show_xy_ticks[1]:
                plt.gca().set_yticks([])

    if equate_axes == 2:
        x_new, y_new = _get_axes_extrema(axes)
        _set_axes_limits(axes, x_new, y_new)

    plt.show()
    return axes


def rnn_heatmap(model, layer_name=None, layer_idx=None, layer=None,
                input_data=None, labels=None, mode='weights',
                cmap='bwr', norm=None, normalize=False, **kwargs):
    """Plots histogram grid of RNN weights/gradients by kernel, gate (if gated),
       and direction (if bidirectional). Also detects NaNs and shows on plots.

    Arguments:
        model: keras.Model/tf.keras.Model.
        layer_idx: int. Index of layer to fetch, via model.layers[layer_idx].
        layer_name: str. Substring of name of layer to be fetched. Returns
               earliest match if multiple found.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `layer_idx` and `layer_name`
        input_data: np.ndarray & supported formats(1). Data w.r.t. which loss is
               to be computed for the gradient. Only for mode=='grads'.
        labels: np.ndarray & supported formats. Labels w.r.t. which loss is
               to be computed for the gradient. Only for mode=='grads'.
        mode: str. One of: 'weights', 'grads'. If former, plots layer weights -
              else, plots layer weights grads w.r.t. `input_data` & `labels`.
        cmap: str. Pyplot cmap (colormap) kwarg for the heatmap. If None,
              uses 'bone' cmap (greyscale), suited for >=0 value heatmaps.
        norm: float iter / str ('auto') / None. Normalizes colors to range
              between norm==(vmin, vmax), according to `cmap`. Ex: `cmap`='bwr'
              ('blue white red') -> all values <=vmin and >=vmax will be shown as
              most intense  blue and red, and those exactly in-between are shown
              as white. If 'auto', will normalize across all non-bias plots
              (per kernels, gates, and directions).
        normalize: bool. If True, scales all values to lie between 0 & 1. Works
              well with a greyscale `cmap` (e.g. None).
    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats

    kwargs:
        scale_width:   float. Scale width  of resulting plot by a factor.
        scale_height:  float. Scale height of resulting plot by a factor.
        show_borders:  bool. If True, shows boxes around plots.
        show_colorbar: bool. If True, shows one colorbar next to plot(s).
        show_bias:     bool. If True, includes plot for bias (if layer uses bias)
        gate_sep_width: float. Pyplot kwarg for `linewidth` in marking gate
              separations in gated RNNs (LSTM, GRU) w/ vertical lines.
    """

    scale_width    = kwargs.get('scale_width',  1)
    scale_height   = kwargs.get('scale_height', 1)
    show_borders   = kwargs.get('show_borders',  True)
    show_colorbar  = kwargs.get('show_colorbar', True)
    show_bias      = kwargs.get('show_bias', True)
    gate_sep_width = kwargs.get('gate_sep_width', 1)

    def _unpack_rnn_info(rnn_info):
        d = rnn_info
        return (d['rnn_type'], d['gate_names'], d['num_gates'],
                d['rnn_dim'],  d['is_bidir'],   d['uses_bias'],
                d['direction_names'])

    def _make_common_norm(data):
        idxs = [0, 1] + [3, 4]*(len(data) == 6) + [2, 3]*(len(data) == 4)
        return (min([np.min(data[idx]) for idx in idxs]),
                max([np.max(data[idx]) for idx in idxs]))

    data, rnn_info = _process_rnn_args(model, layer_name, layer_idx, layer,
                                       input_data, labels, mode)
    (rnn_type, gate_names, num_gates, rnn_dim,
     is_bidir, uses_bias, direction_names) = _unpack_rnn_info(rnn_info)

    if cmap is None:
        cmap = plt.cm.bone
    if normalize:
        data -= np.min(data)
        data /= np.max(data)

    (vmin, vmax) = norm if norm else _make_common_norm(data)
    gate_names = '(%s)' % ', '.join(gate_names)

    for direction_idx, direction_name in enumerate(direction_names):
        fig = plt.figure(figsize=(14*scale_width, 5*scale_height))
        axes = []
        if direction_name != []:
            plt.suptitle(direction_name + ' LAYER', weight='bold', y=.98)

        for type_idx, kernel_type in enumerate(['KERNEL', 'RECURRENT']):
            ax = plt.subplot(1, 2, type_idx+1)
            axes.append(ax)
            w_idx = type_idx + direction_idx*(2 + uses_bias)

            # Plot heatmap, gate separators (IF), colorbar (IF)
            img = ax.imshow(data[w_idx], cmap=cmap, interpolation='nearest',
                            aspect='auto', vmin=vmin, vmax=vmax)
            if gate_sep_width != 0:
                lw = gate_sep_width
                [ax.axvline(rnn_dim * gate_idx - .5, linewidth=lw, color='k')
                 for gate_idx in range(1, num_gates)]

            # Styling
            ax.set_title(kernel_type, fontsize=14, weight='bold')
            ax.set_xlabel(gate_names, fontsize=12, weight='bold')
            y_label = ['Channel units', 'Hidden units'][type_idx]
            ax.set_ylabel(y_label, fontsize=12, weight='bold')

            fig.set_size_inches(14*scale_width, 5*scale_height)
            if not show_borders:
                ax.set_frame_on(False)

        if show_colorbar:
            fig.colorbar(img, ax=axes)

        if uses_bias and show_bias:  # does not equate axes
            plt.figure(figsize=(14*scale_width, 1*scale_height))
            plt.subplot(num_gates+1, 2, (2*num_gates + 1, 2*num_gates + 2))
            weights_viz = np.atleast_2d(data[2 + direction_idx*(2 + uses_bias)])

            plt.imshow(weights_viz, cmap=cmap, interpolation='nearest',
                       aspect='auto', vmin=vmin, vmax=vmax)
            if gate_sep_width != 0:
                lw = gate_sep_width
                [plt.axvline(rnn_dim * gate_idx - .5, linewidth=lw, color='k')
                 for gate_idx in range(1, num_gates)]

            # Styling
            plt.box(on=None)
            plt.title('BIAS', fontsize=12, weight='bold')
            plt.gca().get_yaxis().set_ticks([])
