from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from .utils import _process_rnn_args
from .inspect_gen import _detect_nans


def rnn_histogram(model, layer_name=None, layer_idx=None, layer=None,
                  input_data=None, labels=None, mode='weights', equate_axes=1,
                  **kwargs):
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
        equate_axes: int: 0, 1, 2. 0 --> auto-managed axes. 1 --> kernel &
                     recurrent subplots' x- & y-axes lims set to common value.
                     2 --> 1, but lims shared for forward & backward plots.
                     Bias plot lims never affected.
    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats

    kwargs:
        scale_width:   float. Scale width  of resulting plot by a factor.
        scale_height:  float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plots.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        bins: int. Pyplot `hist` kwarg: number of histogram bins per subplot.
    """

    scale_width   = kwargs.get('scale_width',  1)
    scale_height  = kwargs.get('scale_height', 1)
    show_borders  = kwargs.get('show_borders', False)
    show_xy_ticks = kwargs.get('show_xy_ticks',  [True, True])
    bins          = kwargs.get('bins', 150)

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('scale_width', 'scale_height', 'show_borders',
                          'show_xy_ticks', 'bins')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)
        
    def _pretty_hist(data, bins, ax=None):  # hist w/ looping gradient coloring
        if ax is None:
            N, bins, patches = plt.hist(data, bins=bins, density=True)
        else:
            N, bins, patches = ax.hist(data, bins=bins, density=True)

        if len(data) < 1000:
            return  # fewer bins look better monochrome

        bm = bins.max()
        bins_norm = bins / bm

        n_loops = 8  # number of gradient loops
        alpha = 0.94  # graph opacity
        for bin_norm, patch in zip(bins_norm, patches):
            grad = np.sin(np.pi * n_loops * bin_norm) / 15 + .04
            color = (0.121569 + grad*1.2, 0.466667 + grad, 0.705882 + grad,
                     alpha)  # [.121569, .466667, ...] == matplotlib default blue
            patch.set_facecolor(color)

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

    _catch_unknown_kwargs(kwargs)
    data, rnn_info = _process_rnn_args(model, layer_name, layer_idx, layer,
                                       input_data, labels, mode)
    (rnn_type, gate_names, num_gates, rnn_dim,
     is_bidir, uses_bias, direction_names) = _unpack_rnn_info(rnn_info)
    gated_types  = ['LSTM', 'GRU', 'CuDNNLSTM', 'CuDNNGRU']
    kernel_types = ['KERNEL', 'RECURRENT']

    axes = []
    for direction_idx, direction_name in enumerate(direction_names):
        share = (equate_axes >= 1)
        _, subplot_axes = plt.subplots(num_gates, 2, sharex=share, sharey=share)
        if num_gates == 1:
            subplot_axes = np.expand_dims(subplot_axes, 0)
        axes.append(subplot_axes)
        if direction_name != []:
            plt.suptitle(direction_name + ' LAYER', weight='bold', y=1.05,
                         fontsize=13)

        for type_idx, kernel_type in enumerate(kernel_types):
            for gate_idx in range(num_gates):
                ax = subplot_axes[gate_idx][type_idx]
                matrix_idx = type_idx + direction_idx*(2 + uses_bias)
                matrix_data = data[matrix_idx]

                if rnn_type in gated_types:
                    start = gate_idx * rnn_dim
                    end   = start + rnn_dim
                    matrix_data = matrix_data[:, start:end]
                matrix_data = matrix_data.flatten()

                nan_txt = _detect_nans(matrix_data)
                if nan_txt is not None:  # NaNs detected
                    matrix_data[np.isnan(matrix_data)] = 0  # set NaNs to zero
                _pretty_hist(matrix_data, bins=bins, ax=ax)

                if nan_txt is not None:
                    ax.annotate(nan_txt, fontsize=12, weight='bold', color='red',
                                xy=(0.05, 0.63), xycoords='axes fraction')
                if gate_idx == 0:
                    title = kernel_type + ' GATES' * (rnn_type in gated_types)
                    ax.set_title(title, weight='bold')
                if rnn_type in gated_types:
                    ax.annotate(gate_names[gate_idx], fontsize=12, weight='bold',
                                xy=(0.90, 0.93), xycoords='axes fraction')

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

            matrix_data = data[2 + direction_idx*3].flatten()
            nan_txt = _detect_nans(matrix_data)
            if nan_txt is not None:  # NaNs detected
                matrix_data[np.isnan(matrix_data)] = 0  # set NaNs to zero

            _pretty_hist(matrix_data, bins)

            if nan_txt is not None:
                plt.annotate(nan_txt, fontsize=12, weight='bold', color='red',
                             xy=(0.05, 0.63), xycoords='axes fraction')
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
                cmap='bwr', norm=None, normalize=False,
                absolute_value=False, **kwargs):
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
        norm: float list/tuple; str ('auto'); None. Normalizes colors to range
              between norm==(vmin, vmax), according to `cmap`.
                  Ex: `cmap`='bwr'('blue white red') -> all values <=vmin and
                  >=vmax will be shown as most intense  blue and red, and those
                  exactly in-between are shown as white.
              If 'auto', will normalize across all non-bias plots (per kernels,
              gates, and directions), zero-centered.
              If None, Pyplot handles norm.
        normalize: bool. If True, scales all values to lie between 0 & 1. Works
              well with a greyscale `cmap` (e.g. None). Applied after
              `absolute_value`.
        absolute_value: bool. If True, takes absolute value of all data before
              plotting. Works well with a greyscale `cmap` (e.g. None). Applied
              before `normalize`.
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

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('scale_width', 'scale_height', 'show_borders',
                          'show_colorbar', 'show_bias', 'gate_sep_width')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _print_nans(nan_txt, matrix_data, kernel_type, gate_names, rnn_dim):
        if gate_names[0] == '':
            print(kernel_type, end=":")
            nan_txt = nan_txt.replace('\n', '')
            print(colored(nan_txt, 'red'))
        else:
            print('\n' + kernel_type + ":")
            num_gates = len(gate_names)
            gates_data = []
            for gate_idx in range(num_gates):
                start, end = rnn_dim*gate_idx, rnn_dim*(gate_idx + 1)
                gates_data.append(matrix_data[..., start:end])

            for gate_name, gate_data in zip(gate_names, gates_data):
                nan_txt = _detect_nans(gate_data)
                if nan_txt is not None:
                    nan_txt = nan_txt.replace('\n', '')
                    print(gate_name + ':', colored(nan_txt, 'red'))

    def _make_common_norm(data):
        idxs = [0, 1] + [2, 3]*(len(data) == 4) + [3, 4]*(len(data) == 6)
        return np.max([np.max(np.abs(data[idx])) for idx in idxs])

    def _unpack_rnn_info(rnn_info):
        d = rnn_info
        return (d['rnn_type'], d['gate_names'], d['num_gates'],
                d['rnn_dim'],  d['is_bidir'],   d['uses_bias'],
                d['direction_names'])

    _catch_unknown_kwargs(kwargs)
    data, rnn_info = _process_rnn_args(model, layer_name, layer_idx, layer,
                                       input_data, labels, mode, norm)
    (rnn_type, gate_names, num_gates, rnn_dim,
     is_bidir, uses_bias, direction_names) = _unpack_rnn_info(rnn_info)

    if cmap is None:
        cmap = plt.cm.bone
    if absolute_value:
        data = np.abs(data)
    if normalize:
        for idx in range(len(data)):
            data[idx] -= np.min(data[idx])
            if np.max(data[idx]) == 0:
                data[idx] = -data[idx]
            else:
                data[idx] /= np.max(data[idx])

    if norm=='auto':
        vmax = _make_common_norm(data)
        vmin = -vmax
    elif norm is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = norm

    gate_names_str = '(%s)' % ', '.join(gate_names) if gate_names[0]!='' else ''

    for direction_idx, direction_name in enumerate(direction_names):
        fig = plt.figure(figsize=(14*scale_width, 5*scale_height))
        axes = []
        if direction_name != []:
            plt.suptitle(direction_name + ' LAYER', weight='bold', y=.98,
                         fontsize=13)

        for type_idx, kernel_type in enumerate(['KERNEL', 'RECURRENT']):
            ax = plt.subplot(1, 2, type_idx+1)
            axes.append(ax)
            w_idx = type_idx + direction_idx*(2 + uses_bias)
            matrix_data = data[w_idx]

            nan_txt = _detect_nans(matrix_data)
            if nan_txt is not None:
                _print_nans(nan_txt, matrix_data, kernel_type, gate_names, rnn_dim)

            img = ax.imshow(matrix_data, cmap=cmap, interpolation='nearest',
                            aspect='auto', vmin=vmin, vmax=vmax)
            if gate_sep_width != 0:
                lw = gate_sep_width
                [ax.axvline(rnn_dim * gate_idx - .5, linewidth=lw, color='k')
                 for gate_idx in range(1, num_gates)]

            # Styling
            ax.set_title(kernel_type, fontsize=14, weight='bold')
            ax.set_xlabel(gate_names_str, fontsize=12, weight='bold')
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
