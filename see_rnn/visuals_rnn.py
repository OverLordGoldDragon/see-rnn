import matplotlib.pyplot as plt
import numpy as np

from termcolor import colored

from .utils import _process_rnn_args, _kw_from_configs, _save_rnn_fig
from .inspect_gen import detect_nans
from . import scalefig


def rnn_histogram(model, _id, layer=None, input_data=None, labels=None,
                  mode='weights', equate_axes=1, data=None, configs=None,
                  **kwargs):
    """Plots histogram grid of RNN weights/gradients by kernel, gate (if gated),
       and direction (if bidirectional). Also detects NaNs and shows on plots.

    Arguments:
        model: keras.Model/tf.keras.Model.
        _id: str/int. int -> idx; str -> name
            idx: int. Index of layer to fetch, via model.layers[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `idx` and `name`
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
        data: np.ndarray. Pre-fetched data to plot directly - e.g., returned by
              `get_rnn_weights`. Overrides `input_data`, `labels` and `mode`.
              `model` and layer args are still needed to fetch RNN-specific info.
        configs: dict. kwargs to customize various plot schemes:
            'plot':      passed to ax.imshow();    ax  = subplots axis
            'subplot':   passed to plt.subplots()
            'tight':     passed to fig.subplots_adjust(); fig = subplots figure
            'title':     passed to fig.suptitle()
            'annot':     passed to ax.annotate()
            'annot-nan': passed to ax.annotate() for `nan_txt`
            'save':      passed to fig.savefig() if `savepath` is not None.
    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plots.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        bins: int. Pyplot `hist` kwarg: number of histogram bins per subplot.
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (subplots_figs, subplots_axes) of generated subplots. If layer is
            bidirectional, len(subplots_figs) == 2, and latter's is also doubled.
    """

    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', False)
    show_xy_ticks = kwargs.get('show_xy_ticks', [1, 1])
    show_bias     = kwargs.get('show_bias', True)
    bins          = kwargs.get('bins', 150)
    savepath      = kwargs.get('savepath', None)

    def _process_configs(configs, w, h, equate_axes):
        defaults = {
            'plot':    dict(),
            'subplot': dict(sharex=True, sharey=True, dpi=76, figsize=(9, 9)),
            'tight':   dict(),
            'title':   dict(weight='bold', fontsize=12, y=1.05),
            'annot':     dict(fontsize=12, weight='bold',
                              xy=(.90, .93), xycoords='axes fraction'),
            'annot-nan': dict(fontsize=12, weight='bold', color='red',
                              xy=(.05, .63), xycoords='axes fraction'),
            'save': dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)

        if not equate_axes:
            kw['subplot'].update({'sharex': False, 'sharey': False})
        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks', 'bins',
                          'savepath')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _detect_and_zero_nans(matrix_data):
        nan_txt = detect_nans(matrix_data, include_inf=True)
        if nan_txt is not None:  # NaN/Inf detected
            matrix_data[np.isnan(matrix_data)] = 0  # set NaNs to zero
            matrix_data[np.isinf(matrix_data)] = 0  # set Infs to zero
            if ', ' in nan_txt:
                nan_txt = '\n'.join(nan_txt.split(', '))
            else:
                nan_txt = '\n'.join(nan_txt.split(' '))
        return matrix_data, nan_txt

    def _plot_bias(data, axes, direction_idx, bins, d, kw):
        gs = axes[0, 0].get_gridspec()
        for ax in axes[-1, :]:
            ax.remove()
        axbig = fig.add_subplot(gs[-1, :])

        matrix_data = data[2 + direction_idx * 3].ravel()
        matrix_data, nan_txt = _detect_and_zero_nans(matrix_data)
        _pretty_hist(matrix_data, bins, ax=axbig)

        d['gate_names'].append('BIAS')
        _style_axis(axbig, gate_idx=-1, kernel_type=None, nan_txt=nan_txt,
                    show_borders=show_borders, d=d, kw=kw)
        for ax in axes[-2, :].flat:
            # display x labels on bottom row above bias plot as it'll differ
            # per bias row not sharing axes
            ax.tick_params(axis='both', which='both', labelbottom=True)

    def _pretty_hist(matrix_data, bins, ax):
        # hist w/ looping gradient coloring & nan detection
        N, bins, patches = ax.hist(matrix_data, bins=bins, density=True)

        if len(matrix_data) < 1000:
            return  # fewer bins look better monochrome

        bins_norm = bins / bins.max()
        n_loops = 8   # number of gradient loops
        alpha = 0.94  # graph opacity

        for bin_norm, patch in zip(bins_norm, patches):
            grad = np.sin(np.pi * n_loops * bin_norm) / 15 + .04
            color = (0.121569 + grad * 1.2, 0.466667 + grad, 0.705882 + grad,
                     alpha)  # [.121569, .466667, ...] == matplotlib default blue
            patch.set_facecolor(color)

    def _get_axes_extrema(axes):
        axes = np.array(axes)
        is_bidir = len(axes.shape) == 3 and axes.shape[0] != 1
        x_new, y_new = [], []

        for direction_idx in range(1 + is_bidir):
            axis = np.array(axes[direction_idx]).T
            for type_idx in range(2):  # 2 == len(kernel_types)
                x_new += [np.max(np.abs([ax.get_xlim() for ax in axis[type_idx]]))]
                y_new += [np.max(np.abs([ax.get_ylim() for ax in axis[type_idx]]))]
        return max(x_new), max(y_new)

    def _set_axes_limits(axes, x_new, y_new, d):
        axes = np.array(axes)
        is_bidir = len(axes.shape) == 3 and axes.shape[0] != 1

        for direction_idx in range(1 + is_bidir):
            axis = np.array(axes[direction_idx]).T
            for type_idx in range(2):
                for gate_idx in range(d['n_gates']):
                    axis[type_idx][gate_idx].set_xlim(-x_new, x_new)
                    axis[type_idx][gate_idx].set_ylim(0,      y_new)

    def _style_axis(ax, gate_idx, kernel_type, nan_txt, show_borders, d, kw):
        if nan_txt is not None:
            ax.annotate(nan_txt, **kw['annot-nan'])

        is_gated = d['rnn_type'] in gated_types
        if gate_idx == 0:
            title = kernel_type + ' GATES' * is_gated
            ax.set_title(title, weight='bold')
        if is_gated:
            ax.annotate(d['gate_names'][gate_idx], **kw['annot'])

        if not show_borders:
            ax.set_frame_on(False)
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])

    def _get_plot_data(data, direction_idx, type_idx, gate_idx, d):
        matrix_idx = type_idx + direction_idx * (2 + d['uses_bias'])
        matrix_data = data[matrix_idx]

        if d['rnn_type'] in d['gated_types']:
            start = gate_idx * d['rnn_dim']
            end   = start + d['rnn_dim']
            matrix_data = matrix_data[:, start:end]
        return matrix_data.ravel()

    def _make_subplots(show_bias, direction_name, d, kw):
        if not (d['uses_bias'] and show_bias):
            fig, axes = plt.subplots(d['n_gates'], 2, **kw['subplot'])
            axes = np.atleast_2d(axes)
            return fig, axes

        n_rows = 2 * d['n_gates'] + 1
        fig, axes = plt.subplots(n_rows, 2, **kw['subplot'])

        # merge upper axes pairs to increase height ratio w.r.t. bias plot window
        gs = axes[0, 0].get_gridspec()
        for ax in axes[:(n_rows - 1)].flat:
            ax.remove()
        axbigs1, axbigs2 = [], []
        for row in range(n_rows // 2):
            start = 2 * row
            end = start + 2
            axbigs1.append(fig.add_subplot(gs[start:end, 0]))
            axbigs2.append(fig.add_subplot(gs[start:end, 1]))
        axes = np.vstack([np.array([axbigs1, axbigs2]).T, [*axes.flat[-2:]]])

        if direction_name != []:
            fig.suptitle(direction_name + ' LAYER', **kw['title'])
        return fig, axes

    kw = _process_configs(configs, w, h, equate_axes)
    _catch_unknown_kwargs(kwargs)
    data, rnn_info = _process_rnn_args(model, _id, layer, input_data, labels,
                                       mode, data)
    d = rnn_info
    gated_types  = ('LSTM', 'GRU', 'CuDNNLSTM', 'CuDNNGRU')
    kernel_types = ('KERNEL', 'RECURRENT')
    d.update({'gated_types': gated_types, 'kernel_types': kernel_types})

    subplots_axes = []
    subplots_figs = []
    for direction_idx, direction_name in enumerate(d['direction_names']):
        fig, axes = _make_subplots(show_bias, direction_name, d, kw)
        subplots_axes.append(axes)
        subplots_figs.append(fig)

        for type_idx, kernel_type in enumerate(kernel_types):
            for gate_idx in range(d['n_gates']):
                ax = axes[gate_idx][type_idx]
                matrix_data = _get_plot_data(data, direction_idx,
                                             type_idx, gate_idx, d)

                matrix_data, nan_txt = _detect_and_zero_nans(matrix_data)
                _pretty_hist(matrix_data, bins=bins, ax=ax)

                _style_axis(ax, gate_idx, kernel_type, nan_txt, show_borders,
                            d, kw)
        if d['uses_bias'] and show_bias:
            _plot_bias(data, axes, direction_idx, bins, d, kw)

        if kw['tight']:
            fig.subplots_adjust(**kw['tight'])
        else:
            fig.tight_layout()

    if equate_axes == 2:
        x_new, y_new = _get_axes_extrema(subplots_axes)
        _set_axes_limits(subplots_axes, x_new, y_new, d)

    for fig in subplots_figs:
        scalefig(fig)
    plt.show()
    if savepath is not None:
        _save_rnn_fig(subplots_figs, savepath, kw['save'])
    return subplots_figs, subplots_axes


def rnn_heatmap(model, _id, layer=None, input_data=None, labels=None,
                mode='weights', cmap='bwr', norm='auto', data=None,
                configs=None, **kwargs):
    """Plots histogram grid of RNN weights/gradients by kernel, gate (if gated),
       and direction (if bidirectional). Also detects NaNs and prints in console.

    Arguments:
        model: keras.Model/tf.keras.Model.
        _id: str/int. int -> idx; str -> name
            idx: int. Index of layer to fetch, via model.layers[idx].
            name: str. Name of layer (full or substring) to be fetched.
                       Returns earliest match if multiple found.
        layer: keras.Layer/tf.keras.Layer. Layer whose gradients to return.
               Overrides `idx` and `name`
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
              gates, and directions), zero-centered; however, if `absolute_value`
              is also True, sets vmin==vmax==None instead.
              If None, Pyplot handles norm.
        data: np.ndarray. Pre-fetched data to plot directly - e.g., returned by
              `get_rnn_weights`. Overrides `input_data`, `labels` and `mode`.
              `model` and layer args are still needed to fetch RNN-specific info.
        configs: dict. kwargs to customize various plot schemes:
            'plot':      passed to ax.imshow(); ax = subplots axis
            'plot-bias': passed to ax.imshow() for bias
            'subplot':   passed to fig.subplots(); fig = subplots figure
            'tight':     passed to fig.subplots_adjust()
            'title':     passed to fig.suptitle()
            'subtitle':  passed to ax.set_title()  of each subplot
            'xlabel':    passed to ax.set_xlabel() of each non-bias subplot
            'ylabel':    passed to ax.set_ylabel() of each non-bias subplot
            'colorbar':  passed to fig.colorbar()
            'save':      passed to fig.savefig() if `savepath` is not None.
    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool. If True, shows boxes around plots.
        show_colorbar: bool. If True, shows one colorbar next to plot(s).
        show_bias:     bool. If True, includes plot for bias (if layer uses bias)
        gate_sep_width: float. Pyplot kwarg for `linewidth` in marking gate
              separations in gated RNNs (LSTM, GRU) w/ vertical lines.
        normalize: bool. If True, scales all values to lie between 0 & 1. Works
              well with a greyscale `cmap` (e.g. None). Applied after
              `absolute_value`.
        absolute_value: bool. If True, takes absolute value of all data before
              plotting. Works well with a greyscale `cmap` (e.g. None). Applied
              before `normalize`.
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (subplots_figs, subplots_axes) of generated subplots. If layer is
            bidirectional, len(subplots_figs) == 2, and latter's is also doubled.
    """

    w, h           = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders   = kwargs.get('show_borders',  True)
    show_colorbar  = kwargs.get('show_colorbar', True)
    show_bias      = kwargs.get('show_bias', True)
    gate_sep_width = kwargs.get('gate_sep_width', 1)
    normalize      = kwargs.get('normalize', False)
    absolute_value = kwargs.get('absolute_value', False)
    savepath       = kwargs.get('savepath', None)

    def _process_configs(configs, w, h):
        defaults = {
            'plot':      dict(interpolation='nearest'),
            'plot-bias': dict(interpolation='nearest'),
            'subplot':   dict(dpi=76, figsize=(14, 8)),
            'tight':     dict(),
            'title':     dict(weight='bold', fontsize=13, y=.98),
            'subtitle':  dict(weight='bold', fontsize=13),
            'xlabel':    dict(fontsize=12, weight='bold'),
            'ylabel':    dict(fontsize=12, weight='bold'),
            'colorbar':  dict(fraction=.03),
            'save':      dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)

        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_colorbar',
                          'show_bias', 'gate_sep_width', 'normalize',
                          'absolute_value', 'savepath')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _process_data(data, absolute_value, normalize):
        if absolute_value:
            data = np.abs(data)
        if normalize:
            for idx in range(len(data)):
                data[idx] -= np.min(data[idx])
                if np.max(data[idx]) == 0:
                    data[idx] = -data[idx]
                else:
                    data[idx] /= np.max(data[idx])
        return data

    def _get_style_info(cmap, norm, d):
        def _make_common_norm(data):
            idxs = [0, 1] + [2, 3] * (len(data) == 4) + [3, 4] * (len(data) == 6)
            return np.max([np.max(np.abs(data[idx])) for idx in idxs])

        if cmap is None:
            cmap = plt.cm.bone

        if norm == 'auto':
            if absolute_value:
                vmin, vmax = None, None
            else:
                vmax = _make_common_norm(data)
                vmin = -vmax
        elif norm is None:
            vmin, vmax = None, None
        else:
            vmin, vmax = norm

        if d['gate_names'][0] != '':
            gate_names_str = '(%s)' % ', '.join(d['gate_names'])
        else:
            gate_names_str = ''

        return cmap, vmin, vmax, gate_names_str

    def _print_nans(nan_txt, matrix_data, kernel_type, gate_names, rnn_dim):
        if gate_names[0] == '':
            print(kernel_type, end=":")
            print(colored(nan_txt, 'red'))
        else:
            print('\n' + kernel_type + ":")
            n_gates = len(gate_names)
            gates_data = []
            for gate_idx in range(n_gates):
                start, end = rnn_dim*gate_idx, rnn_dim*(gate_idx + 1)
                gates_data.append(matrix_data[..., start:end])

            for gate_name, gate_data in zip(gate_names, gates_data):
                nan_txt = detect_nans(gate_data, include_inf=True)
                if nan_txt is not None:
                    print(gate_name + ':', colored(nan_txt, 'red'))

    def _detect_and_print_nans(matrix_data, kernel_type, d):
        nan_txt = detect_nans(matrix_data, include_inf=True)
        if nan_txt is not None:
            _print_nans(nan_txt, matrix_data, kernel_type,
                        d['gate_names'], d['rnn_dim'])

    def _style_axis(ax, type_idx, kernel_type, show_borders, is_vector, d, kw):
        if is_vector:
            ax.set_xticks([])

        if d['gate_sep_width'] != 0:
            lw = d['gate_sep_width']
            [ax.axvline(d['rnn_dim'] * gate_idx - .5, linewidth=lw, color='k')
             for gate_idx in range(1, d['n_gates'])]

        ax.set_title(kernel_type, **kw['subtitle'])
        ax.set_xlabel(d['gate_names_str'], **kw['xlabel'])
        y_label = ['Channel units', 'Hidden units'][type_idx]
        ax.set_ylabel(y_label, **kw['ylabel'])

        if not show_borders:
            ax.set_frame_on(False)

    def _make_subplots(show_bias, direction_name, d, kw):
        if not (d['uses_bias'] and show_bias):
            fig, axes = plt.subplots(1, 2, **kw['subplot'])
            axes = np.atleast_2d(axes)
            return fig, axes

        fig, axes = plt.subplots(10, 2, **kw['subplot'])

        # merge upper 11 axes to increase height ratio w.r.t. bias plot window
        gs = axes[0, 0].get_gridspec()
        for ax in axes.flat[:18]:
            ax.remove()
        axbig1 = fig.add_subplot(gs[:8, 0])
        axbig2 = fig.add_subplot(gs[:8, 1])
        axes = np.array([[axbig1, axbig2], [*axes.flat[-2:]]])
        axes = np.atleast_2d(axes)

        if direction_name != []:
            fig.suptitle(direction_name + ' LAYER', **kw['title'])
        return fig, axes

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h)
    data, rnn_info = _process_rnn_args(model, _id, layer, input_data, labels,
                                       mode, data, norm)
    d = rnn_info
    d['gate_sep_width'] = gate_sep_width

    data = _process_data(data, absolute_value, normalize)
    cmap, vmin, vmax, d['gate_names_str'] = _get_style_info(cmap, norm, d)
    kernel_types = ['KERNEL', 'RECURRENT']

    subplots_axes = []
    subplots_figs = []
    for direction_idx, direction_name in enumerate(d['direction_names']):
        fig, axes = _make_subplots(show_bias, direction_name, d, kw)
        subplots_axes.append(axes)
        subplots_figs.append(fig)

        ez = lambda *x: enumerate(zip(*x))
        for type_idx, (kernel_type, ax) in ez(kernel_types, axes.flat):
            w_idx = type_idx + direction_idx * (2 + d['uses_bias'])
            matrix_data = data[w_idx]
            _detect_and_print_nans(matrix_data, kernel_type, d)

            is_vector = matrix_data.ndim == 1
            if is_vector:
                matrix_data = np.expand_dims(matrix_data, -1)

            if 'aspect' not in kw['plot']:
                aspect = 20 / len(matrix_data) if is_vector else 'auto'
                kw['plot']['aspect'] = aspect
            img = ax.imshow(matrix_data, cmap=cmap, vmin=vmin, vmax=vmax,
                            **kw['plot'])
            _style_axis(ax, type_idx, kernel_type, show_borders, is_vector,
                        d, kw)

        if is_vector:
            plt.subplots_adjust(right=.7, wspace=-.4)
        if show_colorbar:
            fig.colorbar(img, ax=axes[0, :], **kw['colorbar'])

        if d['uses_bias'] and show_bias:
            for ax in axes[-1, :]:
                ax.remove()
            gs = axes[0, 0].get_gridspec()
            axbig = fig.add_subplot(gs[-1, :])

            data_idx = 2 + direction_idx * (2 + d['uses_bias'])
            weights_viz = np.atleast_2d(data[data_idx])

            if 'aspect' not in kw['plot-bias']:
                dim = weights_viz.shape[1]
                kw['plot-bias']['aspect'] = dim / 100 if dim > 100 else .5
            axbig.imshow(weights_viz, cmap=cmap, vmin=vmin, vmax=vmax,
                         **kw['plot-bias'])
            # Styling
            if d['gate_sep_width'] != 0:
                lw = d['gate_sep_width']
                [axbig.axvline(d['rnn_dim'] * gate_idx - .5, linewidth=lw,
                               color='k') for gate_idx in range(1, d['n_gates'])]
            axbig.set_frame_on(False)
            axbig.set_title('BIAS', **kw['subtitle'])
            axbig.get_yaxis().set_ticks([])

        if kw['tight']:
            fig.subplots_adjust(**kw['tight'])

    for fig in subplots_figs:
        scalefig(fig)
    plt.show()
    if savepath is not None:
        _save_rnn_fig(subplots_figs, savepath, kw['save'])
    return subplots_figs, subplots_axes
