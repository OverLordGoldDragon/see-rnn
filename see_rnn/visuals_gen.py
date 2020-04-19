import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from termcolor import colored


NOTE = colored("NOTE:", 'blue')


def features_0D(data, marker='o', cmap='bwr', color=None,
                configs=None, **kwargs):
    """Plots 0D aligned scatterplots in a standalone graph.

    iter == list/tuple (both work)

    Arguments:
        data: np.ndarray, 2D: (samples, channels).
        marker: str. Pyplot kwarg specifying scatter plot marker shape.
        cmap: str. Pyplot cmap (colormap) kwarg for the heatmap. Overridden
              by `color`!=None.
        color: (float iter) iter / str / str iter / None. Pyplot kwarg,
              specifying marker colors in order of drawing. If str/ float iter,
              draws all curves in one color. Overrides `cmap`. If None,
              automatically colors along equally spaced `cmap` gradient intervals.
              Ex: ['red', 'blue']; [[0., .8, 1.], [.2, .5, 0.]] (RGB)
        configs: dict. kwargs to customize various plot schemes:
            'plot':    passed to plt.scatter()
            'title':   passed to plt.suptitle()

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
        title_mode:    bool/str. If True, shows generic supertitle.
              If str in ('grads', 'outputs'), shows supertitle tailored to
              `data` dim (2D/3D). If other str, shows `title_mode` as supertitle.
              If False, no title is shown.
        show_y_zero: bool. If True, draws y=0.
        ylims: str ('auto'); float list/tuple. Plot y-limits; if 'auto',
               sets both lims to max of abs(`data`) (such that y=0 is centered).
    """

    w, h         = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders = kwargs.get('show_borders', False)
    title_mode   = kwargs.get('title_mode',   'outputs')
    show_y_zero  = kwargs.get('show_y_zero',  True)
    ylims        = kwargs.get('ylims', 'auto')

    def _process_configs(configs, w, h):
        defaults = {
            'plot':    dict(s=15, linewidth=2),
            'title':   dict(weight='bold', fontsize=14),
            }
        configs = configs or {}
        # override defaults, but keep those not in `configs`
        for key in defaults:
            defaults[key].update(configs.get(key, {}))
        kw = defaults.copy()
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'title_mode', 'show_y_zero',
                          'ylims')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title_mode):
        feature = "Context-feature"
        context = "Context-units"

        if title_mode in ['grads', 'outputs']:
            feature = "Gradients" if title_mode=='grads' else "Outputs"
            context = "Timesteps"
        return "(%s vs. %s) vs. Channels" % (feature, context)

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h)
    if data.ndim != 2:
        raise Exception("`data` must be 2D")

    if color is None:
        cmap = cm.get_cmap(cmap)
        cmap_grad = np.linspace(0, 256, len(data[0])).astype('int32')
        color = cmap(cmap_grad)
        color = np.vstack([color] * data.shape[0])
    x = np.ones(data.shape) * np.expand_dims(np.arange(1, len(data) + 1), -1)

    if show_y_zero:
        plt.axhline(0, color='k', linewidth=1)
    plt.scatter(x.ravel(), data.ravel(), marker=marker, color=color,
                **kw['plot'])

    plt.gca().set_xticks(np.arange(1, len(data) + 1), minor=True)
    plt.gca().tick_params(which='minor', length=4)
    if ylims == 'auto':
        ymax = np.max(np.abs(data))
        ymin = -ymax
    else:
        ymin, ymax = ylims
    plt.gca().set_ylim(ymin, ymax)

    if title_mode:
        title = _get_title(data, title_mode)
        plt.title(title, **kw['title'])
    if not show_borders:
        plt.box(None)
    plt.gcf().set_size_inches(12 * w, 4 * h)
    plt.show()


def features_1D(data, n_rows=None, label_channels=True, equate_axes=True,
                max_timesteps=None, subplot_samples=False,
                configs=None, **kwargs):
    """Plots 1D curves in a standalone graph or subplot grid.

    Arguments:
        data: np.ndarray, 2D/3D. Data to plot.
              2D -> standalone graph; 3D -> subplot grid.
              3D: (samples, timesteps, channels)
              2D: (timesteps, channels) or (timesteps, samples)
        n_rows: int/None. Number of rows in subplot grid. If None,
              determines automatically, closest to n_rows == n_cols.
        label_channels: bool. If True, labels subplot grid chronologically.
        equate_axes:    bool. If True, subplots will share x- and y-axes limits.
        max_timesteps:  int/None. Max number of timesteps to show per plot.
              If None, keeps original.
        subplot_samples: bool. If True, generates a subplot per dim 0 of `data`
              instead of dim 2 - i.e. (Channels vs. Timesteps) vs. Samples.
        configs: dict. kwargs to customize various plot schemes:
            'plot':    passed to ax.plot(); ax = subplots axis
            'subplot': passed to plt.subplots()
            'title':   passed to plt.suptitle()
            'tight':   passed to plt.subplots_adjust()
            'annot':   passed to ax.annotate(); ax = subplots axis

    iter == list/tuple (both work)
    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
        title_mode: bool/str. If True, shows generic supertitle.
              If str in ('grads', 'outputs'), shows supertitle tailored to
              `data` dim (2D/3D). If other str, shows `title_mode` as supertitle.
              If False, no title is shown.
        show_y_zero: bool. If True, draw y=0 for each plot.
        tight: bool. If True, plots compactly by removing subplot padding.
        borderwidth: float / None. Width of subplot borders.
        color: (float iter) iter / str / str iter / None. Pyplot kwarg,
              specifying curve colors in order of drawing. If str/ float iter,
              draws all curves in one color. If None, default coloring is used.
              Ex: ['red', 'blue']; [[0., .8, 1.], [.2, .5, 0.]] (RGB)
    """

    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', True)
    show_xy_ticks = kwargs.get('show_xy_ticks', [1, 1])
    title_mode    = kwargs.get('title_mode', 'outputs')
    show_y_zero   = kwargs.get('show_y_zero', False)
    tight         = kwargs.get('tight', False)
    borderwidth   = kwargs.get('borderwidth', None)
    color         = kwargs.get('color', None)

    def _process_configs(configs, w, h, tight, equate_axes):
        defaults = {
            'plot':    dict(),
            'subplot': dict(sharex=True, sharey=True, dpi=76, figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=14, y=.93 + .12 * tight),
            'tight':   dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0),
            'annot':   dict(weight='bold', fontsize=16, color='g',
                            xy=(.03, .9), xycoords='axes fraction')
            }
        configs = configs or {}
        # override defaults, but keep those not in `configs`
        for key in defaults:
            defaults[key].update(configs.get(key, {}))
        kw = defaults.copy()

        if not equate_axes:
            kw['subplot'].update({'sharex': False, 'sharey': False})
        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks',
                          'title_mode', 'show_y_zero', 'tight', 'borderwidth')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title_mode, subplot_samples):
        feature = "Context-feature"
        context = "Context-units"
        if title_mode in ['grads', 'outputs']:
            feature = "Gradients" if title_mode == 'grads' else "Outputs"
            context = "Timesteps"

        subplot_mode_3d = "vs. Samples) vs. Channels"
        subplot_mode_2d = "vs. Channels"
        if subplot_samples:
            subplot_mode_3d = "vs. Channels) vs. Samples"
            subplot_mode_2d = "vs. Samples"

        if data.ndim == 3 and data.shape[-1] != 1:
            return "((%s vs. %s) %s" % (feature, context, subplot_mode_3d)
        else:
            return "(%s vs. %s) %s" % (feature, context, subplot_mode_2d)

    def _get_feature_outputs(data, ax_idx):
        # ax_idx -> samples if subplot_samples==True, else -> channels
        feature_outputs = []
        for sample in data:
            feature_outputs.append(sample[:max_timesteps, ax_idx - 1])
        return feature_outputs

    def _get_style_info(data, n_rows, color):
        n_subplots = data.shape[-1]
        n_rows, n_cols = _get_nrows_and_ncols(n_rows, n_subplots)

        if color is None:
            n_colors = len(data) if data.ndim == 3 else 1
            color = [None] * n_colors
        return n_rows, n_cols, color

    def _process_data(data):
        if data.ndim not in (2, 3):
            raise Exception("`data` must be 2D or 3D")
        if data.ndim == 2 and subplot_samples:
            print(NOTE, "`subplot_samples` w/ 2D `data` will only change "
                  "title shown, and assumes `data` dims (timesteps, samples)")
        elif subplot_samples:
            data = data.T  # -> (channels, timesteps, samples)
        if data.ndim == 2:
            data = data.T  # -> (samples, timesteps) | (channels, timesteps)
            data = np.expand_dims(data, -1)
        return data

    def _style_axis(ax, kw, show_borders, show_xy_ticks, xmax):
        ax.axis(xmin=0, xmax=xmax)
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])
        if label_channels:
            ax.annotate(str(ax_idx), **kw['annot'])
        if not show_borders:
            ax.set_frame_on(False)

    kw = _process_configs(configs, w, h, tight, equate_axes)
    _catch_unknown_kwargs(kwargs)
    data = _process_data(data)
    n_rows, n_cols, color = _get_style_info(data, n_rows, color)

    fig, axes = plt.subplots(n_rows, n_cols, **kw['subplot'])
    axes = np.asarray(axes)

    if title_mode:
        title = _get_title(data, title_mode, subplot_samples)
        plt.suptitle(title, **kw['title'])

    for ax_idx, ax in enumerate(axes.flat):
        if show_y_zero:
            ax.axhline(0, color='red')

        feature_outputs = _get_feature_outputs(data, ax_idx)
        for idx, out in enumerate(feature_outputs):
            ax.plot(out, color=color[idx], **kw['plot'])

        _style_axis(ax, kw, show_borders, show_xy_ticks,
                    xmax=len(feature_outputs[0]))

    if tight:
        plt.subplots_adjust(**kw['tight'])
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]
    plt.show()


def features_2D(data, n_rows=None, norm=None, cmap='bwr', reflect_half=False,
                timesteps_xaxis=True, max_timesteps=None,
                configs=None, **kwargs):
    """Plots 2D heatmaps in a standalone graph or subplot grid.

    iter == list/tuple (both work)

    Arguments:
        data: np.ndarray, 2D/3D. Data to plot.
              2D -> standalone graph; 3D -> subplot grid.
              3D: (samples, timesteps, channels)
              2D: (timesteps, channels)
        n_rows: int/None. Number of rows in subplot grid. If None,
              determines automatically, closest to n_rows == n_cols.
        norm: float iter/'auto'/None. Normalizes colors to range between
              norm==(vmin, vmax), according to `cmap`. If 'auto', sets
              norm = (-v, v), where v = max(abs(data)).
                  Ex: `cmap`='bwr' ('blue white red') -> all
                  values <=vmin and >=vmax will be shown as most intense blue
                  and red, and those exactly in-between are shown as white.
        cmap: str. Pyplot cmap (colormap) kwarg for the heatmap.
        reflect_half: bool. If True, second half of channels dim will be
              flipped about the timesteps dim.
        timesteps_xaxis: bool. If True, the timesteps dim (`data` dim 1)
              if plotted along the x-axis.
        max_timesteps:  int/None. Max number of timesteps to show per plot.
              If None, keeps original.
        configs: dict. kwargs to customize various plot schemes:
            'plot':     passed to plt.hist()
            'subplot':  passed to plt.subplots()
            'title':    passed to plt.suptitle()
            'tight':    passed to plt.subplots_adjust()
            'colorbar': passed to fig.colorbar(); fig = subplots figure

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_colorbar: bool. If True, shows one colorbar next to plot(s).
        title_mode:    bool/str. If True, shows generic supertitle.
              If str in ('grads', 'outputs'), shows supertitle tailored to
              `data` dim (2D/3D). If other str, shows `title_mode` as supertitle.
              If False, no title is shown.
        tight: bool. If True, plots compactly by removing subplot padding.
        channel_axis: int, 0 or -1. `data` axis holding channels/features.
              -1 --> (samples,  timesteps, channels)
              0  --> (channels, timesteps, samples)
        borderwidth: float / None. Width of subplot borders.
    """

    w, h           = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders   = kwargs.get('show_borders',  True)
    show_xy_ticks  = kwargs.get('show_xy_ticks', [1, 1])
    show_colorbar  = kwargs.get('show_colorbar', False)
    title_mode     = kwargs.get('title_mode',    'outputs')
    tight          = kwargs.get('tight', False)
    channel_axis   = kwargs.get('channel_axis', -1)
    borderwidth    = kwargs.get('borderwidth', None)

    def _process_configs(configs, w, h, tight):
        defaults = {
            'plot':    dict(),
            'subplot': dict(sharex=True, sharey=True, dpi=76, figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=14, y=.93 + .12 * tight),
            'tight':   dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0),
            'colorbar': dict(),
            }
        configs = configs or {}
        # override defaults, but keep those not in `configs`
        for key in defaults:
            defaults[key].update(configs.get(key, {}))
        kw = defaults.copy()

        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks',
                          'show_colorbar', 'title_mode', 'tight',
                          'channel_axis', 'borderwidth')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title_mode, timesteps_xaxis, vmin, vmax):
        feature = "Context-feature"
        context = "Context-units"
        context_order = "(%s vs. Channels)" % context
        extra_dim = ""

        if title_mode in ['grads', 'outputs']:
            feature = "Gradients" if title_mode == 'grads' else "Outputs"
            context = "Timesteps"
        if timesteps_xaxis:
            context_order = "(Channels vs. %s)" % context
        if data.ndim == 3:
            extra_dim = ") vs. Samples"
            context_order = "(" + context_order

        norm_txt = "(%.4G, %.4G)" % (vmin, vmax) if (
            vmin is not None) else "auto"
        return "{} vs. {}{} -- norm={}".format(context_order, feature,
                                               extra_dim, norm_txt)

    def _get_style_info(data, n_rows, norm):
        if norm == 'auto':
            v = np.abs(data).max()
            vmin, vmax = (-v, v)
        else:
            vmin, vmax = norm or (None, None)
        n_subplots = len(data) if data.ndim == 3 else 1
        n_rows, n_cols = _get_nrows_and_ncols(n_rows, n_subplots)

        return n_rows, n_cols, vmin, vmax

    def _process_data(data, max_timesteps, reflect_half,
                      timesteps_xaxis, channel_axis):
        if data.ndim not in (2, 3):
            raise Exception("`data` must be 2D or 3D")

        if max_timesteps is not None:
            data = data[..., :max_timesteps, :]

        if reflect_half:
            data = data.copy()  # prevent passed array from changing
            half_chs = data.shape[-1] // 2
            data[..., half_chs:] = np.flip(data[..., half_chs:], axis=0)

        if timesteps_xaxis:
            if data.ndim != 3:
                data = np.expand_dims(data, 0)
            data = np.transpose(data, (0, 2, 1))
        return data

    def _style_axis(ax, kw, show_borders, show_xy_ticks):
        ax.axis('tight')
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])
        if not show_borders:
            ax.set_frame_on(False)

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h, tight)
    data = _process_data(data, max_timesteps, reflect_half,
                         timesteps_xaxis, channel_axis)
    n_rows, n_cols, vmin, vmax = _get_style_info(data, n_rows, norm)

    fig, axes = plt.subplots(n_rows, n_cols, **kw['subplot'])
    axes = np.asarray(axes)

    if title_mode:
        title = _get_title(data, title_mode, timesteps_xaxis, vmin, vmax)
        plt.suptitle(title, **kw['title'])

    for ax_idx, ax in enumerate(axes.flat):
        img = ax.imshow(data[ax_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                        **kw['plot'])
        _style_axis(ax, kw, show_borders, show_xy_ticks)

    if show_colorbar:
        fig.colorbar(img, ax=axes.ravel().tolist(), **kw['colorbar'])
    if tight:
        plt.subplots_adjust(**kw['tight'])
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]
    plt.show()


def _get_nrows_and_ncols(n_rows, n_subplots):
    if n_rows is None:
        n_rows = int(np.sqrt(n_subplots))
    n_cols = max(int(n_subplots / n_rows), 1)  # ensure n_cols != 0
    n_rows = int(n_subplots / n_cols)

    while not ((n_subplots / n_cols).is_integer() and
               (n_subplots / n_rows).is_integer()):
        n_cols -= 1
        n_rows = int(n_subplots / n_cols)
    return n_rows, n_cols


def features_hist(data, n_rows='vertical', bins=100, xlims=None, tight=True,
                  configs=None, **kwargs):
    """Plots histograms in a subplot grid.

    Arguments:
        data: np.ndarray, n-dim. Data to plot.
              (samples, ..., channels), or (samples,), or (channels,)
        n_rows: int/'vertical'/None. Number of rows in subplot grid. If None,
              determines automatically, closest to n_rows == n_cols.
              If 'vertical', sets n_rows = len(data).
        bins: int. Pyplot kwarg to plt.hist(), # of bins.
        xlims: float tuple. x limits to apply to all subplots.
        tight: bool. If True, plt.subplots_adjust (spacing) according to
              configs['tight'], defaulted to minimize intermediate padding.
        configs: dict. kwargs to customize various plot schemes:
            'plot':    passed to plt.hist()
            'subplot': passed to plt.subplots()
            'title':   passed to plt.suptitle()
            'tight':   passed to plt.subplots_adjust()
            'annot':   passed to ax.annotate(); ax = subplots axis

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
        title: str/None. If not None, show `title` as plt.suptitle.
        borderwidth: float / None. Width of subplot borders.
        annotations: str list/'auto'/None.
            'auto': annotate each subplot with its index.
             list of str: annotate by indexing into the list.
                          If len(list) < len(data), won't annotate remainder.
             None: don't annotate.
    """
    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', True)
    show_xy_ticks = kwargs.get('show_xy_ticks', [1, 1])
    title         = kwargs.get('title', None)
    borderwidth   = kwargs.get('borderwidth', None)
    annotations   = kwargs.get('annotations', 'auto')

    def _process_configs(configs, w, h, tight):
        defaults = {
            'plot':    dict(),
            'subplot': dict(sharex='col', sharey=True, dpi=76, figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=14, y=.93 + .12 * tight),
            'tight':   dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0),
            'annot':   dict(weight='bold', fontsize=16, xy=(.03, .7),
                            xycoords='axes fraction', color='g'),
            }
        configs = configs or {}
        # override defaults, but keep those not in `configs`
        for key in defaults:
            defaults[key].update(configs.get(key, {}))
        kw = defaults.copy()

        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks', 'title',
                          'borderwidth', 'annotations')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_style_info(data, n_rows, annotations):
        n_subplots = len(data)
        if n_rows == 'vertical':
            n_rows, n_cols = n_subplots, 1
        else:
            n_rows, n_cols = _get_nrows_and_ncols(n_rows, n_subplots)

        if annotations == 'auto':
            annotations = list(map(str, range(len(data))))
        return n_rows, n_cols, annotations

    def _style_axis(ax, kw, show_borders, show_xy_ticks, annotations):
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])
        if annotations:
            ax.annotate(annotations.pop(0), **kw['annot'])
        if not show_borders:
            ax.set_frame_on(False)

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h, tight)
    n_rows, n_cols, annotations = _get_style_info(data, n_rows, annotations)

    fig, axes = plt.subplots(n_rows, n_cols, **kw['subplot'])
    axes = np.asarray(axes)
    if title is not None:
        plt.suptitle(title, **kw['title'])

    for ax_idx, ax in enumerate(axes.flat):
        ax.hist(data[ax_idx].ravel(), bins=bins, **kw['plot'])
        _style_axis(ax, kw, show_borders, show_xy_ticks, annotations)

    if xlims is not None:
        ax.set_xlim(*xlims)

    if tight:
        plt.subplots_adjust(**kw['tight'])
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]
    plt.show()


def features_hist_v2(data, colnames=None, bins=100, xlims=None, ylim=None,
                     tight=True, side_annot=None, configs=None, **kwargs):
    """Plots histograms in a subplot grid.

    Arguments:
        data: np.ndarray, n-dim. Data to plot.
              (rows, cols, subdata...)
              Plots dim0 along rows, dim1 along cols, subdata within
              individual subplots (iteratively), flattens remaining dims.
        colnames: str list. Column titles, displayed on top subplot boxes.
        bins: int. Pyplot kwarg to plt.hist(), # of bins.
        xlims: float tuple. x limits to apply to all subplots.
        ylim: float. Top y limit of all subplots.
        tight: bool. If True, plt.subplots_adjust (spacing) according to
              configs['tight'], defaulted to minimize intermediate padding.
        side_annot: str. Text to display to the right side of rightmost subplot
              boxes, enumerated by row number ({side_annot}{row})
        configs: dict. kwargs to customize various plot schemes:
            'plot':       passed to plt.hist()
            'subplot':    passed to plt.subplots()
            'title':      passed to plt.suptitle()
            'tight':      passed to plt.subplots_adjust()
            'colnames':   passed to ax.set_title(); ax = subplots axis
            'side_annot': passed to ax.annotate();  ax = subplots axis

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
        title: str/None. If not None, show `title` as plt.suptitle.
        borderwidth: float / None. Width of subplot borders.
    """
    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', True)
    show_xy_ticks = kwargs.get('show_xy_ticks', [1, 1])
    title         = kwargs.get('title', None)
    borderwidth   = kwargs.get('borderwidth', None)

    def _process_configs(configs, w, h):
        defaults = {
            'plot':    dict(),
            'subplot': dict(sharex='col', sharey=True, dpi=76,
                            figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=15, y=1.06),
            'tight':   dict(left=0, right=1, bottom=0, top=1,
                            wspace=.05, hspace=.05),
            'colnames':   dict(weight='bold', fontsize=14),
            'side_annot': dict(weight='bold', fontsize=14,
                               xy=(1.02, .5), xycoords='axes fraction'),
            }
        configs = configs or {}
        # override defaults, but keep those not in `configs`
        for key in defaults:
            defaults[key].update(configs.get(key, {}))
        kw = defaults.copy()

        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks', 'title',
                          'borderwidth')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _style_axis(ax, kw, show_borders, show_xy_ticks, xlims):
        if row == 0 and colnames is not None:
            ax.set_title(f"{colnames[col]}", **kw['colnames'])
        if side_annot is not None and col == n_cols - 1:
            ax.annotate(f"{side_annot}{row}", **kw['side_annot'])
        if not show_borders:
            ax.set_frame_on(False)
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])
        if xlims is not None:
            ax.set_xlim(*xlims)

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h)
    fig, axes = plt.subplots(len(data), len(data[0]), **kw['subplot'])
    if title is not None:
        plt.suptitle(title, **kw['title'])
    n_rows, n_cols = data.shape[:2]

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            for subdata in data[row][col]:
                ax.hist(subdata.ravel(), bins=bins, **kw['plot'])

            _style_axis(ax, kw, show_borders, show_xy_ticks, xlims)

    if ylim is not None:
        ax.set_ylim(0, ylim)
    if tight:
        plt.subplots_adjust(**kw['tight'])
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]
    plt.show()
