import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from ._backend import NOTE
from .utils import _kw_from_configs, clipnums
from . import scalefig


def features_0D(data, marker='o', cmap='bwr', color=None, configs=None, **kwargs):
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
            'plot':  passed to plt.scatter()
            'title': passed to plt.suptitle()
            'save':  passed to fig.savefig() if `savepath` is not None.

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders: bool.  If True, shows boxes around plot(s).
        title: bool/str. If True, shows generic supertitle.
              If str in {'grads', 'outputs', 'generic'}, shows supertitle
              tailored to `data` dim (0D/1D). If other str, shows `title`
              as supertitle. If False, no title is shown.
        show_y_zero: bool. If True, draws y=0.
        ylims: str ('auto'); float list/tuple. Plot y-limits; if 'auto',
               sets both lims to max of abs(`data`) (such that y=0 is centered).
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (figs, axes) of generated plots.
    """

    w, h         = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders = kwargs.get('show_borders', False)
    title        = kwargs.get('title', 'outputs')
    show_y_zero  = kwargs.get('show_y_zero', True)
    ylims        = kwargs.get('ylims', 'auto')
    savepath     = kwargs.get('savepath', None)

    def _process_configs(configs, w, h):
        defaults = {
            'plot':  dict(s=15, linewidth=2),
            'title': dict(weight='bold', fontsize=14),
            'save':  dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'title', 'show_y_zero',
                          'ylims', 'savepath')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title):
        if title not in {'grads', 'outputs', 'generic'}:
            return title

        if title == 'generic':
            feature = "Context-feature"
            context = "Context-units"
        else:
            feature = "Gradients" if title =='grads' else "Outputs"
            context = "Timesteps"
        return "(%s vs. %s) vs. Channels" % (feature, context)

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h)
    if data.ndim != 2:
        raise Exception("`data` must be 2D or 3D (got ndim=%s)" % data.ndim)

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

    if title:
        title = _get_title(data, title)
        plt.title(title, **kw['title'])
    if not show_borders:
        plt.box(None)

    fig, axes = plt.gcf(), plt.gca()
    fig.set_size_inches(12 * w, 4 * h)
    scalefig(fig)
    plt.show()

    if savepath is not None:
        fig.savefig(savepath, **kw['save'])
    return fig, axes


def features_1D(data, n_rows=None, annotations='auto', share_xy=(1, 1),
                max_timesteps=None, subplot_samples=False, configs=None,
                **kwargs):
    """Plots 1D curves in a standalone graph or subplot grid.

    Arguments:
        data: np.ndarray, 2D/3D. Data to plot.
              2D -> standalone graph; 3D -> subplot grid.
              3D: (samples, timesteps, channels)
              2D: (timesteps, channels) or (timesteps, samples)
        n_rows: int/None. Number of rows in subplot grid. If None,
              determines automatically, closest to n_rows == n_cols.
        annotations: str list/'auto'/None.
            'auto': annotate each subplot with its index.
             list of str: annotate by indexing into the list.
                          If len(list) < len(data), won't annotate remainder.
             None: don't annotate.
        share_xy: bool/str/(tuple/list of bool/str) for (sharex, sharey)
              kwargs passed to plt.subplots(). If bool/str, will use same
              value for sharex & sharey.
              - sharex: bool/str. True  -> subplots will share x-axes limits.
                                  'col' -> limits shared along columns.
                                  'row' -> limits shared along rows.
                                  Overrides 'sharex' in `configs['subplot']`.
              - sharey: bool/str. `sharex`, but for y-axis.
        max_timesteps:  int/None. Max number of timesteps to show per plot.
              If None, keeps original.
        subplot_samples: bool. If True, generates a subplot per dim 0 of `data`
              instead of dim 2 - i.e. (Channels vs. Timesteps) vs. Samples.
        configs: dict. kwargs to customize various plot schemes:
            'plot':    passed to ax.plot();      ax  = subplots axis
            'subplot': passed to fig.subplots(); fig = subplots figure
            'title':   passed to fig.suptitle()
            'tight':   passed to fig.subplots_adjust()
            'annot':   passed to ax.annotate()
            'save':    passed to fig.savefig() if `savepath` is not None.

    iter == list/tuple (both work)
    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
        title: bool/str. If True, shows generic supertitle.
              If str in {'grads', 'outputs', 'generic'}, shows supertitle
              tailored to `data` dim (1D/2D). If other str, shows `title`
              as supertitle. If False, no title is shown.
        show_y_zero: bool. If True, draw y=0 for each plot.
        tight: bool. If True, plots compactly by removing subplot padding.
        borderwidth: float / None. Width of subplot borders.
        color: (float iter) iter / str / str iter / None. Pyplot kwarg,
              specifying curve colors in order of drawing. If str/ float iter,
              draws all curves in one color. If None, default coloring is used.
              Ex: ['red', 'blue']; [[0., .8, 1.], [.2, .5, 0.]] (RGB)
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (figs, axes) of generated plots.
    """

    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', True)
    show_xy_ticks = kwargs.get('show_xy_ticks', (1, 1))
    title         = kwargs.get('title', 'outputs')
    show_y_zero   = kwargs.get('show_y_zero', False)
    tight         = kwargs.get('tight', False)
    borderwidth   = kwargs.get('borderwidth', None)
    color         = kwargs.get('color', None)
    savepath      = kwargs.get('savepath', None)

    def _process_configs(configs, w, h, tight, share_xy):
        def _set_share_xy(kw, share_xy):
            if isinstance(share_xy, (list, tuple)):
                sharex, sharey = share_xy
            else:
                sharex = sharey = share_xy
            if isinstance(sharex, int):
                sharex = bool(sharex)
            if isinstance(sharey, int):
                sharey = bool(sharey)
            kw['subplot']['sharex'] = sharex
            kw['subplot']['sharey'] = sharey

        defaults = {
            'plot':    dict(),
            'subplot': dict(dpi=76, figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=14, y=.93 + .12 * tight),
            'tight':   dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0),
            'annot':   dict(weight='bold', fontsize=16, color='g',
                            xy=(.03, .9), xycoords='axes fraction'),
            'save':    dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)

        _set_share_xy(kw, share_xy)
        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks',
                          'title', 'show_y_zero', 'tight', 'borderwidth',
                          'color', 'savepath')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title, subplot_samples):
        if title not in {'grads', 'outputs', 'generic'}:
            return title

        if title == 'generic':
            feature = "Context-feature"
            context = "Context-units"
        else:
            feature = "Gradients" if title == 'grads' else "Outputs"
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

    def _get_style_info(data, n_rows, color, annotations):
        n_subplots = data.shape[-1]
        n_rows, n_cols = _get_nrows_and_ncols(n_rows, n_subplots)

        if color is None:
            n_colors = len(data) if data.ndim == 3 else 1
            color = [None] * n_colors
        if annotations == 'auto':
            annotations = list(map(str, range(n_subplots)))
        elif annotations is not None:
            # ensure external list is unaffected
            annotations = annotations.copy()
        return n_rows, n_cols, color, annotations

    def _process_data(data):
        if data.ndim not in (2, 3):
            raise Exception("`data` must be 2D or 3D (got ndim=%s)" % data.ndim)
        if data.ndim == 2 and subplot_samples:
            print(NOTE, "`subplot_samples` w/ 2D `data` will only change "
                  "title shown, and assumes `data` dims (timesteps, samples)")
        elif subplot_samples:
            data = data.T  # -> (channels, timesteps, samples)
        if data.ndim == 2:
            data = data.T  # -> (samples, timesteps) | (channels, timesteps)
            data = np.expand_dims(data, -1)
        return data

    def _style_axis(ax, kw, show_borders, show_xy_ticks, annotations, xmax):
        ax.axis(xmin=0, xmax=xmax)
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])
        if annotations:
            ax.annotate(annotations.pop(0), **kw['annot'])
        if not show_borders:
            ax.set_frame_on(False)

    kw = _process_configs(configs, w, h, tight, share_xy)
    _catch_unknown_kwargs(kwargs)
    if isinstance(show_xy_ticks, (int, bool)):
        show_xy_ticks = (show_xy_ticks, show_xy_ticks)
    data = _process_data(data)
    n_rows, n_cols, color, annotations = _get_style_info(data, n_rows, color,
                                                         annotations)
    fig, axes = plt.subplots(n_rows, n_cols, **kw['subplot'])
    axes = np.asarray(axes)

    if title:
        title = _get_title(data, title, subplot_samples)
        fig.suptitle(title, **kw['title'])

    for ax_idx, ax in enumerate(axes.flat):
        if show_y_zero:
            ax.axhline(0, color='red')

        feature_outputs = _get_feature_outputs(data, ax_idx)
        for idx, out in enumerate(feature_outputs):
            ax.plot(out, color=color[idx], **kw['plot'])

        _style_axis(ax, kw, show_borders, show_xy_ticks, annotations,
                    xmax=len(feature_outputs[0]))

    if tight:
        fig.subplots_adjust(**kw['tight'])
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]

    scalefig(fig)
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, **kw['save'])
    return fig, axes


def features_2D(data, n_rows=None, norm=None, cmap='bwr', reflect_half=False,
                timesteps_xaxis=False, max_timesteps=None, share_xy=(1, 1),
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
        share_xy: bool/str/(tuple/list of bool/str) for (sharex, sharey)
              kwargs passed to plt.subplots(). If bool/str, will use same
              value for sharex & sharey.
              - sharex: bool/str. True  -> subplots will share x-axes limits.
                                  'col' -> limits shared along columns.
                                  'row' -> limits shared along rows.
                                  Overrides 'sharex' in `configs['subplot']`.
              - sharey: bool/str. `sharex`, but for y-axis.
        configs: dict. kwargs to customize various plot schemes:
            'plot':     passed to fig.hist(); fig = subplots figure
            'subplot':  passed to plt.subplots()
            'title':    passed to fig.suptitle()
            'tight':    passed to fig.subplots_adjust()
            'colorbar': passed to fig.colorbar()
            'save':     passed to fig.savefig() if `savepath` is not None.

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_colorbar: bool. If True, shows one colorbar next to plot(s).
        title: bool/str. If True, shows generic supertitle.
              If str in {'grads', 'outputs', 'generic'}, shows supertitle
              tailored to `data` dim (2D/3D). If other str, shows `title`
              as supertitle. If False, no title is shown.
        tight: bool. If True, plots compactly by removing subplot padding.
        channel_axis: int, 0 or -1. `data` axis holding channels/features.
              -1 --> (samples,  timesteps, channels)
              0  --> (channels, timesteps, samples)
        borderwidth: float / None. Width of subplot borders.
        bordercolor: str / None. Color of subplot borders. Default black.
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (figs, axes) of generated plots.
    """

    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', True)
    show_xy_ticks = kwargs.get('show_xy_ticks', (1, 1))
    show_colorbar = kwargs.get('show_colorbar', False)
    title         = kwargs.get('title', 'outputs')
    tight         = kwargs.get('tight', False)
    channel_axis  = kwargs.get('channel_axis', -1)
    borderwidth   = kwargs.get('borderwidth', None)
    bordercolor   = kwargs.get('bordercolor', None)
    savepath      = kwargs.get('savepath', None)

    def _process_configs(configs, w, h, tight, share_xy):
        def _set_share_xy(kw, share_xy):
            if isinstance(share_xy, (list, tuple)):
                sharex, sharey = share_xy
            else:
                sharex = sharey = share_xy
            if isinstance(sharex, int):
                sharex = bool(sharex)
            if isinstance(sharey, int):
                sharey = bool(sharey)
            kw['subplot']['sharex'] = sharex
            kw['subplot']['sharey'] = sharey

        defaults = {
            'plot':    dict(),
            'subplot': dict(dpi=76, figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=14, y=.93 + .12 * tight),
            'tight':   dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0),
            'colorbar': dict(),
            'save':     dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)

        _set_share_xy(kw, share_xy)
        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks',
                          'show_colorbar', 'title', 'tight', 'channel_axis',
                          'borderwidth', 'bordercolor', 'savepath')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title, timesteps_xaxis, vmin, vmax):
        if title not in {'grads', 'outputs', 'generic'}:
            return title

        feature = "Context-feature"
        context = "Context-units"
        context_order = "(%s vs. Channels)" % context
        extra_dim = ""

        if title in {'grads', 'outputs'}:
            feature = "Gradients" if title == 'grads' else "Outputs"
            context = "Timesteps"
        if timesteps_xaxis:
            context_order = "(Channels vs. %s)" % context
        if data.ndim == 3:
            extra_dim = ") vs. Samples"
            context_order = "(" + context_order

        norm_txt = "(%.4G, %.4G)" % (vmin, vmax) if (vmin is not None
                                                     ) else "auto"
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
            raise Exception("`data` must be 2D or 3D (got ndim=%s)" % data.ndim)

        if max_timesteps is not None:
            data = data[..., :max_timesteps, :]

        if reflect_half:
            data = data.copy()  # prevent passed array from changing
            half_chs = data.shape[-1] // 2
            data[..., half_chs:] = np.flip(data[..., half_chs:], axis=0)

        if data.ndim != 3:
            # (1, width, height) -> one image
            data = np.expand_dims(data, 0)
        if timesteps_xaxis:
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
    kw = _process_configs(configs, w, h, tight, share_xy)
    if isinstance(show_xy_ticks, (int, bool)):
        show_xy_ticks = (show_xy_ticks, show_xy_ticks)
    data = _process_data(data, max_timesteps, reflect_half,
                         timesteps_xaxis, channel_axis)
    n_rows, n_cols, vmin, vmax = _get_style_info(data, n_rows, norm)

    fig, axes = plt.subplots(n_rows, n_cols, **kw['subplot'])
    axes = np.asarray(axes)

    if title:
        title = _get_title(data, title, timesteps_xaxis, vmin, vmax)
        fig.suptitle(title, **kw['title'])

    for ax_idx, ax in enumerate(axes.flat):
        img = ax.imshow(data[ax_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                        **kw['plot'])
        _style_axis(ax, kw, show_borders, show_xy_ticks)

    if show_colorbar:
        fig.colorbar(img, ax=axes.ravel().tolist(), **kw['colorbar'])
    if tight:
        fig.subplots_adjust(**kw['tight'])
    if borderwidth is not None or bordercolor is not None:
        for ax in axes.flat:
            for s in ax.spines.values():
                if borderwidth is not None:
                    s.set_linewidth(borderwidth)
                if bordercolor is not None:
                    s.set_color(bordercolor)
    scalefig(fig)
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, **kw['save'])
    return fig, axes


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
                  center_zero=False, share_xy=('col', 1), pad_xticks=None,
                  annotations='auto', configs=None, **kwargs):
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
        center_zero: bool. If True, symmetrize xlims: (-max, max).
              Overrides `xlims`.
        share_xy: bool/str/(tuple/list of bool/str) for (sharex, sharey)
              kwargs passed to plt.subplots(). If bool/str, will use same
              value for sharex & sharey.
              - sharex: bool/str. True  -> subplots will share x-axes limits.
                                  'col' -> limits shared along columns.
                                  'row' -> limits shared along rows.
                                  Overrides 'sharex' in `configs['subplot']`.
              - sharey: bool/str. `sharex`, but for y-axis.
        pad_xticks: bool / None. True: display only min/max xticks, and
                    shifted above xaxis and 10% inward. False: no change in
                    shown ticks. None: if bool(`tight` an not `sharex`) ->
                    sets to True. Padding behavior configurable via
                    configs['pad_xticks'] (see below).
        annotations: str list/'auto'/None.
            'auto': annotate each subplot with its index.
             list of str: annotate by indexing into the list.
                          If len(list) < len(data), won't annotate remainder.
             None: don't annotate.
        configs: dict. kwargs to customize various plot schemes:
            'plot':    passed partly* to ax.hist() in hist_clipped();
                         include `peaks_to_clip` to adjust ylims with a
                         number of peaks disregarded. *See help(hist_clipped).
                         ax = subplots axis
            'subplot': passed to plt.subplots()
            'title':   passed to fig.suptitle(); fig = subplots figure
            'tight':   passed to fig.subplots_adjust()
            'annot':   passed to ax.annotate()
            'save':    passed to fig.savefig() if `savepath` is not None.
            'pad_xticks': passed to ax.annotate(). Is nested dict:
                {'min': kw1, 'max': kw2}, applied separately for
                ax.annotate(xmin, **kw1) and ax.annotate(xmax, **kw2).

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
        title: str/None. If not None, show `title` as plt.suptitle.
        borderwidth: float / None. Width of subplot borders.
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (figs, axes) of generated plots.
    """
    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', True)
    show_xy_ticks = kwargs.get('show_xy_ticks', (1, 1))
    title         = kwargs.get('title', None)
    borderwidth   = kwargs.get('borderwidth', None)
    savepath      = kwargs.get('savepath', None)

    def _process_configs(configs, w, h, tight, share_xy):
        def _set_share_xy(kw, share_xy):
            if isinstance(share_xy, (list, tuple)):
                sharex, sharey = share_xy
            else:
                sharex = sharey = share_xy
            if isinstance(sharex, int):
                sharex = bool(sharex)
            if isinstance(sharey, int):
                sharey = bool(sharey)
            kw['subplot']['sharex'] = sharex
            kw['subplot']['sharey'] = sharey

        defaults = {
            'plot':    dict(peaks_to_clip=0),
            'subplot': dict(dpi=76, figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=14, y=.93 + .12 * tight),
            'tight':   dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0),
            'annot':   dict(weight='bold', fontsize=14, xy=(.02, .7),
                            xycoords='axes fraction', color='g'),
            'pad_xticks': {'min': dict(fontsize=12, xy=(.03, .1),
                                       xycoords='axes fraction'),
                           'max': dict(fontsize=12, xy=(.93, .1),
                                       xycoords='axes fraction')},
            'save': dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)

        _set_share_xy(kw, share_xy)
        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _process_pad_xticks(pad_xticks, kw):
        if pad_xticks is None:
            if tight and not kw['subplot']['sharex']:
                pad_xticks = True
            else:
                pad_xticks = False
        return pad_xticks

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks', 'title',
                          'borderwidth', 'savepath')
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
            annotations = list(map(str, range(n_subplots)))
        elif annotations is not None:
            # ensure external list is unaffected
            annotations = annotations.copy()
        return n_rows, n_cols, annotations

    def _style_axis(ax, kw, show_borders, show_xy_ticks, xlims,
                    center_zero, pad_xticks, annotations):
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])
        if annotations:
            ax.annotate(annotations.pop(0), **kw['annot'])
        if not show_borders:
            ax.set_frame_on(False)
        if center_zero:
            maxlim = max(np.abs(ax.get_xlim()))
            ax.set_xlim(-maxlim, maxlim)
        elif xlims is not None:
            ax.set_xlim(*xlims)
        if pad_xticks:
            xmin, xmax = clipnums(ax.get_xlim())
            ax.annotate(xmin, **kw['pad_xticks']['min'])
            ax.annotate(xmax, **kw['pad_xticks']['max'])

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h, tight, share_xy)
    if isinstance(show_xy_ticks, (int, bool)):
        show_xy_ticks = (show_xy_ticks, show_xy_ticks)
    pad_xticks = _process_pad_xticks(pad_xticks, kw)

    n_rows, n_cols, annotations = _get_style_info(data, n_rows, annotations)
    if center_zero and xlims is not None:
        print(NOTE, "`center_zero` will override `xlims`")

    fig, axes = plt.subplots(n_rows, n_cols, **kw['subplot'])
    if n_cols == 1:
        axes = np.expand_dims(axes, -1)
    elif n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if title is not None:
        fig.suptitle(title, **kw['title'])

    for ax_idx, ax in enumerate(axes.flat):
        hist_clipped(data[ax_idx], ax=ax, bins=bins, **kw['plot'])
        _style_axis(ax, kw, show_borders, show_xy_ticks, xlims,
                    center_zero, pad_xticks, annotations)

    if tight:
        fig.subplots_adjust(**kw['tight'])
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]

    scalefig(fig)
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, **kw['save'])
    return fig, axes


def features_hist_v2(data, colnames=None, bins=100, xlims=None, ylim=None,
                     center_zero=False, tight=True, share_xy=('col', 1),
                     pad_xticks=None, side_annot=None, configs=None, **kwargs):
    """Plots histograms in a subplot grid; tailored for multiple histograms
    per gridcell.

    Arguments:
        data: np.ndarray / list of np.ndarray / list of lists. Data to plot.
              (rows, cols, subdata...)
              - Plots dim0 along rows, dim1 along cols, subdata within
              individual subplots (iteratively), flattens remaining dims.
              - If list of (list / np.ndarray), each of (top-level) list's
              entries must contain same number of lists / arrays.
              - `subdata` can have any number of entries of any dims, generating
              dynamic number of histograms per gridcell.
        colnames: str list. Column titles, displayed on top subplot boxes.
        bins: int. Pyplot kwarg to plt.hist(), # of bins.
        xlims: float tuple. x limits to apply to all subplots.
        center_zero: bool. If True, symmetrize xlims: (-max, max).
              Overrides `xlims`.
        ylim: float. Top y limit of all subplots.
        tight: bool. If True, plt.subplots_adjust (spacing) according to
              configs['tight'], defaulted to minimize intermediate padding.
        share_xy: bool/str/(tuple/list of bool/str) for (sharex, sharey)
              kwargs passed to plt.subplots(). If bool/str, will use same
              value for sharex & sharey.
              - sharex: bool/str. True  -> subplots will share x-axes limits.
                                  'col' -> limits shared along columns.
                                  'row' -> limits shared along rows.
                                  Overrides 'sharex' in `configs['subplot']`.
              - sharey: bool/str. `sharex`, but for y-axis.
        pad_xticks: bool / None. True: display only min/max xticks, and
                    shifted above xaxis and 10% inward. False: no change in
                    shown ticks. None: if bool(`tight` an not `sharex`) ->
                    sets to True. Padding behavior configurable via
                    configs['pad_xticks'] (see below).
        side_annot: str. Text to display to the right side of rightmost subplot
              boxes, enumerated by row number ({side_annot}{row})
        configs: dict. kwargs to customize various plot schemes:
            'plot':       passed partly* to ax.hist() in hist_clipped();
                            include `peaks_to_clip` to adjust ylims with a
                            number of peaks disregarded. *See help(hist_clipped).
                            ax = subplots axis
            'subplot':    passed to plt.subplots()
            'title':      passed to fig.suptitle(); fig = subplots figure
            'tight':      passed to fig.subplots_adjust()
            'colnames':   passed to ax.set_title()
            'side_annot': passed to ax.annotate()
            'save':       passed to fig.savefig() if `savepath` is not None.
            'pad_xticks': passed to ax.annotate(). Is nested dict:
                {'min': kw1, 'max': kw2}, applied separately for
                ax.annotate(xmin, **kw1) and ax.annotate(xmax, **kw2).

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
        title: str/None. If not None, show `title` as plt.suptitle.
        borderwidth: float / None. Width of subplot borders.
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (figs, axes) of generated plots.
    """
    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', True)
    show_xy_ticks = kwargs.get('show_xy_ticks', (1, 1))
    title         = kwargs.get('title', None)
    borderwidth   = kwargs.get('borderwidth', None)
    savepath      = kwargs.get('savepath', None)

    def _process_configs(configs, w, h, share_xy):
        def _set_share_xy(kw, share_xy):
            if isinstance(share_xy, (list, tuple)):
                sharex, sharey = share_xy
            else:
                sharex = sharey = share_xy
            if isinstance(sharex, int):
                sharex = bool(sharex)
            if isinstance(sharey, int):
                sharey = bool(sharey)
            kw['subplot']['sharex'] = sharex
            kw['subplot']['sharey'] = sharey

        defaults = {
            'plot':    dict(peaks_to_clip=0),
            'subplot': dict(dpi=76, figsize=(10, 10)),
            'title':   dict(weight='bold', fontsize=15, y=1.06),
            'tight':   dict(left=0, right=1, bottom=0, top=1,
                            wspace=.05, hspace=.05),
            'colnames':   dict(weight='bold', fontsize=14),
            'side_annot': dict(weight='bold', fontsize=14,
                               xy=(1.02, .5), xycoords='axes fraction'),
            'pad_xticks': {'min': dict(fontsize=12, xy=(.03, .1),
                                       xycoords='axes fraction'),
                           'max': dict(fontsize=12, xy=(.93, .1),
                                       xycoords='axes fraction')},
            'save': dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)

        _set_share_xy(kw, share_xy)
        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _process_pad_xticks(pad_xticks, kw):
        if pad_xticks is None:
            if tight and not kw['subplot']['sharex']:
                pad_xticks = True
            else:
                pad_xticks = False
        return pad_xticks

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks', 'title',
                          'borderwidth', 'savepath')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_data_info(data):
        n_rows = len(data)
        n_cols = len(data[0])
        if isinstance(data, (list, tuple)):
            assert all(len(x) == n_cols for x in data[1:]), (
                "When `data` is a list of lists or list of arrays, each "
                "of the (top-level) list's entries must have the same "
                "number of elements")
        return n_rows, n_cols

    def _style_axis(ax, kw, show_borders, show_xy_ticks, xlims, center_zero,
                    pad_xticks):
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
        if center_zero:
            maxlim = max(np.abs(ax.get_xlim()))
            ax.set_xlim(-maxlim, maxlim)
        elif xlims is not None:
            ax.set_xlim(*xlims)
        if pad_xticks:
            xmin, xmax = clipnums(ax.get_xlim())
            ax.annotate(xmin, **kw['pad_xticks']['min'])
            ax.annotate(xmax, **kw['pad_xticks']['max'])

    _catch_unknown_kwargs(kwargs)
    kw = _process_configs(configs, w, h, share_xy)
    if isinstance(show_xy_ticks, (int, bool)):
        show_xy_ticks = (show_xy_ticks, show_xy_ticks)
    pad_xticks = _process_pad_xticks(pad_xticks, kw)

    n_rows, n_cols = _get_data_info(data)
    if center_zero and xlims is not None:
        print(NOTE, "`center_zero` will override `xlims`")

    fig, axes = plt.subplots(n_rows, n_cols, **kw['subplot'])
    if n_cols == 1:
        axes = np.expand_dims(axes, -1)
    elif n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if title is not None:
        fig.suptitle(title, **kw['title'])

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            for subdata in data[row][col]:
                hist_clipped(subdata, ax=ax, bins=bins, **kw['plot'])

            _style_axis(ax, kw, show_borders, show_xy_ticks, xlims,
                        center_zero, pad_xticks)

    if ylim is not None:
        ax.set_ylim(0, ylim)
    if tight:
        fig.subplots_adjust(**kw['tight'])
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]

    scalefig(fig)
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, **kw['save'])
    return fig, axes


def hist_clipped(data, peaks_to_clip=1, ax=None, annot_kw={}, **kw):
    """Histogram with ymax adjusted to a sub-maximal peak, with excluded peaks
    annotated. Useful when extreme peaks dominate plots.

    Arguments:
        data. np.ndarray / list. Data to plot.
        peaks_to_clip: int. ==0 -> regular histogram.
                             >0 -> (e.g. 2) adjust ymax to third peak.
        ax: AxesSubplot. To plot via ax.hist instead of plt.hist.
        annot_kw: dict/None. Passed to plt.annotate().
           None: doesn't annotate.
           {}: uses defaults.
           Non-empty dict: fills with defaults where absent.
        **kw: dict. Passed to plt.hist().

    Returns:
        None
    """
    def _annotate(ax, peaks_info, annot_kw):
        def _process_annot_kw(annot_kw):
            defaults = dict(weight='bold', fontsize=13, color='r',
                            xy=(.83, .85), xycoords='axes fraction')
            if not annot_kw:
                annot_kw = defaults.copy()
            else:
                annot_kw = annot_kw.copy()  # ensure external dict unaffected
                # if `defaults` key not in `annot_kw`, add it & its value
                for k, v in defaults.items():
                    if k not in annot_kw:
                        annot_kw[k] = v
            return annot_kw

        def _make_annotation(peaks_info):
            txt = ''
            for entry in peaks_info:
                txt += "({:.2f}, {})\n".format(entry[0], int(entry[1]))
            return txt.rstrip('\n')

        annot_kw = _process_annot_kw(annot_kw)
        txt = _make_annotation(peaks_info)
        if ax is not None:
            ax.annotate(txt, **annot_kw)
        else:
            plt.annotate(txt, **annot_kw)

    if ax is not None:
        N, bins, _ = ax.hist(np.asarray(data).ravel(), **kw)
    else:
        N, bins, _ = plt.hist(np.asarray(data).ravel(), **kw)

    if peaks_to_clip == 0:
        return

    Ns = np.sort(N)
    lower_max = Ns[-(peaks_to_clip + 1)]

    peaks_info = []
    for peak_idx in range(1, peaks_to_clip + 1):
        patch_idx = np.where(N == Ns[-peak_idx])[0][0]
        peaks_info.append([bins[patch_idx], N[patch_idx]])

    if ax is not None:
        ax.set_ylim(0, lower_max * 1.05)  # include small gap
    else:
        plt.ylim(0, lower_max * 1.05)

    if annot_kw is not None:
        _annotate(ax, peaks_info, annot_kw)
