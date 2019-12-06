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

    kwargs:
        scale_width:   float. Scale width  of resulting plot by a factor.
        scale_height:  float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plots.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        equate_axes: int: 0, 1, 2. 
        bins: int. Pyplot `hist` kwarg: number of histogram bins per subplot.
    """

    scale_width   = kwargs.get('scale_width',   1)
    scale_height  = kwargs.get('scale_height',  1)
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
          ax = np.array(axes[direction_idx]).T
          for type_idx in range(2):  # 2 = len(kernel_types)
            x_new += [np.max(np.abs([axis.get_xlim() for axis in ax[type_idx]]))]
            y_new += [np.max(np.abs([axis.get_ylim() for axis in ax[type_idx]]))]
        return max(x_new), max(y_new)

    def _set_axes_limits(axes, x_new, y_new):
        axes = np.array(axes)
        is_bidir = len(axes.shape)==3 and axes.shape[0]!=1

        for direction_idx in range(1 + is_bidir):
          ax = np.array(axes[direction_idx]).T
          for type_idx in range(2):
            for gate_idx in range(num_gates):
              ax[type_idx][gate_idx].set_xlim(-x_new, x_new)
              ax[type_idx][gate_idx].set_ylim(0,      y_new) 

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
                    ax.annotate(gate_names[gate_idx], xy=(0.90, 0.93), 
                        xycoords='axes fraction', fontsize=12, weight='bold')

                nan_txt = _detect_nans(weight_data)
                if nan_txt is not None:
                    ax.annotate(nan_txt, xy=(0.05, 0.63), color='red', 
                            weight='bold', xycoords='axes fraction', fontsize=12)

                if not show_borders:
                    ax.set_frame_on(False)
                if not show_xy_ticks[0]:
                    ax.set_xticks([])
                if not show_xy_ticks[1]:
                    ax.set_yticks([])

        plt.tight_layout()
        plt.gcf().set_size_inches(11*scale_width, 4.5*scale_height)
        
        if uses_bias: # does not equate axes
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
                cmap='bwr', norm=None, gate_sep_width=.5, 
                show_colorbar=False, show_bias=True, 
                scale_width=1, scale_height=1):
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
        data -= np.min(data)  # TODO: rid? include as kwarg option?
        data /= np.max(data)  # normalize 

    (vmin, vmax) = norm if norm else _make_common_norm(data)
    gate_names = '(%s)' % ', '.join(gate_names)
    
    for direction_idx, direction_name in enumerate(direction_names):
        plt.figure(figsize=(14*scale_width, 5*scale_height))
        if direction_name != []:
            plt.suptitle(direction_name + ' LAYER', weight='bold', y=.98)
            
        for type_idx, kernel_type in enumerate(['KERNEL', 'RECURRENT']):
            plt.subplot(1, 2, type_idx+1)
            w_idx = type_idx + direction_idx*(2 + uses_bias)
            
            # Plot heatmap, gate separators (IF), colorbar (IF)
            plt.imshow(data[w_idx], cmap=cmap, interpolation='nearest', 
                       aspect='auto', vmin=vmin, vmax=vmax)
            if gate_sep_width != 0:
                lw = gate_sep_width
                [plt.axvline(rnn_dim * gate_idx - .5, linewidth=lw, color='k') 
                 for gate_idx in range(1, num_gates)]
            if show_colorbar and type_idx==1:
                plt.colorbar()
                
            # Styling
            plt.title(kernel_type, fontsize=14, weight='bold')
            plt.xlabel(gate_names, fontsize=12, weight='bold')
            y_label = ['Channel units', 'Hidden units'][type_idx]
            plt.ylabel(y_label, fontsize=12, weight='bold')
            plt.gcf().set_size_inches(14*scale_width, 5*scale_height)
    
        if uses_bias and show_bias: # does not equate axes
            plt.figure(figsize=(14*scale_width, 1*scale_height))
            plt.subplot(num_gates+1, 2, (2*num_gates + 1, 2*num_gates + 2))
            weights_viz = np.atleast_2d(data[2 + direction_idx*(2 + uses_bias)])

            # Plot heatmap, gate separators (IF), colorbar (IF)
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
