# See RNN

[![Build Status](https://travis-ci.com/OverLordGoldDragon/see-rnn.svg?token=dGKzzAxzJjaRLzddNsCd&branch=master)](https://travis-ci.com/OverLordGoldDragon/see-rnn)
[![Coverage Status](https://coveralls.io/repos/github/OverLordGoldDragon/see-rnn/badge.svg?branch=master&service=github&kill_cache=1)](https://coveralls.io/github/OverLordGoldDragon/see-rnn?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e15b1b772c3f4dc9ba7988784a2b9bf6)](https://www.codacy.com/manual/OverLordGoldDragon/see-rnn?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OverLordGoldDragon/see-rnn&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0-blue.svg)

RNN weights, gradients, &amp; activations visualization in Keras &amp; TensorFlow (LSTM, GRU, SimpleRNN, CuDNN, & all others)

<img src="https://user-images.githubusercontent.com/16495490/69360375-df132d80-0ca3-11ea-80ef-e5749965e3ff.png" width="900">
<img src="https://user-images.githubusercontent.com/16495490/69359963-133a1e80-0ca3-11ea-9c9a-2c59baa112dd.png" width="850">

## Features
  - **Weights, gradients, activations** visualization
  - **Kernel visuals**: kernel, recurrent kernel, and bias shown explicitly
  - **Gate visuals**: gates in gated architectures (LSTM, GRU) shown explicitly
  - **Channel visuals**: cell units (feature extractors) shown explicitly


## Examples

```python
# for all examples
grads = get_rnn_gradients(model, x, y, layer_idx=1)  # return_sequences=True
grads = get_rnn_gradients(model, x, y, layer_idx=2)  # return_sequences=False
# all examples use timesteps=100
```

<hr>

**EX 1: bi-LSTM, 32 units** - activations, `activation='relu'`

 - Each subplot is an independent RNN channel's output (`return_sequences=True`)
 - In this example, each channel/filter appears to extract complex independent features of varying bias, frequency, and probabilistic distribution

<img src="https://i.stack.imgur.com/k7RrD.png" width="800">

<img src="https://i.stack.imgur.com/HF8gH.png" width="800">

<hr>

**EX 2: one sample, uni-LSTM, 6 units** -- `return_sequences=True`, trained for 20 iterations <br>
`show_features_1D(grads[0], n_rows=2)`

 - _Note_: gradients are to be read _right-to-left_, as they're computed (from last timestep to first)
 - Rightmost (latest) timesteps consistently have a higher gradient
 - **Vanishing gradient**: ~75% of leftmost timesteps have a zero gradient, indicating poor time dependency learning

[![enter image description here][1]][1]

<hr>

**EX 3: all (16) samples, uni-LSTM, 6 units** -- `return_sequences=True`, trained for 20 iterations <br>
`show_features_1D(grads, n_rows=2)`<br>
`show_features_2D(grads, n_rows=4, norm=(-.01, .01))`

 - Each sample shown in a different color (but same color per sample across channels)
 - Some samples perform better than one shown above, but not by much
 - The heatmap plots channels (y-axis) vs. timesteps (x-axis); blue=-0.01, red=0.01, white=0 (gradient values)

[![enter image description here][2]][2]
[![enter image description here][3]][3]

<hr>

**EX 4: all (16) samples, uni-LSTM, 6 units** -- `return_sequences=True`, trained for 200 iterations <br>
`show_features_1D(grads, n_rows=2)`<br>
`show_features_2D(grads, n_rows=4, norm=(-.01, .01))`

 - Both plots show the LSTM performing clearly better after 180 additional iterations
 - Gradient still vanishes for about half the timesteps
 - All LSTM units better capture time dependencies of one particular sample (blue curve, first plot) - which we can tell from the heatmap to be the first sample. We can plot that sample vs. other samples to try to understand the difference

[![enter image description here][4]][4]
[![enter image description here][5]][5]

<hr>

**EX 5: 2D vs. 1D, uni-LSTM**: 256 units, `return_sequences=True`, trained for 200 iterations <br>
`show_features_1D(grads[0])`<br>
`show_features_2D(grads[:, :, 0], norm=(-.0001, .0001))`

 - 2D is better suited for comparing many channels across few samples
 - 1D is better suited for comparing many samples across a few channels

[![enter image description here][6]][6]

<hr>

**EX 6: bi-GRU, 256 units (512 total)** -- `return_sequences=True`, trained for 400 iterations <br>
`show_features_2D(grads[0], norm=(-.0001, .0001), reflect_half=True)`

 - Backward layer's gradients are flipped for consistency w.r.t. time axis
 - Plot reveals a lesser-known advantage of Bi-RNNs - _information utility_: the collective gradient covers about twice the data. _However_, this isn't free lunch: each layer is an independent feature extractor, so learning isn't really complemented
 - Lower `norm` for more units is expected, as approx. the same loss-derived gradient is being distributed across more parameters (hence the squared numeric average is less)

<img src="https://i.stack.imgur.com/ueGVB.png" width="420">

<hr>

**EX 7: 0D, all (16) samples, uni-LSTM, 6 units** -- `return_sequences=False`, trained for 200 iterations<br>
`show_features_0D(grads)`

 - `return_sequences=False` utilizes only the last timestep's gradient (which is still derived from all timesteps, unless using truncated BPTT), requiring a new approach
 - Plot color-codes each RNN unit consistently across samples for comparison (can use one color instead)
 - Evaluating gradient flow is less direct and more theoretically involved. One simple approach is to compare distributions at beginning vs. later in training: if the difference isn't significant, the RNN does poorly in learning long-term dependencies

<img src="https://i.stack.imgur.com/693EO.png" width="560">

<hr>

**EX 8: LSTM vs. GRU vs. SimpleRNN, unidir, 256 units** -- `return_sequences=True`, trained for 250 iterations<br>
`show_features_2D(grads, n_rows=8, norm=(-.0001, .0001), show_xy_ticks=[0,0], show_title=False)`

 - _Note_: the comparison isn't very meaningful; each network thrives w/ different hyperparameters, whereas same ones were used for all. LSTM, for one, bears the most parameters per unit, drowning out SimpleRNN
 - In this setup, LSTM definitively stomps GRU and SimpleRNN

[![enter image description here][7]][7]

<hr>


**EX 9: uni-LSTM, 256 units, weights** -- `batch_shape = (16, 100, 20)` (input)<br>
`rnn_histogram(model, 'lstm', equate_axes=False, show_bias=False)`<br>
`rnn_histogram(model, 'lstm', equate_axes=True,  show_bias=False)`<br>
`rnn_heatmap(model, 'lstm')`

 - Top plot is a histogram subplot grid, showing weight distributions per kernel, and within each kernel, per gate
 - Second plot sets `equate_axes=True` for an even comparison across kernels and gates, improving quality of comparison, but potentially degrading visual appeal
 - Last plot is a heatmap of the same weights, with gate separations marked by vertical lines, and bias weights also included
 - Unlike histograms, the heatmap _preserves channel/context information_: input-to-hidden and hidden-to-hidden transforming matrices can be clearly distinguished
 - Note the large concentration of maximal values at the Forget gate; as trivia, in Keras (and usually), bias gates are all initialized to zeros, except the Forget bias, which is initialized to ones



<img src="https://i.stack.imgur.com/1Deh4.png" width="600">

<img src="https://i.stack.imgur.com/IZN6k.png" width="600">

<img src="https://i.stack.imgur.com/E9GkQ.png" width="620">


<hr>

**EX 10: bi-CuDNNLSTM, 256 units, weights** -- `batch_shape = (16, 100, 16)` (input)<br>
`rnn_histogram(model, 'bidir', equate_axes=2)`<br>
`rnn_heatmap(model, 'bidir', norm=(-.8, .8))`

 - Bidirectional is supported by both; biases included in this example for histograms
 - Note again the bias heatmaps; they no longer appear to reside in the same locality as in EX 1. Indeed, `CuDNNLSTM` (and `CuDNNGRU`) biases are defined and initialized differently - something that can't be inferred from histograms

<img src="https://i.stack.imgur.com/vkGiF.png" width="900">

<img src="https://i.stack.imgur.com/gEjp0.png" width="900">

<hr>

**EX 11: uni-CuDNNGRU, 64 units, weights gradients** -- `batch_shape = (16, 100, 16)` (input)<br>
`rnn_heatmap(model, 'gru', mode='grads', input_data=x, labels=y, cmap=None, absolute_value=True)`

 - We may wish to visualize _gradient intensity_, which can be done via `absolute_value=True` and a greyscale colormap
 - Gate separations are apparent even without explicit separating lines in this example:
   - `New` is the most active kernel gate (input-to-hidden), suggesting more error correction on _permitting information flow_
   - `Reset` is the least active recurrent gate (hidden-to-hidden), suggesting least error correction on memory-keeping

<img src="https://i.stack.imgur.com/cwiAS.png" width="600">

<hr>

**EX 12: NaN detection: LSTM, 512 units, weights** -- `batch_shape = (16, 100, 16)` (input)

 - Both the heatmap and the histogram come with built-in NaN detection - kernel-, gate-, and direction-wise
 - Heatmap will print NaNs to console, whereas histogram will mark them directly on the plot
 - Both will set NaN values to zero before plotting; in example below, all related non-NaN weights were already zero

<img src="https://i.stack.imgur.com/T6ZAa.png" width="600">


## Usage 

Minimal example below - for full usage, see module docstrings, which describe all functionality. _Note_: if using `tensorflow.keras` imports, set `import os; os.environ["TF_KERAS"]='True'`.

[visuals_gen.py](https://github.com/OverLordGoldDragon/see-rnn/blob/master/see_rnn/visuals_gen.py) functions can also be used to visualize `Conv1D` activations, gradients, or any other meaningfully-compatible data formats. Likewise, [inspect_gen.py](https://github.com/OverLordGoldDragon/see-rnn/blob/master/see_rnn/inspect_gen.py) also works for non-RNN layers.

```python
import numpy as np
from keras.layers import Input, LSTM
from keras.models import Model
from keras.optimizers import Adam
from see_rnn import get_rnn_gradients, show_features_1D, show_features_2D
from see_rnn import show_features_0D

def make_model(rnn_layer, batch_shape, units):
    ipt = Input(batch_shape=batch_shape)
    x   = rnn_layer(units, activation='tanh', return_sequences=True)(ipt)
    out = rnn_layer(units, activation='tanh', return_sequences=False)(x)
    model = Model(ipt, out)
    model.compile(Adam(4e-3), 'mse')
    return model
    
def make_data(batch_shape):
    return np.random.randn(*batch_shape), \
           np.random.uniform(-1, 1, (batch_shape[0], units))

def train_model(model, iterations, batch_shape):
    x, y = make_data(batch_shape)
    for i in range(iterations):
        model.train_on_batch(x, y)
        print(end='.')  # progbar
        if i % 40 == 0:
            x, y = make_data(batch_shape)

units = 6
batch_shape = (16, 100, 2*units)

model = make_model(LSTM, batch_shape, units)
train_model(model, 300, batch_shape)

x, y  = make_data(batch_shape)
grads_all  = get_rnn_gradients(model, x, y, layer_idx=1)  # return_sequences=True
grads_last = get_rnn_gradients(model, x, y, layer_idx=2)  # return_sequences=False

show_features_1D(grads_all, n_rows=2, show_xy_ticks=[1,1])
show_features_2D(grads_all, n_rows=8, show_xy_ticks=[1,1], norm=(-.01, .01))
show_features_0D(grads_last)
```



  [1]: https://i.stack.imgur.com/PVoU0.png
  [2]: https://i.stack.imgur.com/OaX6I.png
  [3]: https://i.stack.imgur.com/RW24R.png
  [4]: https://i.stack.imgur.com/SUIN3.png
  [5]: https://i.stack.imgur.com/nsNR1.png
  [6]: https://i.stack.imgur.com/Ci2AP.png
  [7]: https://i.stack.imgur.com/vWgc8.png
