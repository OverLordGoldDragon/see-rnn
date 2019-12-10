### How does it work?

Every Keras RNN layer has a predefined object structure from which weights, gradients, and outputs/activations can be extracted. See RNN 
iterates through every possible architecture (LSTM, GRU, SimpleRNN, CuDNNLSTM, CuDNNGRU, IndRNN), and extracts information accordingly on a 
per-case basis. Exploring this repo's code can give useful insights into Keras' RNN implementations and limitations.

<hr>

**I/O Dimensionalities (all RNNs)** - critical to understanding architecture basics

 - **Input**: `(batch_size, timesteps, channels)` - or, equivalently, `(samples, timesteps, features)`
 - **Output**: same as Input, except:
   - `channels`/`features` is now the _# of RNN units_, and:
   - `return_sequences=True` --> `timesteps_out = timesteps_in` (output a prediction for each input timestep)
   - `return_sequences=False` --> `timesteps_out = 1` (output prediction only at the last timestep processed)
   
<hr>

**Weights**

Suppose shape of input to RNN layers below is `(batch_size, timesteps, input_features)`, and the layers have `units` channels

 - **All RNNs**:
   - *Kernel*: input-to-hidden transformations
   - *Recurrent kernel*: hidden-to-hidden transformations
   - *Bias*: applied to either or both
 - **LSTM** - `Input, Forget, Cell, Output` -- gate ordering for all kernels & biases
 - **GRU** - `Update, Reset, New` -- gate ordering for all kernels & biases
 - **SimpleRNN** - nongated

```python
# LSTM
kernel.shape           == (input_features, 4*units)
recurrent_kernel.shape == (units, 4*units)
bias.shape             == (4*units,)  # ONLY FOR recurrent_kernel 
# GRU
kernel.shape           == (input_features, 3*units)
recurrent_kernel.shape == (units, 3*units)
bias.shape             == (3*units,)  # ONLY FOR recurrent_kernel
# SimpleRNN
kernel.shape           == (input_features, units)
recurrent_kernel.shape == (units, units)
bias.shape             == (units,)  # operates on hidden transformation; see source code

# CuDNNLSTM
kernel.shape           == (input_features, 4*units)
recurrent_kernel.shape == (units, 4*units)
bias.shape             == (8*units,)  # BOTH FOR kernel & recurrent_kernel
# CuDNNGRU
kernel.shape           == (input_features, 3*units)
recurrent_kernel.shape == (units, 3*units)
bias.shape             == (6*units,)  # BOTH FOR kernel & recurrent_kernel
```

**Weight build source codes**:

 [LSTM kernels + bias](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1915) -- 
 [GRU kernels + bias](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1311) -- 
 [SimpleRNN kernels + bias](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L896) <br> 
 [CuDNNLSTM bias (& others)](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L471) --
 [CuDNNGRU bias (& others)](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L260)
