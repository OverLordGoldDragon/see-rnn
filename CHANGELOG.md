### 1.14.6 (7-25-2020): `title_mode` -> `title`; allow custom title

Docs state title will set to `title_mode` if not one of `'grads', 'outputs'`, but this was false; fixed now. Also renamed to `title`.


----


### 1.14.5 (7-6-2020): Convenience, appearance improvements; Bugfixes


#### Features

 - Added `bordercolor` kwarg to `Features_2D` which allows setting color of border lines, useful when images are majority-dark
 - Improved xaxis annotation handling for `pad_xticks` kwarg in `Features_hist` and `Features_hist_v2`; behavior configurable via `configs['pad_xticks']`
 - `show_xy_ticks` can now be an `int` or `bool`, which will automatically set to tuple `(int, int)`.
 - Changed default kwarg `timesteps_xaxis` to `False` in `Features_2D` which would rotate image data

#### Breaking
 
 - `pad_xticks` is now bool instead of int
 
#### Bugfixes

 - `Features_2D`: moved `ndim != 3` check outside of `timesteps_xaxis`, which would fail to `expand_dims(0, ...)` for `=False` w/ 2D input
 - `weights_norm`: the case of one `weights` per layer would process incorrectly due to iterating over an expected list, where `get_weights` 
   returns the array itself in case of a single weight

#### Misc
 
 - Added test cases to account for fixed bugs


----


### 1.14.4 (6-10-2020): Fix L1-norm case in `weights_norm` 

`norm_fn=np.abs` would compute L1 norm as: `np.sqrt(np.sum(np.abs(x)))`, which is incorrect; the sqrt is redundant. `norm_fn=np.abs` will now work correctly. L2-norm case always worked correctly.

For L2-norm, set `norm_fn = (np.sqrt, np.square) = (outer_fn, inner_fn)`, which will compute `outer_fn(sum(inner_fn(x)))`. Note that `norm_fn=np.square` will **no longer compute L2-norm correctly**.


#### Misc

  - A warning would be thrown even if `_id=''` or is otherwise falsy, which is redundant.


----

### 1.14.2 (5-31-2020): TF2.2-Graph `sample_weight` bugfix

#### Bugfixes

 - Passes `None` instead of `np.ones(len(x))` in `get_gradients(sample_weight=None)`. This is a [TF2.2 bug](https://github.com/tensorflow/tensorflow/issues/39888), not See RNN bug.
 - Will still bug if `sample_weight is not None` - nothing to do here except wait for TF 2.3, or nightly when fixed


----


### 1.14.1 (5-26-2020)

#### Bugfixes

 - `'softmax'` activation for `_id='*'` in `get_gradients` wasn't handled properly
 - Added test for softmax; other activations might error, exhaustive list for `None` gradient yielders undetermined

#### Misc

 - Moved testing imports to new `backend.py`
 - Changed pathing logic in `test_all.py` to allow running as `__main__`
 - Added `conftest.py` to disable plots when Spyder unit-testing, and allow when ran as `__main__`


----


### 1.14.0 (5-24-2020): TF2.2.0 support; Features

#### Features

 - Up to date with TensorFlow 2.2.0
 - Support for `sample_weight` and `learning_phase` for all backends (TF1, TF2, Eager, Graph, `keras`, `tf.keras`)
 - Support for multi-input and multi-output networks
 - `params` added to `get_gradients`; directly get grads of pre-fetched weights & outputs

#### Breaking

 - `_make_grads_fn` no longer supports Eager for `tf.keras` (but does for `keras`)
 - `_get_grads` deprecated
 - `sample_weights` -> `sample_weight` in `get_gradients`

#### Bugfixes

 - `_id='*'` will now omit `'softmax'` activation layers in `tf.keras` `get_gradients`, which error with `None` for gradient
 - Corrected gateless architecture detection for `_get_cell_weights`

#### Misc

 - Testing moved to TF 2.2, no longer includes TF 2.0 or TF 2.1
 - Added `_get_grads_eager` to `inspect_gen.py`
 - Added `_get_params`, `_layer_of_output` to `utils.py`
 - Improved `Input` layer exclusion for `_id='*'`
 - Added note + tip to `get_gradients` on performance
 - Extended GPU detection method in tests to work with TF2.2
 