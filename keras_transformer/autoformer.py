"""
Python module implementing the necessary layers for an
Autoformer in Keras
"""

import math
from itertools import zip_longest

import keras
from keras import layers, ops
from keras import KerasTensor as Tensor

from .embed import DataEmbedding

from collections.abc import Sequence
from pydantic import Field, PositiveInt
from typing import Annotated, Optional

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrLayer(layers.Layer):
  """
  Keras implementation of the autocorrelation attention
  mechanism
  """

  def __init__(self, params: dict, **kwargs):
    super(CorrLayer, self).__init__(**kwargs)
    self.params = params

    self.k = params["k"]
    self.h = params["h"]

    self.d_keys: PositiveInt = params.get("d_keys", None)
    self.d_values: PositiveInt = params.get("d_values", None)

    self.dropout = layers.Dropout(params.get("dropout_rate", 1e-3))

    self.output_attention = params.get("output_attention", False)

    self.query_proj: layers.Dense = None
    self.key_proj: layers.Dense = None
    self.value_proj: layers.Dense = None
    self.out_proj: layers.Dense = None

  def build(self, input_shape: tuple[tuple[int, ...], ...]):
    self.d_keys = self.d_keys or input_shape[0][-1] // self.h
    self.d_values = self.d_values or input_shape[0][-1] // self.h

    self.query_proj = layers.Dense(self.h * self.d_keys)
    self.key_proj = layers.Dense(self.h * self.d_keys)
    self.value_proj = layers.Dense(self.h * self.d_values)
    self.out_proj = layers.Dense(input_shape[0][-1])

  def _time_delay_agg_training(self, values: Tensor,
                               corr: Tensor) -> Tensor:
    # Find top K (Batch-normalized)
    top_k = int(self.k * math.log(values.shape[-1]))
    mean_value = ops.mean(corr, axis=(1, 2))
    delay = ops.top_k(ops.mean(mean_value, axis=0), top_k)[1]

    # Nonlinear weights for aggregation
    tmp_corr = ops.softmax(ops.take(mean_value, delay, axis=1),
                           axis=-1)

    # Aggregation
    delays_agg = ops.zeros_like(values)
    for idx in range(top_k):
      # pattern = ops.roll(values, -delay[idx], axis=-1)
      delays_agg += \
          ops.roll(values, -delay[idx], axis=-1) \
          * tmp_corr[:, idx, None, None, None]

    return delays_agg

  def _time_delay_agg_inference(self, values: Tensor, corr: Tensor,) -> Tensor:
    # Find top K (Batch-normalized)
    top_k = int(self.k * math.log(values.shape[-1]))
    weights, delay = ops.top_k(ops.mean(corr, axis=(1, 2)),
                               top_k)

    # Nonlinear weights for aggregation
    tmp_corr = ops.softmax(weights, axis=-1)

    # Aggregation
    delays_agg = ops.zeros_like(values)
    for idx in range(top_k):
      # pattern = ops.roll(values, -delay[:, idx], axis=-1)
      delays_agg += \
          ops.roll(values, -delay[:, idx], axis=-1) \
          * tmp_corr[:, idx, None, None, None]

    return delays_agg

  def call(self, inputs: tuple[Tensor, Tensor, Tensor],
           training=None) -> tuple[Tensor, Tensor]:
    queries, keys, values = inputs

    _, q_len, _ = queries.shape
    _, k_len, _ = keys.shape
    h = self.h

    queries = ops.reshape(self.query_proj(queries),
                          (-1, q_len, h, self.d_keys))
    keys = ops.reshape(self.key_proj(keys),
                       (-1, k_len, h, self.d_keys))
    values = ops.reshape(self.value_proj(values),
                         (-1, k_len, h, self.d_values))

    # Ensure dimension compatibility
    if q_len > k_len:
      zeros = ops.zeros_like(queries[:, :(q_len - k_len), ...])
      keys = ops.concatenate((keys, zeros), axis=1)
      values = ops.concatenate((values, zeros), axis=1)
    else:
      keys = keys[:, :q_len, :, :]
      values = values[:, :q_len, :, :]

    # Period-based dependencies
    q_fft = ops.rfft(ops.transpose(queries, (0, 2, 3, 1)))
    k_fft = ops.rfft(ops.transpose(keys, (0, 2, 3, 1)))
    corr = ops.irfft((q_fft[0] * k_fft[0], -q_fft[1] * k_fft[1]),
                     fft_length=q_len)

    if training:
      out = ops.transpose(
          self._time_delay_agg_training(ops.transpose(values, (0, 2, 3, 1)),
                                        corr),
          axes=(0, 3, 1, 2)
      )
    else:
      out = ops.transpose(
          self._time_delay_agg_inference(ops.transpose(values, (0, 2, 3, 1)),
                                         corr),
          axes=(0, 3, 1, 2)
      )
    out = self.out_proj(ops.reshape(out, (-1, q_len, h * self.d_values)))

    if self.output_attention:
      return self.dropout(out), ops.transpose(corr,
                                              axes=(0, 3, 1, 2))
    else:
      return self.dropout(out), None


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrEncoderLayer(layers.Layer):
  """
  Keras implementation of a single Autoformer encoder layer
  """

  def __init__(self, params: dict, **kwargs):
    super(CorrEncoderLayer, self).__init__(**kwargs)
    self.params = params

    self.d_ff: int = params.get("d_ff", None)
    self.activation = params.get("activation", "relu")
    self.tau: int = params["tau"]

    self.attn_layer = CorrLayer(params)

    self.dropout = layers.Dropout(params.get("dropout_rate", 0.1))
    self.ff_layer: keras.Sequential = None

  def build(self, input_shape: tuple[int, ...]):
    self.d_ff = self.d_ff or 4 * input_shape[-1]

    self.ff_layer = keras.Sequential(
        [
            layers.Dense(units=self.d_ff, use_bias=False,
                         activation=self.activation),
            self.dropout,
            layers.Dense(units=input_shape[-1], use_bias=False,
                         activation=None),
            self.dropout
        ]
    )

  def _series_decomp(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
    trend = ops.average_pool(inputs, self.tau, strides=1,
                             padding="same", data_format="channels_last")
    seasonality = inputs - trend
    return seasonality, trend

  def call(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
    x, attn = self.attn_layer([inputs, inputs, inputs])
    x = inputs + self.dropout(x)
    x, _ = self._series_decomp(x)

    y = self.ff_layer(x)
    y, _ = self._series_decomp(x + y)

    return y, attn

  def get_config(self):
    config = super().get_config()

    config.update({
        "params": self.params
    })

    return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrEncoder(layers.Layer):
  """
  Keras implementation of the Autoformer encoder block
  """

  def __init__(self, params: dict, **kwargs):
    super(CorrEncoder, self).__init__(**kwargs)
    self.params = params

    n_layers = params["N"]

    self.embed = DataEmbedding(params["d_model"],
                               params["dropout_rate"],
                               params["embed_type"],
                               params.get("freq"))

    if isinstance(params["tau"], Sequence):
      if len(params["tau"]) < n_layers:
        self.enc_layers = [
            CorrEncoderLayer({**params, "tau": tau})
            for tau, _ in zip_longest(params["tau"],
                                      range(n_layers),
                                      fillvalue=params["tau"][-1])
        ]
      else:
        self.enc_layers = [
            CorrEncoderLayer({**params, "tau": tau})
            for tau, _ in zip(params["tau"][-n_layers:],
                              range(n_layers))
        ]
    else:
      self.enc_layers = [
          CorrEncoderLayer(params)
          for _ in range(n_layers)
      ]

    self.norm_layer = keras.Sequential([
        layers.LayerNormalization(axis=-1),
        layers.LayerNormalization(axis=-2, scale=False)
    ])

  def build(self, input_shape):
    embedded_shape = [*input_shape[0][:-1], self.params["d_model"]]
    for encoder_layer in self.enc_layers:
      encoder_layer.build(embedded_shape)

  def call(self,
           inputs: tuple[Tensor, Tensor]
           ) -> tuple[Tensor, tuple[Optional[Tensor], ...]]:
    attns = []
    x = self.embed(inputs)

    for enc_layer in self.enc_layers:
      x, attn = enc_layer(x)
      attns.append(attn)

    return self.norm_layer(x), tuple(attns)

  def get_config(self):
    config = super().get_config()

    config.update({
        "params": self.params
    })

    return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoderLayer(layers.Layer):
  """
  Keras implementation of a single Autoformer decoder layer
  """

  def __init__(self, params: dict, **kwargs):
    super(CorrDecoderLayer, self).__init__(**kwargs)
    self.params = params

    self.d_ff: int = params.get("d_ff", None)
    self.activation = params.get("activation", "softplus")
    self.tau = params["tau"]

    self.dropout = layers.Dropout(params.get("dropout_rate", 0.1))
    self.self_attn = CorrLayer({**params, "output_attention": False})
    self.cross_attn = CorrLayer({**params, "output_attention": False})

    self.ff_layer: keras.Sequential = None

    self.out_proj = layers.Conv1D(filters=params["d"], kernel_size=3,
                                  strides=1, padding="same",
                                  use_bias=False,
                                  data_format="channels_last")

  def build(self, input_shape: tuple[tuple[int, ...], ...]):
    self.d_ff = self.d_ff or 4 * input_shape[0][-1]

    self.ff_layer = keras.Sequential(
        [
            layers.Dense(units=self.d_ff, use_bias=False,
                         activation=self.activation),
            self.dropout,
            layers.Dense(units=input_shape[0][-1], use_bias=False,
                         activation=None),
            self.dropout
        ]
    )

  def _series_decomp(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
    trend = ops.average_pool(inputs, self.tau, strides=1,
                             padding="same", data_format="channels_last")
    seasonality = inputs - trend
    return seasonality, trend

  def call(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    x, cross = inputs

    x += self.dropout(self.self_attn([x, x, x])[0])
    x, xt_1 = self._series_decomp(x)

    x += self.dropout(self.cross_attn([x, cross, cross])[0])
    x, xt_2 = self._series_decomp(x)

    y = self.ff_layer(x)
    y, xt_3 = self._series_decomp(x + y)

    residual_trend = self.out_proj(xt_1 + xt_2 + xt_3)

    return y, residual_trend

  def get_config(self):
    config = super().get_config()

    config.update({
        "params": self.params
    })

    return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoder(layers.Layer):
  """
  Keras implementation of the Autoformer decoder block
  """

  def __init__(self, params: dict, **kwargs):
    super(CorrDecoder, self).__init__(**kwargs)
    self.params = params

    d = params["d"]
    n_layers = params["M"]

    self.o = params["O"]
    self.output_components = params["output_components"]

    self.embed = DataEmbedding(params["d_model"],
                               params["dropout_rate"],
                               params["embed_type"],
                               params.get("freq"))

    if isinstance(params["tau"], Sequence):
      if len(params["tau"]) < n_layers:
        self.dec_layers = [
            CorrDecoderLayer({**params, "tau": tau})
            for tau, _ in zip_longest(params["tau"],
                                      range(n_layers),
                                      fillvalue=params["tau"][-1])
        ]
      else:
        self.dec_layers = [
            CorrDecoderLayer({**params, "tau": tau})
            for tau, _ in zip(params["tau"][-n_layers:],
                              range(n_layers))
        ]
    else:
      self.dec_layers = [
          CorrDecoderLayer(params)
          for _ in range(n_layers)
      ]

    self.norm_layer = self.norm_layer = keras.Sequential([
        layers.LayerNormalization(axis=-1),
        layers.LayerNormalization(axis=-2, scale=False)
    ])
    self.out_proj = layers.Dense(d, activation="linear",
                                 use_bias=True)

  def build(self, input_shape):
    embedded_shape = input_shape[-1]
    for decoder_layer in self.dec_layers:
      decoder_layer.build((embedded_shape, embedded_shape))

  def call(self,
           inputs: tuple[Tensor | tuple[Tensor, ...], ...]
           ) -> tuple[Tensor, Tensor]:
    (xs, xt), xm, cross = inputs
    xs = self.embed([xs, xm])

    for dec_layer in self.dec_layers:
      xs, residual_trend = dec_layer([xs, cross])
      xt += residual_trend

    xs = self.out_proj(self.norm_layer(xs))

    xs = xs[:, -self.o:, :]
    xt = xt[:, -self.o:, :]

    if self.output_components:
      return xs + xt, xs, xt
    else:
      return xs + xt

  def get_config(self):
    config = super().get_config()

    config.update({
        "params": self.params
    })

    return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoderInit(layers.Layer):
  """
  Adapter layer for the seasonal/trend mechanism of the
  Autoformer.
  """

  def __init__(self, params: dict, **kwargs):
    super(CorrDecoderInit, self).__init__(**kwargs)
    self.params = params

    # Whether to adapt a manual input (known future input) or
    # generate a placeholder (paper strategy)
    self.manual_dec_input = params["manual_dec_input"]

    # moving average window
    if isinstance(params["tau"], Sequence):
      self.tau = params["tau"][0]
    else:
      self.tau = params["tau"]
    self.o = params["O"]  # prediction horizon

  def call(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    if self.manual_dec_input:
      x_enc, x_enc_marks, x_dec, x_dec_marks = inputs
      _, l, _ = x_enc.shape
      x_dec = ops.concatenate((x_enc[:, l // 2:, :], x_dec), axis=1)
      x_dec_t = ops.average_pool(x_dec, pool_size=self.tau, strides=1,
                                 padding="same", data_format="channels_last")
      x_dec_s = x_dec - x_dec_t
    else:
      x_enc, x_enc_marks, x_dec_marks = inputs
      _, l, _ = x_enc.shape
      half_input = x_enc[:, l // 2:, :]
      half_trend = ops.average_pool(half_input, pool_size=self.tau,
                                    strides=1, padding="same",
                                    data_format="channels_last")
      half_season = half_input - half_trend

      x_dec_s = ops.pad(half_season, ((0, 0), (0, self.o), (0, 0)),
                        mode="constant", constant_values=0)
      x_dec_t = ops.concatenate((half_trend,
                                 ops.tile(ops.mean(x_enc,
                                                   axis=1,
                                                   keepdims=True),
                                          (1, self.o, 1))),
                                axis=1)

    x_dec_marks = ops.concatenate((x_enc_marks[:, l // 2:, :],
                                   x_dec_marks),
                                  axis=1)

    return x_dec_s, x_dec_t, x_dec_marks

  def get_config(self):
    config = super().get_config()

    config.update({
        "params": self.params
    })

    return config


def restore_custom_objects():
  keras.saving.get_custom_objects().update(
      {
          "Autoformer>CorrLayer": CorrLayer,
          "Autoformer>CorrEncoderLayer": CorrEncoderLayer,
          "Autoformer>CorrEncoder": CorrEncoder,
          "Autoformer>CorrDecoderLayer": CorrDecoderLayer,
          "Autoformer>CorrDecoder": CorrDecoder,
          "Autoformer>CorrDecoderInit": CorrDecoderInit
      }
  )
