"""
Auxiliary functions for the cortical dataset
"""

import os

from scipy import io as sio

import numpy as np
import scipy.signal as signal

from typing_extensions import Annotated
from pydantic import PositiveInt, PositiveFloat
from pydantic.functional_validators import AfterValidator

from enum import Enum


def is_odd(i: PositiveInt) -> PositiveInt:
  assert i % 2 == 1, "Not and odd number."
  return i


OddInt = Annotated[PositiveInt, AfterValidator(is_odd)]


class Modality(Enum):
  PREDICTION = 0
  ESTIMATION = 1


ROOT = "./data/"


def matfile2array(root: str = ROOT) -> tuple[np.ndarray, np.ndarray]:
  """
  Returns the dataset as a couple of 4D tensors of size:
  No. Subjects [S] x No. Realization [M] x No. Periods [P] x No. Samples [N]
  """
  mat_dict = sio.loadmat(os.path.join(root, "Benchmark_EEG_medium.mat"),
                         squeeze_me=True)
  angle = np.stack([participant["angle"].item()
                    for participant in mat_dict["EEGdata"]])
  comp = np.stack([participant["comp"].item()
                   for participant in mat_dict["EEGdata"]])
  return angle, comp


def preprocess_array(array: np.ndarray,
                     w_len: OddInt = 1,
                     relevance_factor: PositiveFloat = 1.,
                     downsample_factor: PositiveInt = 1,
                     normalize: bool = False) -> np.ndarray:
  """
  Filter the data in ``array`` both temporally (using a median filter
  parametrized by :param:``w_len``) and channel-wise (using a weighted
  average parametrized by :param:``relevance_factor``). Then, return a
  resampled version according to :param:``downsample_factor``.
  """
  filtered_array = signal.medfilt(array, (1, 1, 1, w_len))
  weighted_array = (1 - relevance_factor) * np.mean(
      filtered_array,
      axis=-2,
      keepdims=True
  ) + relevance_factor * filtered_array
  if normalize:
    mean, std = np.mean(weighted_array), np.std(weighted_array)
    weighted_array = (weighted_array - mean) / std
  return weighted_array[..., ::downsample_factor]


def array2io(angle: np.ndarray,
             comp: np.ndarray,
             l_input: PositiveInt,
             l_output: PositiveInt,
             stride: PositiveInt = 1,
             modality: Modality = Modality.PREDICTION,
             preserve_all: bool = True,
             seed: int = 0) -> tuple[np.ndarray, ...]:
  """
  Build the input tensors, with encoder/decoder sequenced length determined by
  ``l_input``/``l_output``; sequences are taken from strided
  windows (parametrized by ``stride``).

  When setting ``modality`` to PREDICTION, the decoder inputs are initialized
  as suggested in the original article; on the other hand, when setting it to
  ESTIMATION, the expected mechanical estimulus is passed as future context
  instead.

  Finally, ``preserve_all`` is a boolean that, if set to False, randomly
  samples the slices from the available periods (which are averaged for the
  small dataset, implying information redundancy) instead of keeping all slices
  from all periods. ``seed`` controls the random seed for the aforementioned
  sampling.
  """
  assert angle.shape == comp.shape
  x_enc_marks = np.arange(angle.shape[-1] - l_input - l_output,
                          step=stride, dtype="int32")[:, None] + \
      np.arange(l_input)[None, :]
  x_dec_marks = np.arange(l_input, angle.shape[-1] - l_output,
                          step=stride, dtype="int32")[:, None] + \
      np.arange(l_output)[None, :]
  if preserve_all:
    period_selection = np.arange(angle.shape[-2])[:, None, None]
    x_enc_marks = x_enc_marks[None, ...]
    x_dec_marks = x_dec_marks[None, ...]
  else:
    period_selection = np.random.default_rng(seed).integers(
        size=(x_enc_marks.shape[0], 1),
        low=0, high=angle.shape[-2])
  x_enc = np.stack((
      np.reshape(angle[..., period_selection, x_enc_marks], (-1, l_input)),
      np.reshape(comp[..., period_selection, x_enc_marks], (-1, l_input))
  ), axis=-1)

  y_dec = np.stack((
      np.reshape(angle[..., period_selection, x_dec_marks], (-1, l_output)),
      np.reshape(comp[..., period_selection, x_dec_marks], (-1, l_output))
  ), axis=-1)

  if modality is Modality.PREDICTION:
    x_dec = np.zeros_like(y_dec, dtype="float32")
  elif modality is Modality.ESTIMATION:
    x_dec = np.concatenate((
        np.reshape(angle[..., period_selection, x_dec_marks],
                   (-1, l_output, 1)),
        np.tile(
            np.mean(
                np.reshape(comp[..., period_selection,
                           x_enc_marks], (-1, l_input, 1)),
                axis=-2, keepdims=True
            ), (1, l_output, 1)
        )
    ), axis=-1)

  if preserve_all:
    x_enc_marks = np.tile(x_enc_marks[0, ...],
                          (np.prod(angle.shape[:3]), 1))
    x_dec_marks = np.tile(x_dec_marks[0, ...],
                          (np.prod(angle.shape[:3]), 1))
  else:
    x_enc_marks = np.tile(x_enc_marks,
                          (np.prod(angle.shape[:2]), 1))
    x_dec_marks = np.tile(x_dec_marks,
                          (np.prod(angle.shape[:2]), 1))

  return x_enc, x_enc_marks[:, :, None], x_dec, x_dec_marks[:, :, None], y_dec
