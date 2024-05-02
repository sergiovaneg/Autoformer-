"""
Export all data from the best results
"""

import os
import datetime

import optuna

import cortical_utils
from cortical_utils import Modality

import keras

import numpy as np
import scipy.io as sio

import argparse

parser = argparse.ArgumentParser(
    prog="Cortical Response - Autoformer++ post-processing"
)
parser.add_argument("-d", "--devices",
                    nargs="+", default=[0],
                    type=int, help="GPU to be used")
parser.add_argument("-s", "--study",
                    action="store", required=True,
                    type=str, help="Name of the study")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    [str(dev) for dev in args.devices])

os.environ["KERAS_BACKEND"] = "jax"
# pylint: disable=wrong-import-position
from keras_transformer import generic_transformer

DATA_ROOT = "./data/"
DATABASE = "sqlite:///" + DATA_ROOT + "studies.db"
STUDY_NAME = args.study

W_LEN = 9
RELEVANCE = 0.25

DS_FACTOR = 1
O = int(64 / DS_FACTOR)

N_RUNS = 10

# Load study
study = optuna.load_study(study_name=STUDY_NAME,
                          storage=DATABASE)
model_params = study.best_trial.params

# Get model
generic_transformer.restore_custom_objects()
model = keras.saving.load_model(
    os.path.join(DATA_ROOT, "models", STUDY_NAME,
                 "model_" + str(study.best_trial.number) + ".keras"),
    compile=False
)
model.summary()

angle, comp = cortical_utils.matfile2array(DATA_ROOT)
angle = cortical_utils.preprocess_array(array=angle,
                                        w_len=W_LEN,
                                        relevance_factor=RELEVANCE,
                                        downsample_factor=DS_FACTOR,
                                        normalize=True)
comp = cortical_utils.preprocess_array(array=comp,
                                       w_len=W_LEN,
                                       relevance_factor=RELEVANCE,
                                       downsample_factor=DS_FACTOR,
                                       normalize=True)

original_shape = comp.shape
dataset = cortical_utils.array2io(
    angle=angle,
    comp=comp,
    l_input=model_params["I"],
    l_output=O,
    stride=O,
    modality=Modality.ESTIMATION,
    preserve_all=True)

x_true = dataset[-1][..., 0].reshape(
    (*original_shape[:3], -1), order="C")
y_true = dataset[-1][..., -1].reshape(
    (*original_shape[:3], -1), order="C")
x = dataset[:4]

times = np.zeros(N_RUNS + 1, dtype="timedelta64[ms]")
for idx in range(N_RUNS + 1):
  start = datetime.datetime.now(datetime.UTC)
  y_pred = model.predict(x, batch_size=64, verbose=2)[..., -1].reshape(
      (*original_shape[:3], -1), order="C")
  times[idx] = np.timedelta64(datetime.datetime.now(datetime.UTC) - start, "ms")

# Skip warm-up: times=times[1:]
print(f"Average time: {
    np.mean(times[1:].astype(np.float64)).astype("timedelta64[ms]")
} ± {
    np.std(times[1:].astype(np.float64)).astype("timedelta64[ms]")
}")

os.makedirs(os.path.join(DATA_ROOT, "results", STUDY_NAME),
            exist_ok=True, mode=0o774)
sio.savemat(os.path.join(DATA_ROOT, "results",
                         "autoformer++_results.mat"),
            {"y_true": y_true, "y_pred": y_pred, "x": x_true,
             "model_params": model_params})
