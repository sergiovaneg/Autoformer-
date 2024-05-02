"""
Train similarly configured Autoformer to act as baseline for the modifications
"""

import os

import datetime

import numpy as np

import optuna

import cortical_utils
from cortical_utils import Modality

import scipy.io as sio

import argparse

parser = argparse.ArgumentParser(
    prog="Cortical Response - Autoformer baseline"
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
import keras  # pylint: disable=wrong-import-position
# pylint: disable=wrong-import-position
from keras_transformer import generic_transformer

DATA_ROOT = "./data/"
DATABASE = "sqlite:///" + DATA_ROOT + "studies.db"
STUDY_NAME = args.study

W_LEN = 9
RELEVANCE = 0.25

DS_FACTOR = 1
O = round(64 / DS_FACTOR)
STRIDE = round(np.log2(O))

EPOCHS = 200
PATIENCE = round(EPOCHS / 10)
VAL_SPLIT = 0.3

N_RUNS = 10

study = optuna.load_study(study_name=STUDY_NAME,
                          storage=DATABASE)
model_params = study.best_trial.params
model_params.update({
    "O": O, "d": 2, "d_out": 1,
    "output_components": False,
    "output_attention": False,
    "embed_type": "None",
    "manual_dec_input": False
})

model_params["tau"] = model_params[f"tau_{max(model_params["N"],
                                              model_params["M"]) - 1}"]

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

model = generic_transformer.create_autoformer_model(**model_params)

model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
model.summary()

if not os.path.exists(os.path.join(
    DATA_ROOT,
    "autoformer_baseline.weights.h5"
)):
  split_idx = round((1 - VAL_SPLIT) * angle.shape[1])
  x_enc_train, xm_enc_train, _, xm_dec_train, y_train = cortical_utils.array2io(
      angle=angle[:, :split_idx, ...],
      comp=comp[:, :split_idx, ...],
      l_input=model_params["I"],
      l_output=64,
      stride=STRIDE,
      modality=Modality.PREDICTION,
      preserve_all=False)
  x_enc_val, xm_enc_val, _, xm_dec_val, y_val = cortical_utils.array2io(
      angle=angle[:, split_idx:, ...],
      comp=comp[:, split_idx:, ...],
      l_input=model_params["I"],
      l_output=64,
      stride=STRIDE,
      modality=Modality.PREDICTION,
      preserve_all=False)

  model.fit(x=[x_enc_train, xm_enc_train, xm_dec_train],
            y=y_train[..., -1, None],
            batch_size=model_params["batch_size"],
            epochs=EPOCHS,
            shuffle=False,
            validation_data=([x_enc_val, xm_enc_val, xm_dec_val],
                             y_val[..., -1, None]),
            callbacks=[keras.callbacks.TerminateOnNaN(),
                       keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                     mode="min"),
                       keras.callbacks.ModelCheckpoint(
                os.path.join(DATA_ROOT,
                             "models",
                             "autoformer_baseline.weights.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min")],
            verbose=1)
  model.load_weights(
      os.path.join(DATA_ROOT,
                   "models",
                   "autoformer_baseline.weights.h5"))
else:
  model.load_weights(
      os.path.join(DATA_ROOT,
                   "models",
                   "autoformer_baseline.weights.h5"))
  model.summary()

original_shape = comp.shape
x_enc, xm_enc, _, xm_dec, y = cortical_utils.array2io(
    angle=angle,
    comp=comp,
    l_input=model_params["I"],
    l_output=O,
    stride=O,
    modality=Modality.PREDICTION,
    preserve_all=True)

times = np.zeros(N_RUNS + 1, dtype="timedelta64[ms]")
for idx in range(N_RUNS + 1):
  start = datetime.datetime.now(datetime.UTC)
  y_pred = model.predict([x_enc, xm_enc, xm_dec],
                         batch_size=64, verbose=2)
  times[idx] = np.timedelta64(datetime.datetime.now(datetime.UTC) - start, "ms")

# Skip warm-up: times=times[1:]
print(f"Average time: {
    np.mean(times[1:].astype(np.float64)).astype("timedelta64[ms]")
} ± {
    np.std(times[1:].astype(np.float64)).astype("timedelta64[ms]")
}")

y_pred = y_pred[..., -1].reshape((*original_shape[:3], -1), order="C")
y_true = y[..., -1].reshape((*original_shape[:3], -1), order="C")
x = y[..., 0].reshape((*original_shape[:3], -1), order="C")

os.makedirs(os.path.join(DATA_ROOT, "results"),
            exist_ok=True, mode=0o774)
sio.savemat(os.path.join(DATA_ROOT, "results",
                         "autoformer_results.mat"),
            {"y_true": y_true, "y_pred": y_pred, "x": x,
             "model_params": model_params})
