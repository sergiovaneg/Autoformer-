"""
Train similar networks to act as baseline for the Autoformer
"""

import os

import datetime

import optuna

import cortical_utils
from cortical_utils import Modality

import numpy as np
import scipy.io as sio

import argparse

parser = argparse.ArgumentParser(
    prog="Cortical Response - LSTM baseline"
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

DATA_ROOT = "./data/"
DATABASE = "sqlite:///" + DATA_ROOT + "studies.db"
STUDY_NAME = args.study

W_LEN = 9
RELEVANCE = 0.25

DS_FACTOR = 1
O = int(64 / DS_FACTOR)

EPOCHS = 200
PATIENCE = int(EPOCHS / 10)
VAL_SPLIT = 0.3

N_RUNS = 10

study = optuna.load_study(study_name=STUDY_NAME,
                          storage=DATABASE)
model_params = study.best_trial.params
model_params = {
    "I": model_params["I"],
    "d_model": model_params["d_model"],
    "dropout_rate": model_params["dropout_rate"],
    "d_ff": model_params["d_ff"],
    "n_blocks": model_params["M"] + model_params["N"],
    "batch_size": model_params["batch_size"]
}

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

# Input layers
layers = [
    keras.Input([model_params["I"], 2]),
    keras.layers.Conv1D(model_params["d_model"], 3,
                        padding="same", kernel_initializer="he_normal",
                        activation="leaky_relu")]

# Hidden layers
for _ in range(model_params["n_blocks"] - 1):
  dropout = keras.layers.Dropout(model_params["dropout_rate"])
  layers.extend([
      keras.layers.LSTM(model_params["d_model"], return_sequences=True),
      keras.layers.Dense(model_params["d_ff"], "softplus"),
      dropout,
      keras.layers.Dense(model_params["d_model"]),
      dropout,
  ])

# Output layers
dropout = keras.layers.Dropout(model_params["dropout_rate"])
layers.extend([
    keras.layers.LSTM(model_params["d_model"], return_sequences=False),
    keras.layers.Dense(model_params["d_ff"], "softplus"),
    dropout,
    keras.layers.Dense(model_params["d_model"]),
    dropout,
    keras.layers.Dense(1)
])

model = keras.Sequential(layers)

model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
model.summary()

if not os.path.exists(os.path.join(
    DATA_ROOT, "models",
    "lstm_baseline.weights.h5"
)):
  split_idx = round((1 - VAL_SPLIT) * angle.shape[1])
  x_train, _, _, _, y_train = cortical_utils.array2io(
      angle=angle[:, :split_idx, ...],
      comp=comp[:, :split_idx, ...],
      l_input=model_params["I"],
      l_output=1,
      stride=1,
      modality=Modality.PREDICTION,
      preserve_all=False)
  y_train = y_train[..., -1]
  x_val, _, _, _, y_val = cortical_utils.array2io(
      angle=angle[:, split_idx:, ...],
      comp=comp[:, split_idx:, ...],
      l_input=model_params["I"],
      l_output=1,
      stride=1,
      modality=Modality.PREDICTION,
      preserve_all=False)
  y_val = y_val[..., -1]

  model.fit(x=x_train, y=y_train,
            batch_size=model_params["batch_size"],
            epochs=EPOCHS,
            shuffle=False,
            validation_data=(x_val, y_val),
            callbacks=[keras.callbacks.TerminateOnNaN(),
                       keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                     mode="min"),
                       keras.callbacks.ModelCheckpoint(
                os.path.join(DATA_ROOT,
                             "models",
                             "lstm_baseline.weights.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min")],
            verbose=1)
  model.load_weights(
      os.path.join(DATA_ROOT,
                   "models",
                   "lstm_baseline.weights.h5"))
else:
  model.load_weights(
      os.path.join(DATA_ROOT,
                   "models",
                   "lstm_baseline.weights.h5"))
  model.summary()

original_shape = comp.shape
x, _, _, _, y = cortical_utils.array2io(
    angle=angle,
    comp=comp,
    l_input=model_params["I"],
    l_output=O,
    stride=O,
    modality=Modality.PREDICTION,
    preserve_all=True)

y_pred = np.zeros(y.shape[:-1])

times = np.zeros(N_RUNS + 1, dtype="timedelta64[ms]")
for idx in range(N_RUNS + 1):
  start = datetime.datetime.now(datetime.UTC)

  aux_x = x.copy()
  for t in range(O):
    y_pred[:, t] = model.predict(aux_x, batch_size=model_params["batch_size"],
                                 verbose=2).flatten()
    aux_x[:, :-1, :] = aux_x[:, 1:, :]
    aux_x[:, -1, :] = np.stack([y[:, t, 0], y_pred[:, t]], -1)

  times[idx] = np.timedelta64(datetime.datetime.now(datetime.UTC) - start, "ms")

# Skip warm-up: times=times[1:]
print(f"Average time: {
    np.mean(times[1:].astype(np.float64)).astype("timedelta64[ms]")
} ± {
    np.std(times[1:].astype(np.float64)).astype("timedelta64[ms]")
}")

y_pred = y_pred.reshape((*original_shape[:3], -1), order="C")
y_true = y[..., -1].reshape((*original_shape[:3], -1), order="C")
x = y[..., 0].reshape((*original_shape[:3], -1), order="C")

os.makedirs(os.path.join(DATA_ROOT, "results"),
            exist_ok=True, mode=0o774)
sio.savemat(os.path.join(DATA_ROOT, "results",
                         "lstm_results.mat"),
            {"y_true": y_true, "y_pred": y_pred, "x": x,
             "model_params": model_params})
