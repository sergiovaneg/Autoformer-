"""
Hyperparameter Optimization 
"""

import os
import sys
import copy

import json
import argparse

import numpy as np
import scipy.io as sio

import optuna

import cortical_utils
from cortical_utils import Modality

parser = argparse.ArgumentParser(
    prog="Cortical Response - Autoformer++ hyperparameter optimization"
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

keras.utils.set_random_seed(0)

import jax  # pylint: disable=wrong-import-position
# pylint: disable=wrong-import-position
from keras_transformer import generic_transformer

DATA_ROOT = "./data/"
DATABASE = "sqlite:///" + DATA_ROOT + "studies.db"
STUDY_NAME = args.study

os.makedirs(os.path.join(DATA_ROOT, "models", STUDY_NAME),
            exist_ok=True, mode=0o774)

W_LEN = 9
RELEVANCE = 0.25

DS_FACTOR = 1
O = int(64 / DS_FACTOR)
STRIDE = round(np.log2(O))

EPOCHS = 200
PATIENCE = round(EPOCHS / 10)
VAL_SPLIT = 0.3

N_TRIALS = 1024


class KerasValidationPruner(keras.callbacks.Callback):
  def __init__(self, trial: optuna.Trial):
    super().__init__()
    self.trial = trial

  def on_epoch_end(self, epoch, logs=None):
    self.trial.report(logs["val_loss"], step=epoch)
    if self.trial.should_prune():
      self.trial.set_user_attr("r2", logs["val_r2_score"])
      raise optuna.TrialPruned()


class StoreBestCallback:
  """
  Outputs the current best results to a csv file in archival storage
  to visualize with Matlab in local
  """

  def __init__(self, angle_array: np.ndarray,
               comp_array: np.ndarray, l_output: int = O,
               modality: Modality = Modality.PREDICTION):
    self.angle_array = angle_array
    self.comp_array = comp_array
    self.l_output = l_output
    self.modality = modality

  def __call__(self, study_: optuna.study.Study,
               trial_: optuna.trial.FrozenTrial) -> None:
    if study_.best_trial.number == trial_.number:
      original_shape = self.comp_array.shape

      dataset = cortical_utils.array2io(
          angle=self.angle_array,
          comp=self.comp_array,
          l_input=trial_.params["I"],
          l_output=self.l_output,
          stride=self.l_output,
          modality=self.modality,
          preserve_all=True)

      generic_transformer.restore_custom_objects()
      model = keras.saving.load_model(
          os.path.join(DATA_ROOT, "models", STUDY_NAME,
                       "model_" + str(trial_.number) + ".keras"),
          compile=False
      )

      y_true = dataset[-1][..., -1].reshape(
          (*original_shape[:3], -1), order="C")
      x = dataset[:4] if self.modality is Modality.ESTIMATION \
          else [dataset[0], dataset[1], dataset[3]]
      y_pred = model.predict(
          x,
          batch_size=round(np.log(trial_.params.get("batch_size", 32))),
          verbose=2
      )[..., -1].reshape((*original_shape[:3], -1), order="C")
      x = dataset[-1][..., 0].reshape(
          (*original_shape[:3], -1), order="C")

      os.makedirs(os.path.join(DATA_ROOT, "results", STUDY_NAME),
                  exist_ok=True, mode=0o774)
      sio.savemat(os.path.join(DATA_ROOT, "results",
                               STUDY_NAME, f"model_{trial_.number}.mat"),
                  {"y_true": y_true, "y_pred": y_pred, "x": x,
                   "model_params": trial_.params})


class Objective:
  """
  Objective class storing the dataset
  """

  def __init__(self, angle_array: np.ndarray,
               comp_array: np.ndarray, l_output: int = O,
               stride: int = STRIDE,
               modality: Modality = Modality.PREDICTION,
               preserve_all: bool = True,
               study_name: str = STUDY_NAME):

    self.model_params = {
        "O": l_output, "d": 2, "d_out": 1,
        "output_components": False,
        "output_attention": False,
        "embed_type": "None",
        "manual_dec_input": modality == Modality.ESTIMATION
    }

    self.angle_array = angle_array
    self.comp_array = comp_array
    self.stride = stride
    self.modality = modality
    self.preserve_all = preserve_all

    with open("./study_parameters/" + study_name + ".json", "r",
              encoding=sys.getdefaultencoding()) as file:
      self.search_params: dict = json.load(file)

  @staticmethod
  def r2_score(y_true, y_pred):
    y_true = keras.ops.reshape(y_true, [-1, y_true.shape[-1]])
    y_pred = keras.ops.reshape(y_pred, [-1, y_pred.shape[-1]])
    numerator = keras.ops.sum(keras.ops.square(y_true - y_pred), axis=0)
    denominator = keras.ops.sum(keras.ops.square(
        y_true - keras.ops.mean(y_true, axis=0, keepdims=True)
    ), axis=0)
    return keras.ops.mean(1 - numerator / denominator)

  def _parse_key(self, trial: optuna.Trial,
                 param_type: str,
                 key: str, key_params: dict) -> int | float:
    match param_type:
      case "int":
        result = \
            trial.suggest_int(key, **key_params)
      case "discrete_uniform":
        result = \
            trial.suggest_discrete_uniform(key, **key_params)
      case "float":
        result = \
            trial.suggest_float(key, **key_params)
      case "uniform":
        result = \
            trial.suggest_uniform(key, **key_params)
    return result

  def __call__(self, trial: optuna.Trial) -> float:
    model_params = copy.deepcopy(self.model_params)
    search_params = copy.deepcopy(self.search_params)
    for key in search_params:
      if key in ["h", "tau"]:
        continue
      param_type = search_params[key].pop("type")
      model_params[key] = self._parse_key(trial, param_type, key,
                                          search_params[key])

    # Exception for multiple tau
    model_params["tau"] = []
    tau_type = search_params["tau"].pop("type")
    search_params["tau"]["high"] = min(model_params["I"],
                                       search_params["tau"]["high"])
    n_blocks = model_params.get("n_blocks", None) or \
        max(model_params.get("N"), model_params.get("M"))
    for idx in range(n_blocks):
      model_params["tau"].append(
          self._parse_key(trial, tau_type,
                          "tau_" + str(idx),
                          search_params["tau"])
      )
      search_params["tau"]["high"] = min(search_params["tau"]["high"],
                                         model_params["tau"][-1])

    # Exception for number of heads
    h_type = search_params["h"].pop("type")
    search_params["h"]["low"] = min(search_params["h"]["low"],
                                    model_params["d_model"])
    search_params["h"]["high"] = min(search_params["h"]["high"],
                                     model_params["d_model"])
    model_params["h"] = self._parse_key(trial, h_type, "h",
                                        search_params["h"])

    # Skip lazy trials
    if (("n_blocks" in model_params.keys())
        and (model_params["n_blocks"] == 1)) \
        or (("n_blocks" not in model_params.keys())
            and (model_params["N"] + model_params["M"] == 2)):
      raise optuna.TrialPruned()

    # Deterministic validation split
    split_idx = round((1 - VAL_SPLIT) * self.angle_array.shape[1])
    dataset_train = cortical_utils.array2io(
        angle=self.angle_array[:, :split_idx, ...],
        comp=self.comp_array[:, :split_idx, ...],
        l_input=model_params["I"],
        l_output=model_params["O"],
        stride=self.stride,
        modality=self.modality,
        preserve_all=self.preserve_all)
    dataset_val = cortical_utils.array2io(
        angle=self.angle_array[:, split_idx:, ...],
        comp=self.comp_array[:, split_idx:, ...],
        l_input=model_params["I"],
        l_output=model_params["O"],
        stride=self.stride,
        modality=self.modality,
        preserve_all=self.preserve_all)

    learning_rate = model_params.get("learning_rate", 1e-3)
    model = generic_transformer.create_autoformer_model(**model_params)

    trial.set_user_attr("Weight count", model.count_params())

    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[Objective.r2_score])
    if model_params["manual_dec_input"]:
      x_train = dataset_train[:4]
      x_val = dataset_val[:4]
    else:
      x_train = [dataset_train[0], dataset_train[1], dataset_train[3]]
      x_val = [dataset_val[0], dataset_val[1], dataset_val[3]]

    batch_size = model_params.get("batch_size", 32)
    history = model.fit(
        x=x_train,
        y=dataset_train[4][..., -1, None],
        batch_size=batch_size,
        validation_batch_size=round(np.log(batch_size)),
        epochs=EPOCHS,
        shuffle=False,
        validation_data=(x_val, dataset_val[4][..., -1, None]),
        callbacks=[keras.callbacks.TerminateOnNaN(),
                   keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                 mode="min"),
                   keras.callbacks.ModelCheckpoint(
            os.path.join(DATA_ROOT, "models", STUDY_NAME,
                         "model_" + str(trial.number) + ".keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        ),
            KerasValidationPruner(trial)],
        verbose=2)

    mse_idx = history.history["val_loss"].index(
        min(history.history["val_loss"]))
    mse = history.history["val_loss"][mse_idx]
    r2 = history.history["val_r2_score"][mse_idx]

    trial.set_user_attr("r2", r2)
    jax.clear_caches()

    return mse


if __name__ == "__main__":
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

  objective = Objective(angle, comp,
                        l_output=O,
                        preserve_all=False,
                        modality=Modality.ESTIMATION)
  output_cb = StoreBestCallback(angle, comp,
                                l_output=O,
                                modality=Modality.ESTIMATION)
  del angle, comp

  study = optuna.create_study(storage=DATABASE,
                              direction="minimize",
                              load_if_exists=True, study_name=STUDY_NAME,
                              pruner=optuna.pruners.PatientPruner(
                                  wrapped_pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                  patience=PATIENCE),
                              sampler=optuna.samplers.TPESampler(seed=0))
  for past_trial in study.trials:
    if past_trial.state == optuna.trial.TrialState.RUNNING:
      study.enqueue_trial(past_trial.params)
      study.tell(past_trial.number, state=optuna.trial.TrialState.FAIL)
  study.optimize(objective, n_trials=N_TRIALS, callbacks=[output_cb])

  print(f"Best MSE: {study.best_value}")
  best_trial = study.best_trial
  print("Best parameters: ")
  for param, value in best_trial.params.items():
    print(f"\t{param}: {value}")
