"""
Script to plot the abstract figures.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io as sio
import numpy as np

VAL_SPLIT = 0.3

sns.set_theme(context="talk",
              style="white",
              font="Roboto")

DATA_ROOT = "./data/"

PATIENT = 0
REALIZATION = 0

afpp_results = sio.loadmat(os.path.join(DATA_ROOT,
                                        "results",
                                        "autoformer++_results.mat"))

af_results = sio.loadmat(os.path.join(DATA_ROOT,
                                      "results",
                                      "autoformer_results.mat"))

inf_results = sio.loadmat(os.path.join(DATA_ROOT,
                                        "results",
                                        "informer_results.mat"))

lstm_results = sio.loadmat(os.path.join(DATA_ROOT,
                                        "results",
                                        "lstm_results.mat"))

split_idx = round((1 - VAL_SPLIT) * afpp_results["x"].shape[1])

afpp_x = np.mean(afpp_results["x"][:, split_idx:, ...], -2)
afpp_y_pred = np.mean(afpp_results["y_pred"][:, split_idx:, ...], -2)
afpp_y_true = np.mean(afpp_results["y_true"][:, split_idx:, ...], -2)

af_x = np.mean(af_results["x"][:, split_idx:, ...], -2)
af_y_pred = np.mean(af_results["y_pred"][:, split_idx:, ...], -2)
af_y_true = np.mean(af_results["y_true"][:, split_idx:, ...], -2)

inf_x = np.mean(inf_results["x"][:, split_idx:, ...], -2)
inf_y_pred = np.mean(inf_results["y_pred"][:, split_idx:, ...], -2)
inf_y_true = np.mean(inf_results["y_true"][:, split_idx:, ...], -2)

lstm_x = np.mean(lstm_results["x"][:, split_idx:, ...], -2)
lstm_y_pred = np.mean(lstm_results["y_pred"][:, split_idx:, ...], -2)
lstm_y_true = np.mean(lstm_results["y_true"][:, split_idx:, ...], -2)

assert np.array_equal(afpp_x, lstm_x) \
  and np.array_equal(af_x, lstm_x) \
  and np.array_equal(inf_x, lstm_x), "Update the benchmarks"

del afpp_results, af_results, inf_results, lstm_results

fig = plt.figure(figsize=(20, 10))
fig.set_tight_layout(True)
plt.autoscale(enable=True, axis="both", tight=True)
plt.plot(np.arange(afpp_y_true.shape[-1]),
         afpp_y_true[PATIENT, REALIZATION, :],
         ":",
         label="Ground Truth")
plt.plot(np.arange(afpp_y_true.shape[-1]),
         afpp_y_pred[PATIENT, REALIZATION, :],
         label="Autoformer++ Forecast")
plt.plot(np.arange(af_y_true.shape[-1]),
         af_y_pred[PATIENT, REALIZATION, :],
         "--",
         label="Autoformer Forecast")
plt.plot(np.arange(inf_y_true.shape[-1]),
         inf_y_pred[PATIENT, REALIZATION, :],
         "--",
         label="Informer Forecast")
plt.plot(np.arange(lstm_y_true.shape[-1]),
         lstm_y_pred[PATIENT, REALIZATION, :],
         "--",
         label="LSTM Forecast")
plt.legend()
plt.xlim([0,512])

os.makedirs("./figures/", exist_ok=True, mode=0o774)
fig.savefig(os.path.join("figures",
                         "results_talk.eps"))

print(f"Autoformer++ MSE: {
  np.mean((afpp_y_pred - afpp_y_true)**2)} ± {
    np.std((afpp_y_pred - afpp_y_true)**2)}")
print(f"Autoformer MSE: {
  np.mean((af_y_pred - af_y_true)**2)} ± {
    np.std((af_y_pred - af_y_true)**2)}")
print(f"Informer MSE: {
  np.mean((inf_y_pred - inf_y_true)**2)} ± {
    np.std((inf_y_pred - inf_y_true)**2)}")
print(f"LSTM MSE: {
  np.mean((lstm_y_pred - lstm_y_true)**2)} ± {
    np.std((lstm_y_pred - lstm_y_true)**2)}")
