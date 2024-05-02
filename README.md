# Autoformer++

Code used to generate the results in "Multi-Window Autoformer for Dynamic Systems Modelling (Autoformer ++)" for the 8th Edition of the Workshop on Nonlinear System Identification Benchmarks.

The architecture is derived from the [Autoformer](https://github.com/thuml/Autoformer) \([Wu, 2021](https://doi.org/10.48550/arXiv.2106.13008)\), but implemented in Keras with the Jax Backend (should also compile with Tensorflow/Pytorch, but not tested), and improves it by allowing multiple periodicity windows across the Encoder/Decoder stacks, as well as future control signals passed to the Decoder.

The [slides](./Multi_Window_Autoformer_for_Dynamic_Systems_Modelling.pdf) used during the presentation are also available in the repo.

## Usage

0. Set up the Python environment; [`keras_jax.yml`](./keras_jax.yml) provides a Conda env descriptor that can be used to automatically generate the virtual environment using `conda env create --file keras_jax.yml`
1. Download the [medium-sized dataset](https://drive.google.com/file/d/1XrkV43ZKq-vlcwVz5OUiW2A2N51Bz5Lt/view?usp=sharing) \([Vlaar, 2017](https://doi.org/10.1109/TNSRE.2017.2751650)\) and place `Benchmark_EEG_medium.mat` in the [`data`](./data/) directory
2. Run the [hyperparameter optimization](./cortical_hyperopt.py) using `nohup python cortical_hyperopt.py -s "study_workshop" > cortical_hyperopt_output.txt &` to monitor the progress without locking the terminal
3. Process the best model using `nohup python process_best.py -s "study_workshop" > best_output_timed.txt &`
4. Generate the baselines using:
    - `nohup python autoformer_baseline.py -s "study_workshop" > autoformer_baseline_timed.txt &`
    - `nohup python informer_baseline.py -s "study_workshop" > informer_baseline_timed.txt &`
    - `nohup python lstm_baseline.py -s "study_workshop" > lstm_baseline_timed.txt &`
5. Generate the plots and numerical results using [`plot_results.py`](./plot_results.py) (No arguments)

Additional hyperparameter spaces can be searched by creating a *JSON* file following the example in [`study_workshop.json`](./study_parameters/study_workshop.json) and placing it in the [`study_parameters`](./study_parameters/) directory.

## Citation

If you find this repo useful, please cite our paper.

```bibtex
@conference{Vanegas2024Autoformer++,
  title = {Multi-Window Autoformer for Dynamic Systems Modelling},
  author = {Sergio Vanegas and Lasse Lensu and Fredy Ruiz},
  year = 2024,
  month = {April},
  booktitle= {Book of Abstracts - Workshop on Nonlinear System Identification Benchmarks},
  publisher = {Dalle Molle Institute for Artificial Intelligence},
  address = {Lugano, Switzerland},
  pages = {25},
  editor = {Dario Piga and Marco Forgione and Maarten Schoukens},
  organization = {LUT University}
}
```
