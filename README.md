# Autoformer++

Code used to generate the results in "Multi-Window Autoformer for Dynamic Systems Modelling (Autoformer ++)" for the 8th Edition of the Workshop on Nonlinear System Identification Benchmarks.

Usage:

0. Set up the Python environment; [`keras_jax.yml`](./keras_jax.yml) provides a Conda env descriptor that can be used to automatically generate the virtual environment using `conda env create --file keras_jax.yml`
1. Download the [medium-sized dataset](https://drive.google.com/file/d/1XrkV43ZKq-vlcwVz5OUiW2A2N51Bz5Lt/view?usp=sharing) and place `Benchmark_EEG_medium.mat` in the *data* directory
2. Run the [hyperparameter optimization](./cortical_hyperopt.py) using `nohup python cortical_hyperopt.py -s "study_workshop" > cortical_hyperopt_output.txt &` to monitor the progress without locking the terminal
3. Process the best model using `nohup python process_best.py -s "study_workshop" > best_output_timed.txt &`
4. Generate the baselines using:
    - `nohup python autoformer_baseline.py -s "study_workshop" > autoformer_baseline_timed.txt &`
    - `nohup python informer_baseline.py -s "study_workshop" > informer_baseline_timed.txt &`
    - `nohup python lstm_baseline.py -s "study_workshop" > lstm_baseline_timed.txt &`
5. Generate the plots and numerical results using [`plot_results.py`](./plot_results.py) (No arguments)

Additional hyperparameter spaces can be searched by creating a json file following the example in [`study_workshop.json`](./study_parameters/study_workshop.json) and placing it in the [`study_parameters`](./study_parameters/) directory.
