# CATE Benchmark
A testing platform to assess the performance of CATE estimators across popular datasets.

## Installation
The easiest way to replicate the running environment is through [Anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda). Once installed, follow the steps below.


1. Download the repo.
2. Enter the directory (i.e. `cd cate-benchmark`).
3. Run the following command to recreate the 'cate-bench' conda environment:

`conda env create -f environment.yml`

4. Download datasets from [here](https://essexuniversity.box.com/s/69hiufo5cejvjux7a6zrsie5v7fd5s8o). Once downloaded, extract them to 'datasets' directory.

## Usage
The 'experiments' folder contains a few example scripts that run the code, ranging from basic to more advanced. To learn more about available script parameters, see the contents of 'main.py'.

By default, any files created as part of running the scripts are saved under 'results'.

## Analysing results
A separate directory is created per each selected estimator when running the scripts to store various results. The following result files are usually created:
- info.log (intermediate results as the script is being executed)
- scores.csv (final scores per relevant metric)
- times.csv (training + prediction time in seconds consumed by an estimator)

In addition, it is possible to get a summary of multiple estimators in a single table. This can be done via the 'results/process.py' script, which in turn produces 'combined.csv' file. For an example usage, see some of the existing running scripts.

## Adding other estimators
The code can be easily extended to use more estimators.

1. Go to 'main.py'.
2. Edit 'get_parser()': add new key to 'estimation_model'.
3. Edit '_get_model()': using the new key, return an instance of your model.
4. Edit 'estimate(): train your model on the data and provide predictions.

## Other projects
Projects using the CATE benchmark:
- [Undersmoothing Data Augmentation](https://github.com/misoc-mml/undersmoothing-data-augmentation)
