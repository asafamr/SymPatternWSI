### Word Sense Induction with Neural biLM and Symmetric Patterns

This repository contains reproducing code for the results in Word Sense
Induction with Neural biLM and Symmetric Patterns paper

##### Usage instructions:
The code is written in python 3.6 and runs the SemEval 2013 task 13
evaluation code provided by the SemEval 2013 workshop, which is written
in Java and therefore require installed JRE.

It also requires the AllenNLP python package as well as a few other
packages.
Below are detailed instructions for running the code.


Prerequisites:

* Java >=1.6 installed and in PATH
* Python 3.6, preferably a clean Conda environment

Installing a clean conda environment from scratch:
```bash
# create and activate new python 3.6 conda envrionment
conda create -n spwsi python=3.6 anaconda
source activate spwsi

# for cuda support install pytorch first (see link below and note the cuda version!)
# in our system a full run w/ cuda took ~2 minutes. w/o ~20 minutes
# if you don't mind that,
# you can skip next line and run spwsi_elmo.py with arguments: --cuda -1
conda install pytorch cuda80 -c pytorch

# additional python pacakges from requirements.txt
pip install -r requirements.txt

# install spacy English model, download ELMo's output matrix and SemEval 2013 task code
python -m spacy download en
sh download_resources.sh
```

for pytorch installation w/ cuda support see instructions at
[pytorch website](https://pytorch.org/)

finally, to do the actual WSI using ELMo's LM, run spwsi_elmo.py

script arguments:
```bash
python spwsi_elmo.py -h
usage: spwsi_elmo.py [-h] [--n-clusters N_CLUSTERS]
                     [--n-representatives N_REPRESENT]
                     [--n-samples-side N_SAMPLES_SIDE] [--cuda CUDA]
                     [--debug-dir DEBUG_DIR] [--disable-lemmatization]
                     [--disable-symmetric-patterns] [--disable-tfidf]
                     [--append-results-file APPEND_RESULTS_FILE]
                     [--run-prefix RUN_PREFIX]
                     [--elmo-batch-size ELMO_BATCH_SIZE]
                     [--prediction-cutoff PREDICTION_CUTOFF]
                     [--cutoff-elmo-vocab CUTOFF_ELMO_VOCAB]

BiLM Symmetric Patterns WSI Demo

optional arguments:
  -h, --help            show this help message and exit
  --n-clusters N_CLUSTERS
                        number of clusters per instance (default: 8)
  --n-representatives N_REPRESENT
                        number of representations per sentence (default: 20)
  --n-samples-side N_SAMPLES_SIDE
                        number of samples per representations side (default:
                        4)
  --cuda CUDA           cuda device for ELMo (-1 to disable) (default: 0)
  --debug-dir DEBUG_DIR
                        logs and keys are written will be written to this dir
                        (default: debug)
  --disable-lemmatization
                        disable ELMO prediction lemmatization (default: False)
  --disable-symmetric-patterns
                        disable "x and y" symmetric pattern and predict
                        substitutes inplace (default: False)
  --disable-tfidf       disable tfidf transformer (default: False)
  --append-results-file APPEND_RESULTS_FILE
                        append final run results to this file (default: None)
  --run-prefix RUN_PREFIX
                        will be prepended to log file names (default: )
  --elmo-batch-size ELMO_BATCH_SIZE
                        ELMo prediction batch size (optimization only)
                        (default: 50)
  --prediction-cutoff PREDICTION_CUTOFF
                        ELMo predicted distribution top K cutoff (default: 50)
  --cutoff-elmo-vocab CUTOFF_ELMO_VOCAB
                        optimization: only use top K words for faster output
                        matrix multiplication (default: 50000)
```
The default parameters specified above are used in the paper


to run on cuda device 0
```bash
python spwsi_elmo.py --cuda 0
```

A detailed execution report will be available inside "debug" directory
after running the script.

Additionally, a key file, whose name is ending with '.key', will be
created inside the debug dir.
This file will contain a mapping from the task dataset entries to their
induced senses. The SemEval task code evaluate this mapping and produce
two metric scores: FNMI and FBC as descibed in the paper.
You can re-run this key file evaluation using evaluate_key.sh

Developed and tested on Ubuntu 16, Cuda 8.0

### Results:
TODO:add
