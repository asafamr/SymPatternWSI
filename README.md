### Word Sense Induction with Neural biLM and Symmetric Patterns

This repository contains reproducing code for the paper Word Sense Induction with Neural biLM and Symmetric Patterns by (Amrami and Goldberg 2018)

##### Usage instructions:
The code is written in python 3.6 and runs the SemEval 2013 task 13
evaluation code provided by the SemEval 2013 workshop, which is written in Java and therefore require installed JRE.

It also requires the AllenNLP python package as well as a few other packages.
Below are detailed instructions for running the code.


Prerequisites:

* Java >=1.6 installed and in PATH
* Python 3.6, preferably a clean Conda environment

Installing a clean conda environment from scratch:
```bash
# create and activate new python 3.6 conda envrionment
conda create -n spwsi python=3.6 anaconda
source activate spwsi

# for cuda support install pytorch (see link below and note cuda version)
conda install pytorch cuda80 -c pytorch

# additional python pacakges from requirements.txt
pip install -r requirements.txt

# install spacy English model, download ELMo's output matrix and SemEval 2013 task code
python -m spacy download en
sh download_resources.sh
```

 for pytorch installation w/ cuda support see instructions at [pytorch website](https://pytorch.org/)

finally, to run the WSI with ELMo's LM, run spwsi_elmo.py
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
                        number of clusters per instance
  --n-representatives N_REPRESENT
                        number of representations per sentence
  --n-samples-side N_SAMPLES_SIDE
                        number of samples per representations side
  --cuda CUDA           cuda device for ELMo (-1 to disable)
  --debug-dir DEBUG_DIR
                        logs and keys are written will be written to this dir
  --disable-lemmatization
                        disable ELMO prediction lemmatization
  --disable-symmetric-patterns
                        disable "x and y" symmetric pattern and predict
                        substitutes inplace
  --disable-tfidf       disable tfidf transformer
  --append-results-file APPEND_RESULTS_FILE
                        append final run results to this file
  --run-prefix RUN_PREFIX
                        will be prepended to log file names
  --elmo-batch-size ELMO_BATCH_SIZE
                        ELMo prediction batch size
  --prediction-cutoff PREDICTION_CUTOFF
                        ELMo predicted distribution top K cutoff
  --cutoff-elmo-vocab CUTOFF_ELMO_VOCAB
                        optimization: only use top K words for faster output
                        matrix multiplication
```

to run with cuda device 0
```bash
spwsi_elmo.py --cuda 0
```

a detailed report will be available inside "debug" directory after running the script together with the produced task key file.
you can re-run the key file scoring using evaluate_key.sh
