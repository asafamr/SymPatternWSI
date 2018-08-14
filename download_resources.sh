#!/usr/bin/env bash
mkdir -p resources
echo downloading SemEval 2013 task 13 data and evaluation code...
wget -N https://www.cs.york.ac.uk/semeval-2013/task13/data/uploads/semeval-2013-task-13-test-data.zip -P resources

# unzip might not be installed
python - <<EOF
import zipfile
with zipfile.ZipFile("./resources/semeval-2013-task-13-test-data.zip","r") as zip_ref:
    zip_ref.extractall('resources')
EOF

echo downloading ELMo output matrix and vocublary...
wget -N https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5 -P resources
wget -N https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/vocab-2016-09-10.txt -P resources
