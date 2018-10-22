#!/bin/bash

curl -O https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d ./pretrained_models
rm -f glove.6B.zip
