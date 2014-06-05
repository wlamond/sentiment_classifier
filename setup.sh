#!/bin/bash

VIRTENV="sentiment_classifier"

if [ ! -d "$VIRTENV" ]; then 
    virtualenv $VIRTENV
fi

source $VIRTENV/bin/activate
pip install -r requirements.txt

echo "select the stopwords package from the dialog"
python nltk_download.py

echo "run the following to enable the virtual env:"
echo "\$ source $VIRTENV/bin/activate"
echo "run the following to disable the virtual env:"
echo "\$ deactivate"
