#!/bin/sh

TESTFILE="test.tsv"
TRAINFILE="train.tsv"

# unzip!
`unzip $TESTFILE.zip`
`unzip $TRAINFILE.zip`

# get sizes of files
TESTLEN=`wc -l $TESTFILE | awk '{ print $1 }'`
TRAINLEN=`wc -l $TRAINFILE | awk '{ print $1 }'`

# get rid of headers and shuffle the training data
`tac $TESTFILE | head -n $((TESTLEN - 1)) > cleaned_$TESTFILE`
`tac $TRAINFILE | head -n $((TRAINLEN - 1)) | shuf > cleaned_$TRAINFILE`

# generate validation data (2/3-1/3 split)
`split -l $((TRAINLEN * 2/3)) cleaned_$TRAINFILE`
