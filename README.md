sentiment_classifier
====================

A program for classifying sentiment in English text.


Setup
=====

First head to www.kaggle.com/c/sentiment-analysis-on-movie-reviews and download the data. You should get both the train.tsv.zip and test.tsv.zip files. Put those files in the data directory of this project.

Next, run clean_data.sh in the data directory. It'll unzip the training and test data and remove the headers. It will also shuffle the training data and generate a validation data set.

Finally, run setup.sh in the root of the project directory. This will set up a virtual env and install all the project dependencies in requirements.txt. A window will open from the NLTK installation. The only package required for this project is the stopwords package. setup.sh takes between 10 and 15 minutes to complete and requires an internet connection.

To test, first source the virtual env: 

	 $ source sentiment_classifier/bin/activate

then run:

	 $ python sentiment.py --validate

This will train and score the classifier on the validation set. The accuracy as of the writing of this README is ~60%. By default, the training phase uses all but one processor. You may specify the number of processors used at training time with the ncores parameter. For example, to run on 2 cores:

	 $ python sentiment.py --validate --ncores 2

You may train the classifier on the entire training set with the following: 

	 $ python sentiment.py --train

and generate a submission file for the competition with: 

	 $ python sentiment.py --test > submission.txt

Single examples can also be classified once the model is trained. Create a file with a sample per line, and run: 

	 $ python sentiment.py --sample < sample_file.txt

to classify each line from standard input.
