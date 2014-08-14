import argparse
import string
import numpy
import sys
import re
import os
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from nltk.stem.snowball import SnowballStemmer

training_file      = 'data/cleaned_train.tsv'
test_file          = 'data/cleaned_test.tsv'
test_output_header = 'PhraseId,Sentiment\n'

validate_training_file = 'data/xaa'
validate_test_file     = 'data/xab'

class Sentiment_Classifier:


    def __init__(self, training=False, validate=False, require_dense=False, ncores=-2):

        self.columns_per_training_example = 4
        self.columns_per_test_example     = 3

        self.require_dense = require_dense
        self.ncores = ncores

        self.model_pickle_file = None
        self.transformer_pickle_file = None
        self.kernel_sampler_file = None

        self._setup_pickle_files()

        # used to filter
        self._stemmer = SnowballStemmer('english')

        if (training or validate):
            self._setup_classifier_and_transformer()
        else:
            self._load_model_and_transformer()


    def _setup_pickle_files(self):
        """ set up the directory for the model and transformer to be stored in once trained. """
        pickle_dir = 'pickled_objects'
        d = os.path.dirname(pickle_dir)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        self.model_pickle_file = pickle_dir + '/model.pkl'
        self.transformer_pickle_file = pickle_dir + '/transformer.pkl'


    def _store_model_and_transformer(self):
        joblib.dump(self.classifier, self.model_pickle_file)
        joblib.dump(self.transformer, self.transformer_pickle_file)


    def _load_model_and_transformer(self):
        self.classifier = joblib.load(self.model_pickle_file)
        self.transformer = joblib.load(self.transformer_pickle_file)


    def _setup_classifier_and_transformer(self):
        self.transformer = TfidfVectorizer(use_idf=False, decode_error='ignore', ngram_range=(1,3))
        self.classifier = OneVsRestClassifier(LogisticRegression(), n_jobs=self.ncores)


    def _write_message(self, msg):
        sys.stderr.write(msg + '\n')


    def _filter(self, sentence):
        sentence_list = sentence.split()
        sentence_list = map(lambda x: self._stemmer.stem(x), sentence_list)
        return ' '.join(sentence_list)

    def _fit_transform(self, X):
        return self.transformer.fit_transform(X)

    def _transform(self, X):
        return self.transformer.transform(X)
    
    ''' Get features related to word lengths. Counts of words of each length.
	Max word length, min word length, ratio of words to sentence length '''
    def _word_len_features(self, sentence):
	word_lengths = [len(word) for word in sentence.split()]
	if len(word_lengths) == 0:
	    # return 0 for each feature
            return [0] * 12
	else:
	    # add arbitrary counts up to size 10 (up to 20 is actually better,
	    # but we should probably come up with a better way than arbitrary counts,
	    # larger range buckets perhaps)
	    len_counts = [0] * 9
	    for i in range(1,10):
		len_counts[i-1] = word_lengths.count(i)
	    len_counts.extend([sum(word_lengths)/float(len(sentence)), \
	    max(word_lengths), min(word_lengths)])
	    return len_counts   
 
    def _get_extra_features(self, sentence):
	sentence_len = float(len(sentence))
	get_count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
	digits_count =  get_count(sentence, '0123456789')
	# punctuation count didn't help, pehaps indvidual punctuation count will
	#punct_count = get_count(sentence, string.punctuation)
	
	features = [sentence_len, 
		    sum(1 for c in sentence if c.isupper())/float(sentence_len),
		   digits_count/sentence_len]
        features.extend(self._word_len_features(sentence))
	return features
   
    def get_features_and_labels(self, training_file):
        self._write_message('reading data')
        training_examples = [(phrase_id, sentence_id, self._filter(sentence), 
			    self._get_extra_features(sentence), 
			    sentiment)
                            for phrase_id, sentence_id, sentence, sentiment
                            in self._read_file(training_file, self.columns_per_training_example)]

        self._write_message('generating mapped data')
        phrase_ids, sentence_ids, sentences, extra_features, y = zip(*training_examples)
        return sentences, extra_features, y


    def get_features_and_ids(self, data_file):
        self._write_message('reading data')
        examples = [(phrase_id, sentence_id, self._filter(sentence))
                    for phrase_id, sentence_id, sentence
                    in self._read_file(data_file, self.columns_per_test_example)]

        self._write_message('generating mapped data')
        phrase_ids, sentence_ids, sentences = zip(*examples)
        X = self._transform(sentences)
        return phrase_ids, X

    def _train(self, X, y):
        self.classifier.fit(X, y)


    def train(self, training_file):
        """ train the model """
        X, extra_features, y = self.get_features_and_labels(training_file)
        X = self._fit_transform(X)
        sparse_features = sparse.csr_matrix(numpy.array(extra_features))
        X = sparse.hstack((X, sparse_features))	
	if self.require_dense:
            X = X.toarray()
        #X = self.kernel.fit_transform(X)
        self._write_message('training model')
        self._train(X, y)
        # save the classifier for later!
        self._store_model_and_transformer()


    def validate(self, validate_file):
        X, extra_features, y = self.get_features_and_labels(validate_file)
	X = self._transform(X)
        sparse_features = sparse.csr_matrix(numpy.array(extra_features))
        X = sparse.hstack((X, sparse_features))	        
	if self.require_dense:
            X = X.toarray()
        #X = self.kernel.transform(X)
        self._write_message('validate model')
        print self._score(X, y)


    def _score(self, X, y):
        """ score the model """
        score = self.classifier.score(X, y)
        return score


    def _predict(self, X):
        """ predict a single example """
        y = self.classifier.predict(X)
        return y


    def test(self, test_file):
        """ generate the submission file. """
        self._write_message('predicting test outcomes')
        ids, X = self.get_features_and_ids(test_file)
        if self.require_dense:
            X = X.toarray()
        #X = self.kernel.transform(X)
        y = self._predict(X)
        self.write_output(ids, y)


    def classify_string(self):
        """ Classify lines from stdin """
        for s in sys.stdin:
            X = self.transformer.transform([s])
            self._write_line(self._predict(X)[0])


    def _write_line(self, s):
        sys.stdout.write(str(s) + '\n')


    def write_output(self, ids, y):
        """ write the result of the test method. """
        # write the new predictions and the IDs to stdout
        sys.stdout.write(test_output_header)
        for i in xrange(len(ids)):
            self._write_line(str(ids[i]) + ',' + str(y[i]))


    def _read_file(self, filename, expected_elements):
        """ generator that reads lines from the given file
        and appends missing data as needed """
        with open(filename, 'r') as f:
            for line in f:
                t = tuple(line.strip().split('\t'))
                if len(t) != expected_elements:
                    t = t + ('',)
                yield t


def main():

    args = argparse.ArgumentParser()
    args.add_argument('--train', action='store_true')
    args.add_argument('--test', action='store_true')
    args.add_argument('--validate', action='store_true')
    args.add_argument('--sample', action='store_true')
    args.add_argument('--ncores', type=int, default=-2)
    args = args.parse_args()

    # pass test flag in so the constructer can load the
    # model and transformer. It doesn't need to do that for training
    model = Sentiment_Classifier(training=args.train, validate=args.validate, ncores=args.ncores)

    if args.train:
        model.train(training_file)

    if args.test:
        model.test(test_file)

    if args.validate:
        model.train(validate_training_file)
        model.validate(validate_test_file)

    if args.sample:
        model.classify_string()

if __name__ == '__main__':
    main()
