#!/usr/bin/python

'''
Python script to cross valiidate the best performing logistic regression and 
neural network models using different random samples of pseudo-negatives.
'''


# Import required modules/packages
import sys
import os
import os.path
import numpy as np
import pandas as pd
import random
import datetime

from itertools import chain

from nltk.stem.porter import PorterStemmer 
from nltk.stem.lancaster import LancasterStemmer 
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, SpatialDropout1D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model


#####
# Functions
#####

# Function which creates an embedding layer by extracting the GloVe vectors 
# associated with the words found in the input data
def make_embedding_layer(word_index, embeddings_index, initial):
	print('Preparing embedding matrix.')

	num_words = len(word_index) + 1

	# Embedding matrix is created, nrows = num_words, ncols = 100)
	if initial == 'rand': 
		embedding_matrix = np.random.normal(np.stack(embeddings_index.values()).mean(), np.stack(embeddings_index.values()).std(), (num_words, EMBEDDING_DIM))

	else:
		embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

	for word, i in word_index.items():
		if i == 400001:
			continue

		# Extract GloVe vector for identified word    
		embedding_vector = embeddings_index.get(word)

		if embedding_vector is not None:
			# Add GloVe vector to embedding matrix
			embedding_matrix[i] = embedding_vector

	# Set 0 index to zeros for padded values
	embedding_matrix[0] = np.zeros(EMBEDDING_DIM)

	# Load pre-trained word embeddings into an Embedding layer for use in machine
	# learning
	# Note that trainable = False to keep the embeddings fixed
	embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], 
								input_length=MAX_SEQUENCE_LENGTH,	trainable=False)

	return(embedding_layer)


# Function loading GloVe word vectors
def load_embeddings(filepath):
	import pickle
	import os.path

	print("Constructing embeddings index from glove data...")
	embeddings_index = {}
	f = open(filepath)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	# Return the GloVe embeddings
	return(embeddings_index)


# Function to stem words using a provided stemmer function
def stem_words(input_string, stemmer):
	# convert to unicode from utf-8
	word_list = [unicode(i, 'utf-8') for i in input_string.split(' ')]

	# Stem/Lemmatize words
	if stemmer == 'wnl':
		stem_list = [WordNetLemmatizer().lemmatize(i) for i in word_list]
	elif stemmer == 'porter':
		stem_list = [PorterStemmer(mode = 'NLTK_EXTENSIONS').stem(i) for i in word_list]
	elif stemmer == 'lancaster':
		stem_list = [LancasterStemmer().stem(i) for i in word_list]

	# Join strings
	stem_str = (' ').join(stem_list)

	# Convert back to utf-8??
	return stem_str.encode('utf8')


# Function to remove stop words from text strings
def rm_stop_words(input_string, stop_words):

	# Split input by spaces and remove words in stop_words
	word_list = [i for i in input_string.split(' ') if i not in stop_words]

	return (' ').join(word_list)


# Funtion to load and return the stop words contained in the specified file, 
# for use with CountVectorizer's stop_words member 
def read_stop_words(file):
	with open('../Data/' + file, 'r') as f:
		lines = [line.rstrip('\n') for line in f.readlines()]
	return frozenset(lines)


# Function to randomly split the data for use in logstic regression models
def split_data_lr(x, y, perc_train):
	# List indexes in x and y
	indexes = range(len(x))

	# Shuffle indexes
	random.seed(seed)
	random.shuffle(indexes)

	# Identify number of indexes to retain for training
	n_train = int(len(x)*perc_train)

	# Create training and testing data
	train_indexes = indexes[:n_train]
	test_indexes = indexes[n_train:]

	# Assign text to train or test data
	x_train = [x[i] for i in train_indexes]
	x_test = [x[i] for i in test_indexes]

	# Assign y values to train or test
	y_train = [y[i] for i in train_indexes]
	y_test = [y[i] for i in test_indexes]

	# Return training and test data
	return x_train, x_test, y_train, y_test


# Function to obtain the text data
def get_datasets(attribute, stop_words, stemmer, rand_seed):
	print 'Loading text data...'

	# Code to concat Title and Abstract if attribute == 'Title_Abstract'
	if attribute == 'TitleAbstract':
		p_df['TitleAbstract'] = p_df['Title'] + ' ' + p_df['Abstract']
		n_df['TitleAbstract'] = n_df['Title'] + ' ' + n_df['Abstract']
		
	positives = p_df[attribute].tolist()
	negatives = n_df[attribute].tolist()
	
	random.seed(rand_seed)
	negatives_samp = random.sample(negatives, len(positives))

	# Rm stop words
	if stop_words == 'NLTK_stop_words.txt' or stop_words == 'stop_words.txt':
		print 'Removing stop words...'
		my_stop_words = read_stop_words(stop_words)

		positives = [rm_stop_words(s, my_stop_words) for s in positives]
		negatives_samp = [rm_stop_words(s, my_stop_words) for s in negatives_samp]
	
	# Stem if needed
	if stemmer == 'wnl' or stemmer == 'porter' or stemmer == 'lancaster':
		print 'Stemming texts...'
		positives = [stem_words(s, stemmer) for s in positives]
		negatives_samp = [stem_words(s, stemmer) for s in negatives_samp]
	
	return {'positives': positives, 'negatives': negatives_samp}


# Function which loads the text data, prepares it for analysis, creates the word
# embedding layer and outputs data ready to be fed into neural net
def create_nn_data(attribute, embeddings_index, perc_train, stop_words, initial_vector):

	data = get_datasets(attribute = attribute, stop_words = stop_words, stemmer = 'None')

	raw_x_data = data['positives'] + data['negatives']

	print 'No. of positive documents: ' + str(len(data['positives']))
	print 'No. of negative documents: ' + str(len(data['negatives']))

	# Create y data
	y_data = np.append(np.ones(len(data['positives'])), np.zeros(len(data['negatives'])))

	# all text data ... 
	all_texts = data['positives'] + data['all_negatives'] + data['unclassifieds']

	emb_words = list(embeddings_index.keys())
	if initial_vector == 'remove':
		tok = Tokenizer(num_words = len(emb_words), filters = '')
		tok.fit_on_texts(emb_words)
	else:
		tok = Tokenizer(num_words = len(emb_words), filters = '', oov_token = len(emb_words)+1)
		tok.fit_on_texts(emb_words)
	sequences = tok.texts_to_sequences(raw_x_data)
	word_index = tok.word_index


	x_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	indices = range(x_data.shape[0])
	random.seed(seed)
	random.shuffle(indices)
	x_data = x_data[indices]
	y_data = y_data[indices]
	
	n_train = int(x_data.shape[0] * perc_train)

	x_train = x_data[:n_train]
	y_train = y_data[:n_train]

	x_test = x_data[n_train:] 
	y_test = y_data[n_train:]

	embedding_layer = make_embedding_layer(word_index, embeddings_index, initial_vector)

 	return x_train, y_train,  x_test, y_test, embedding_layer, tok # , data['unclassifieds']


 # Make a create_lr_data function
def create_lr_data(attribute, perc_train, stop_words, stemmer, weights, features, n_grams):

	data = get_datasets(attribute = attribute, stop_words = stop_words, stemmer = stemmer)

	raw_x_data = data['positives'] + data['negatives']

	print 'No. of positive documents: ' + str(len(data['positives']))
	print 'No. of negative documents: ' + str(len(data['negatives']))

	# Create y data
	y_data = np.append(np.ones(len(data['positives'])), np.zeros(len(data['negatives'])))

	# Split data for training and testing
	x_train, x_test, y_train, y_test = split_data_lr(x = raw_x_data, y = y_data, perc_train = 0.8)

	# Create vectorizer object
	if weights == 'tf-idf':
		if stop_words == 'over_0.85_df':
			vectorizer_obj = TfidfVectorizer(analyzer = features, ngram_range = n_grams, max_df = 0.85)
		else:
			vectorizer_obj = TfidfVectorizer(analyzer = features, ngram_range = n_grams)

	if weights == 'tf':
		if stop_words == 'over_0.85_df':
			vectorizer_obj = CountVectorizer(analyzer = features, ngram_range = n_grams, max_df = 0.85)
		else:
			vectorizer_obj = CountVectorizer(analyzer = features, ngram_range = n_grams)

	return x_train, y_train, x_test, y_test, vectorizer_obj


def resample_lr(out_df, dataset):
	# Load model specs...
	model_df = pd.read_csv("../Results/Model_metrics/LR/" + dataset + "_models_to_use.csv")

	# Only use best model - row 0
	# for row in range(0, model_df.shape[0]):
	print("Model: " +str(0))
	attrib = model_df.loc[0,"Attribute"]
	stop = model_df.loc[0,"Stop_words"]
	stemmer_type = model_df.loc[0,"Stemmer"]
	weight_type = model_df.loc[0,"Weighting"]
	feature_type = model_df.loc[0,"Feature"]
	n_grams = eval(str(model_df.loc[0, 'Ngram_range']))

	for rand_seed in range(2, 102):
		print("\tCross-validating lr model. Seed: " + rand_seed)
		# Create lr data
		x_train, y_train, x_test, y_test, vectorizer_obj = create_lr_data(attribute = attrib, 
																	perc_train = 0.8, 
																	stop_words = stop, 
																	stemmer = stemmer_type, 
																	weights = weight_type, 
																	features = feature_type, 
																	n_grams = n_grams)

		# straitified k-fold Cross-validation...
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

		date_time = str(datetime.datetime.now())

		n_train_docs = len(x_train)
		
		i = 1 

		for (train, test) in kfold.split(x_train, y_train):

			print '\tRunning fold: ' + str(i) + '/' + str(10) 

			date_time_start = datetime.datetime.now()

			x_val = map(x_train.__getitem__, test)
			y_val = map(y_train.__getitem__, test)

			x_train_tmp = map(x_train.__getitem__, train)
			y_train_tmp = map(y_train.__getitem__, train)

			vectorizer_obj.fit(x_train_tmp)
			x_train_vect = vectorizer_obj.transform(x_train_tmp)
			x_val_vect = vectorizer_obj.transform(x_val)

			# Train logistic classifier
			cv_model = LogisticRegression(penalty='l2', class_weight='balanced').fit(x_train_vect, y_train_tmp)
			
			# Record predictions for validation set
			cv_pred = cv_model.predict_proba(x_val_vect)[:,1].tolist()

			# Train on all x_train
			if i == 1:
				print'\tGenerating test set predictions...'
				# Vectorize full training set
				fitted_vect = vectorizer_obj.fit(x_train)
				x_train_full_vect = vectorizer_obj.transform(x_train)
				x_test_full_vect = vectorizer_obj.transform(x_test)

				# Use full training set to fit logistic
				# Train logistic classifier
				test_model = LogisticRegression(penalty='l2', class_weight='balanced').fit(x_train_full_vect, y_train)
		
				# Generate predictions for test data
				test_pred = test_model.predict_proba(x_test_full_vect)[:,1].tolist()

				n_test_docs = len(x_test)	

			else:
				test_pred = None
				y_test = None
				n_test_docs = None

			row_num = len(out_df)

			out_df.at[row_num, 'Attribute'] = attrib
			out_df.at[row_num, 'Seed_number'] = rand_seed
			out_df.at[row_num, 'N_training_docs'] = n_train_docs
			out_df.at[row_num, 'Classifier'] = 'LogisticRegression'
			out_df.at[row_num, 'Cost_function'] = 'l2'
			out_df.at[row_num, 'Stop_words'] = stop
			out_df.at[row_num, 'Stemmer'] = str(stemmer_type) 
			out_df.at[row_num, 'Weighting'] = str(weight_type)
			out_df.at[row_num, 'Feature'] = feature_type
			out_df.at[row_num, 'Ngram_range'] = str(n_grams)
			out_df.at[row_num, 'CV_folds'] = str(10)
			out_df.at[row_num, 'Fold'] = str(i)
			out_df.at[row_num, 'CV_Predicted_Probs'] = str(cv_pred)
			out_df.at[row_num, 'CV_True_Class'] = str(y_val)
			out_df.at[row_num, 'N_test_docs'] = str(n_test_docs)
			out_df.at[row_num, 'Test_Predicted_Probs'] = str(test_pred)
			out_df.at[row_num, 'Test_True_Class'] = str(y_test)
			out_df.at[row_num, 'Date_Time'] = date_time

			i += 1	

	return(out_df)


def resample_nn(out_df, dataset):

	# Load embedings
	embeddings_index = load_embeddings('../Data/glove_100d.txt')

	# Load model specs...
	model_df = pd.read_csv("../Results/Model_metrics/NN/" + dataset + "_models_to_use.csv")

	# for row in range(0, model_df.shape[0]):
	print("Model: " +str(0))
	attrib = model_df.loc[0,"Attribute"]
	stop = model_df.loc[0,"Stop_words"]
	init = model_df.loc[0,"Unknown_words"]

	for rand_seed in range(2, 102):
		print("\tCross-validating nn model. Seed: " + rand_seed)
		# Load and format the data to input into the algorithm/neural net
		x_train, y_train, x_test, y_test, embedding_layer, tokenizer = create_nn_data(attribute = attrib, 
																					embeddings_index = embeddings_index, 
																					perc_train = 0.8, 
																					stop_words = stop, 
																					initial_vector = init)

		# Identify number of texts in train and test
		n_train_docs = x_train.shape[0]
		n_test_docs = x_test.shape[0]

		# Function to create the neural net model
		def create_model():
			# Create a 1D convnet with global maxpooling
			sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
			embedded_sequences = embedding_layer(sequence_input)
			x = SpatialDropout1D(0.3)(embedded_sequences)
			x = Conv1D(128, 5, activation='relu')(x)
			x = MaxPooling1D(5)(x)
			x = SpatialDropout1D(0.3)(x)
			x = Conv1D(128, 5, activation='relu')(x)
			x = MaxPooling1D(5)(x)
			x = SpatialDropout1D(0.3)(x)
			x = Conv1D(128, 5, activation='relu')(x)
			x = GlobalMaxPooling1D()(x)
			x = Dense(128, activation='relu')(x)
			preds = Dense(1, kernel_initializer='normal', activation='sigmoid')(x)
	    	
			model = Model(sequence_input, preds)
			model.compile(loss='binary_crossentropy',
							optimizer='adam',
							metrics=['accuracy']) #, precision, recall]
			
			return model

	    # Construct a k-fold for cross validation on training data
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
		
		date_time = str(datetime.datetime.now())

		i = 1
		for (train, test) in kfold.split(x_train, y_train):
			print "\tRunning Fold " + str(i) + "/" + str(10)

			date_time_start = datetime.datetime.now()

			cv_model = None # Clearing the NN.
			cv_model = create_model()
			cv_model.fit(x_train[train], y_train[train], epochs=10, batch_size=10, verbose=0)
			
			cv_pred = list(chain.from_iterable(cv_model.predict(x_train[test]).tolist()))
			tmp_y_val = y_train[test].tolist()
			
			if i == 1:
				print'\tGenerating test set predictions...'
				test_model = None
				test_model = create_model()
				test_model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=0)
				
				test_pred = list(chain.from_iterable(test_model.predict(x_test).tolist()))
				tmp_y_test = y_test.tolist()

			else:
				test_pred = None
				tmp_y_test = None
				n_test_docs = None

			# Add info to df
			row_num = len(output_df)

			out_df.at[row_num, 'Attribute'] = attrib
			out_df.at[row_num, 'Seed_number'] = rand_seed
			out_df.at[row_num, 'N_training_docs'] = n_train_docs
			out_df.at[row_num, 'Classifier'] = 'Neural Network'
			out_df.at[row_num, 'Cost_function'] = 'Binary cross entropy'
			out_df.at[row_num, 'Stop_words'] = stop
			out_df.at[row_num, 'Unknown_words'] = init
			out_df.at[row_num, 'CV_folds'] = str(10)
			out_df.at[row_num, 'Fold'] = str(i)
			out_df.at[row_num, 'CV_Predicted_Probs'] = str(cv_pred)
			out_df.at[row_num, 'CV_True_Class'] = str(tmp_y_val)
			out_df.at[row_num, 'N_test_docs'] = str(n_test_docs)
			out_df.at[row_num, 'Test_Predicted_Probs'] = str(test_pred)
			out_df.at[row_num, 'Test_True_Class'] = str(tmp_y_test)
			out_df.at[row_num, 'Date_Time'] = date_time

			i += 1

return(out_df)


def resample_models(dataset):
	if dataset == "lpi":
		p_df = pd.read_csv("../Data/Text/prepared_lpi_positives.csv")
		n_df = pd.read_csv("../Data/Text/prepared_lpi_negatives.csv")

	elif dataset == "predicts":
		p_df = pd.read_csv("../Data/Text/prepared_predicts_positives.csv")
		n_df = pd.read_csv("../Data/Text/prepared_predicts_negatives.csv")


	lr_out_fp = "../Results/Model_metrics/LR/" + dataset + "_resample_metrics.csv"
	nn_out_fp = "../Results/Model_metrics/NN/" + dataset + "_resample_metrics.csv"

	lr_out_df = pd.DataFrame(columns = ['Attribute',
								'Random_seed',
								'N_training_docs',
								'Classifier',
								'Cost_function', 
								'Stop_words',
								'Stemmer',
								'Weighting',
								'Feature',
								'Ngram_range',
								'CV_folds',
								'Fold',
								'CV_Predicted_Probs',
								'CV_True_Class',
								'N_test_docs',
								'Test_Predicted_Probs',
								'Test_True_Class',
								'Date_Time'])

	nn_out_df = pd.DataFrame(columns = ['Attribute',
								'Random_seed',
								'N_training_docs',
								'Classifier',
								'Cost_function', 
								'Stop_words',
								'Unknown_words',
								'CV_folds',
								'Fold',
								'CV_Predicted_Probs',
								'CV_True_Class',
								'N_test_docs',
								'Test_Predicted_Probs',
								'Test_True_Class',
								'Date_Time'])

	lr_out_df = resample_lr(lr_out_df, dataset)
	nn_out_df = resample_nn(nn_out_df, dataset)

	print("Saving resampled metrics to file.")	
	lr_out_df.to_csv(lr_out_fp)
	nn_out_df.to_csv(nn_out_fp)


##
# Main Code
##

# for each dataset
resample_models("lpi")
resample_models("predicts")