#!/usr/bin/python

'''Python code to cross-validate best performing lr/nn models on PREDICTS data.

Inputs:		4 csv files, positives, negatives, lr model specification and cnn 
			model specification.
Outputs:	Model files
			Metric files
			Vectorizer files (for use with LR models)
			Tokenizer files (for use with NN models)
'''

# Import packages

import os
import glob
import pandas as pd
import random
import datetime
import numpy as np
import re

from itertools import chain

from nltk.stem.porter import PorterStemmer 
from nltk.stem.lancaster import LancasterStemmer 
from nltk.stem import WordNetLemmatizer

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

from sklearn.externals import joblib

#####
# Functions
#####

# Function for saving nn models and weights for future reference
def save_nn_model(model_object, tokenizer_object, model_num, attribute, stop_words, unknown_words, date_time):

	model_dir = '../Results/Models/NN'
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	model_fp = '../Results/Models/NN/model_predicts_' + str(model_num) + '_' + str(attribute) + '_' + str(stop_words).rstrip('.txt') + '_' + str(unknown_words) +  '_' + str(date_time) + '.h5'
	tokenizer_fp = '../Results/Models/NN/tokenizer_predicts_' +str(model_num) + '_' + str(attribute) + '_' + str(stop_words).rstrip('.txt') + '_' + str(unknown_words) +  '_' + str(date_time) + '.pkl'
	
	model_object.save(model_fp)
	print '\tSaved model...'
	joblib.dump(tokenizer_object, tokenizer_fp, compress = 1)
	print '\tSaved tokenizer...'


# Function for pickling lr models and vectorizers and saving them for future reference
def save_lr_model(model_object, vectorizer_object, model_num, attribute, stop_words, stemmer, weighting, feature, ngrams, date_time):

	model_dir = '../Results/Models/LR'
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	model_fp = '../Results/Models/LR/model_predicts_' + str(model_num) + '_' + str(attribute) + '_' + str(stop_words).rstrip('.txt') + '_' + str(stemmer) + '_' + str(weighting) + '_' + str(feature) + '_' + str(ngrams) + '_' + str(date_time) + '.pkl'
	vect_fp = '../Results/Models/LR/vectorizer_predicts_' + str(model_num) + '_' + str(attribute) + '_' + str(stop_words).rstrip('.txt') + '_' + str(stemmer) + '_' + str(weighting) + '_' + str(feature) + '_' + str(ngrams) + '_' + str(date_time) + '.pkl'
		
	joblib.dump(model_object, model_fp, compress = 1)
	print '\tSaved model...'
	joblib.dump(vectorizer_object, vect_fp, compress = 1)
	print '\tSaved vectorizer...'


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
	print("done.")

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
	else:
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


# 
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


# Function to remove unknown words from text strings
def rm_unknown_words(input_string, intersect_words):

	# Split input by spaces and remove words in stop_words
	word_list = [i for i in input_string.split(' ') if i in intersect_words]

	return (' ').join(word_list)


# Function to obtain the text data
def get_datasets(attribute, stop_words, stemmer):
	print 'Loading text data...'

	# Code to concat Title and Abstract if attribute == 'Title_Abstract'
	if attribute == 'TitleAbstract':
		p_df['TitleAbstract'] = p_df['Title'] + ' ' + p_df['Abstract']
		n_df['TitleAbstract'] = n_df['Title'] + ' ' + n_df['Abstract']
		
	positives = p_df[attribute].tolist()
	negatives = n_df[attribute].tolist()
	
	random.seed(seed)
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
def create_data(attribute, embeddings_index, perc_train, stop_words, initial_vector):

	data = get_datasets(attribute, stop_words, stemmer = 'None')

	raw_x_data = data['positives'] + data['negatives']

	print 'Positive documents: ' + str(len(data['positives']))
	print 'Negative documents: ' + str(len(data['negatives']))

	# Create y data
	y_data = np.append(np.ones(len(data['positives'])), np.zeros(len(data['negatives'])))

	emb_words = list(embeddings_index.keys())
	if initial_vector == 'remove':
		tok = Tokenizer(num_words = len(emb_words), filters = '')
		tok.fit_on_texts(emb_words)
	else:
		tok = Tokenizer(num_words = len(emb_words), filters = '', oov_token = len(emb_words)+1)
		tok.fit_on_texts(emb_words)
	sequences = tok.texts_to_sequences(raw_x_data)
	word_index = tok.word_index

	print('Found %s unique tokens.' % len(word_index))

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

 	return x_train, y_train,  x_test, y_test, embedding_layer, tok 


def model_fit_apply(model, output_df): 

	classifier = model['Classifier']

	# Extract components of model
	if classifier == 'LogisticRegression':
		attribute = str(model['Attribute'])
		stop_words = str(model['Stop_words'])
		stemmer = str(model['Stemmer'])
		weighting = str(model['Weighting'])
		feature = str(model['Feature'])
		n_gram = tuple(map(int, str(model['Ngram_range'])[1:-1].split(',')))
		model_num = str(model['Model_number'])

		# Load relevant data
		text_data = get_datasets(attribute, stop_words, stemmer)

		x_data = text_data['positives'] + text_data['negatives']

		print 'Positive documents: ' + str(len(text_data['positives']))
		print 'Negative documents: ' + str(len(text_data['negatives']))

		# Create y data
		y_data = np.append(np.ones(len(text_data['positives'])), np.zeros(len(text_data['negatives'])))

		# Split data for training and testing
		x_train, x_test, y_train, y_test = split_data_lr(x = x_data, y = y_data, perc_train = 0.8)

		n_train = len(x_train)

		# Create vectorizer object
		if weighting == 'tf-idf':
			if stop_words == 'over_0.85_df':
				vectorizer_obj = TfidfVectorizer(analyzer = feature, ngram_range = n_gram, max_df = 0.85)
			else:
				vectorizer_obj = TfidfVectorizer(analyzer = feature, ngram_range = n_gram)

		if weighting == 'tf':
			if stop_words == 'over_0.85_df':
				vectorizer_obj = CountVectorizer(analyzer = feature, ngram_range = n_gram, max_df = 0.85)
			else:
				vectorizer_obj = CountVectorizer(analyzer = feature, ngram_range = n_gram)

		# cv
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

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
			fit_model = LogisticRegression(penalty='l2', class_weight='balanced').fit(x_train_vect, y_train_tmp)
		
			y_pred = fit_model.predict_proba(x_val_vect)
			y_pred_1_list = y_pred[:,1].tolist()


			if i == 1:
				print'Generating test set predictions...'
				# Vectorize full training set
				fitted_vect = vectorizer_obj.fit(x_train)
				x_train_full_vect = vectorizer_obj.transform(x_train)
				x_test_full_vect = vectorizer_obj.transform(x_test)


				# Use full training set to fit logistic
				# Train logistic classifier
				full_fit_model = LogisticRegression(penalty='l2', class_weight='balanced').fit(x_train_full_vect, y_train)
		
				# Generate predictions for test data
				y_pred_test = full_fit_model.predict_proba(x_test_full_vect)
				y_pred_test_1_list = y_pred_test[:,1].tolist()
			

				n_test_docs = len(x_test)	

				# save_lr_model(model_object = full_fit_model, vectorizer_object = fitted_vect, 
				# 	model_num = model_num, attribute = attribute, stop_words = stop_words, 
				# 	stemmer = stemmer, weighting = weighting, feature = feature, ngrams = n_gram, date_time = str(date_time_start))

			else:
				y_pred_test_1_list = None
				y_test = None
				n_test_docs = None

			date_time_end = datetime.datetime.now()

			fold_duration = datetime.timedelta.total_seconds(date_time_end - date_time_start)

			row_num = len(output_df)

			output_df.at[row_num, 'Attribute'] = attribute
			output_df.at[row_num, 'Model_number'] = model_num
			output_df.at[row_num, 'N_training_docs'] = n_train
			output_df.at[row_num, 'Classifier'] = 'LogisticRegression'
			output_df.at[row_num, 'Cost_function'] = 'l2'
			output_df.at[row_num, 'Stop_words'] = stop_words
			output_df.at[row_num, 'Stemmer'] = stemmer 
			output_df.at[row_num, 'Weighting'] = weighting
			output_df.at[row_num, 'Feature'] = feature
			output_df.at[row_num, 'Ngram_range'] = str(n_gram)
			output_df.at[row_num, 'CV_folds'] = str(10)
			output_df.at[row_num, 'Fold'] = str(i)
			output_df.at[row_num, 'CV_Predicted_Probs'] = str(y_pred_1_list)
			output_df.at[row_num, 'CV_True_Class'] = str(y_val)
			output_df.at[row_num, 'N_test_docs'] = str(n_test_docs)
			output_df.at[row_num, 'Test_Predicted_Probs'] = str(y_pred_test_1_list)
			output_df.at[row_num, 'Test_True_Class'] = str(y_test)
			output_df.at[row_num, 'Date_Time'] = str(date_time_start)
			output_df.at[row_num, 'Duration'] = fold_duration
			output_df.at[row_num, 'Dataset'] = "predicts"

			i += 1		
		
		return output_df



	elif classifier == 'Neural Network':
		attribute = model['Attribute']
		stop_words = model['Stop_words']
		unknown_words = model['Unknown_words']
		model_num = str(model['Model_number'])


		# load data (incorp stop word removal, stemmer = 'None')
		data = create_data(attribute = attribute, embeddings_index = embeddings_index, 
							perc_train = 0.8, stop_words = stop_words, initial_vector = unknown_words)

		x_train = data[0]
		y_train = data[1]
			
		x_test = data[2]
		y_test = data[3]
	
		embedding_layer = data[4]
		tokenizer = data[5]

		# Identify number of texts in train and test
		n_training_docs = x_train.shape[0]
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

			# print("Compiling model...")
			model = Model(sequence_input, preds)
			model.compile(loss='binary_crossentropy',
							optimizer='adam',
							metrics=['accuracy']) 
				
			return model

	    # Construct a k-fold for cross validation on training data
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

		# print('Training model.')
			
		date_time = str(datetime.datetime.now())

		i = 1
		for (train, test) in kfold.split(x_train, y_train):
			print "Running Fold " + str(i) + "/" + str(10)

			date_time_start = datetime.datetime.now()

			cv_model = None # Clearing the NN.
			cv_model = create_model()
			cv_model.fit(x_train[train], y_train[train], epochs=10, batch_size=10, verbose=0)
			cv_pred = cv_model.predict(x_train[test])
			cv_pred = list(chain.from_iterable(cv_pred.tolist()))
			tmp_y_val = y_train[test].tolist()
				
			if i == 1:
				print 'Generating test set predictions...'
				test_model = None
				test_model = create_model()
				test_model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=0)
				test_pred = test_model.predict(x_test)
				test_pred = list(chain.from_iterable(test_pred.tolist()))
				tmp_y_test = y_test.tolist()

				# Save model and tokenizer
				# save_nn_model(model_object = test_model, tokenizer_object = tokenizer, 
				# 		model_num = model_num, attribute = attribute, 
				# 		stop_words = stop_words, unknown_words = unknown_words, date_time = date_time)

			else:
				test_pred = None
				tmp_y_test = None
				n_test_docs = None

				
			date_time_end = datetime.datetime.now()

			fold_duration = datetime.timedelta.total_seconds(date_time_end - date_time_start)

			# Add info to df
			row_num = len(output_df)

			output_df.at[row_num, 'Attribute'] = attribute
			output_df.at[row_num, 'Model_number'] = model_num
			output_df.at[row_num, 'N_training_docs'] = n_training_docs
			output_df.at[row_num, 'Classifier'] = 'Neural Network'
			output_df.at[row_num, 'Cost_function'] = 'Binary cross entropy'
			output_df.at[row_num, 'Stop_words'] = stop_words
			output_df.at[row_num, 'Unknown_words'] = unknown_words
			output_df.at[row_num, 'CV_folds'] = str(10)
			output_df.at[row_num, 'Fold'] = str(i)
			output_df.at[row_num, 'CV_Predicted_Probs'] = str(cv_pred)
			output_df.at[row_num, 'CV_True_Class'] = str(tmp_y_val)
			output_df.at[row_num, 'N_test_docs'] = str(n_test_docs)
			output_df.at[row_num, 'Test_Predicted_Probs'] = str(test_pred)
			output_df.at[row_num, 'Test_True_Class'] = str(tmp_y_test)
			output_df.at[row_num, 'Date_Time'] = date_time
			output_df.at[row_num, 'Duration'] = fold_duration
			output_df.at[row_num, 'Dataset'] = "predicts"

			i += 1

		return output_df

	return None



#####
# Main code
#####

seed = 1

###
# Define global params 
###
# for length of texts
MAX_SEQUENCE_LENGTH = 1000
# word vector dimension
EMBEDDING_DIM = 100


# Load the embeddings index
embeddings_index = load_embeddings('../Data/glove_100d.txt')


# Load positives and negatives
p_df = pd.read_csv('../Data/Text/prepared_predicts_positives.csv')
n_df = pd.read_csv('../Data/Text/prepared_predicts_negatives.csv')

lr_models = pd.read_csv('../Results/Model_metrics/LR/lpi_models_to_use.csv')
nn_models = pd.read_csv('../Results/Model_metrics/NN/lpi_models_to_use.csv')

# Create df structure to store results
lr_df = pd.DataFrame(columns = ['Attribute',
								'Model_number',
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
								'Date_Time',
								'Duration',
								'Dataset'])


nn_df = pd.DataFrame(columns = ['Attribute',
								'Model_number',
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
								'Date_Time',
								'Duration',
								'Dataset'])


# For each model, run model function....
for i in range(lr_models.shape[0]):
	tmp_mod = lr_models.loc[i,]
	lr_df = model_fit_apply(model = tmp_mod, output_df = lr_df) 

for i in range(nn_models.shape[0]):
	tmp_mod = nn_models.loc[i]
	nn_df = model_fit_apply(model = tmp_mod, output_df = nn_df) 


# Save outputs
lr_df.to_csv('../Results/Model_metrics/LR/predicts_cv_metrics.csv', encoding = 'utf-8', index = False)
nn_df.to_csv('../Results/Model_metrics/NN/predicts_cv_metrics.csv', encoding = 'utf-8', index = False)

