#!usr/bin/python

'''
Python code to demonstrate the fitting of text-classification models (LR A and
CNN A), given a specified input csv of positive texts.
Note: requires the Anaconda environment "python_env" 
'''


## Load modules
import sys
import os
import os.path
import numpy as np
import pandas as pd
import random
import datetime
import re
from string import punctuation

from itertools import chain

from nltk.stem.lancaster import LancasterStemmer 

from sklearn.feature_extraction.text import TfidfVectorizer
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


## Functions

## Text processing functions

# Function to stem words using a Lancaster stemmer
def stem_words(input_string): 
	# convert to unicode from utf-8
	word_list = [unicode(i, 'utf-8') for i in input_string.split(' ')]

	# Stem words
	stem_list = [LancasterStemmer().stem(i) for i in word_list]

	# Join strings
	stem_str = (' ').join(stem_list)

	# Convert back to utf-8
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


# Function to replace encoded letter+symbol with just letter
def replace_foreign_symbol(input_list):
	for i in range(len(input_list)):
		to_replace = re.findall(foreign_symbol_ex, input_list[i])
		
		if len(to_replace) > 0:
			for j in to_replace:
				section = re.findall(brackets_letter_ex, j)
				letter = re.findall(letter_ex, section[0])
				input_list[i] = input_list[i].replace(j, letter[0])
			
	return (input_list)


# Funtion to replace symbol with relevant letter
def replace_weird_symbol(input_list):
	for i in range(len(input_list)):
		to_replace = re.findall(weird_symbol_ex, input_list[i])

		if len(to_replace) > 0:
			for j in to_replace:
				section = re.findall(slash_letter_ex, j)
				letter = re.findall(letter_ex, section[0])
				input_list[i] = input_list[i].replace(j, letter[0])

	return (input_list)


# Function to enact regex substitution on an array of strings
def sub_all(regex, array, replacement=''):
	"""Regex substitution on an array of strings."""
	return [regex.sub(replacement, str(element)) for element in array]


# Function for removing punctution from input string 
def rm_punctuation(string, replacement=' '):       # , exclude="'-'"
	"""Remove punctuation from an input string """
	for p in set(list(punctuation)): # - set(list(exclude)):
		string = string.replace(p, replacement)

	string = ' '.join(string.split())  # Remove excess whitespace
	return string


# Function to process text
def text_preprocessing(input_list):
	output = sub_all(html_ex, input_list)
	output = sub_all(basic_backslash_tag_ex, output, replacement = ' ')
	output = sub_all(backslash_mod_ex, output)
	output = replace_foreign_symbol(output)
	output = replace_weird_symbol(output)
	output = sub_all(endash_type1_ex, output, replacement = ' ')
	output = sub_all(endash_type2_ex, output, replacement = ' ')
	output = sub_all(slash_percent_ex, output, replacement = ' ')
	output = [s.replace('\r\n', ' ') for s in output]
	output = [s.replace('\\n', ' ') for s in output]
	output = [s.replace('ufffd', ' ') for s in output]
	output = [s.replace('\\\\&lt;', ' ') for s in output]
	output = [s.replace('\\\\&gt;', ' ') for s in output]
	output = sub_all(numbering_ex, output)
	output = sub_all(num_ex, output)
	output = [rm_punctuation(s) for s in output]
	output = [s.lower() for s in output]
	
	return (output)


# Function to train and save an lra model for later use
def train_save_lr(x, y):
	# stem texts
	print("Stemming words.")
	x_stem = [stem_words(s) for s in x]
	# Vectorize texts
	vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), max_df = 0.85)
	x_dat = vectorizer.fit_transform(x_stem)
	
	# Fit LR 
	print("Fitting LR A model.")
	lr_mod = LogisticRegression(penalty='l2', class_weight='balanced').fit(x_dat, y)

	# Save models and vectorizer
	joblib.dump(lr_mod, "../Results/lra_mod.pkl", compress = 1)
	print ('\tSaved model...')
	joblib.dump(vectorizer, "../Results/lra_vect.pkl", compress = 1)
	print ('\tSaved vectorizer...')
	return(None)


# Function to create an embedding layer for use in cnns
def make_embedding_layer(word_index, embeddings_index, initial):
	print('Preparing embedding matrix.')
	# Number of words is either the number of words identified in the tokenizer index
	num_words = len(word_index) + 1

	# Embedding matrix is created, nrows = num_words, ncols = 100)
	# initial == 'rand': 
	embedding_matrix = np.random.normal(np.stack(embeddings_index.values()).mean(), np.stack(embeddings_index.values()).std(), (num_words, EMBEDDING_DIM))

	for word, i in word_index.items():

		# Don't modify unknown word vector
		if i == 400001:
			continue

		# Extract GloVe vector for identified word    
		embedding_vector = embeddings_index.get(word)

		if embedding_vector is not None:
			# Add GloVe vector to embedding matrix
			embedding_matrix[i] = embedding_vector

	# Set 0 index to zeros for padded values
	embedding_matrix[0] = np.zeros(EMBEDDING_DIM)

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


# Function to fit and save a cnna model 
def train_save_cnn(x, y):
	# Load glove
	embeddings_index = load_embeddings("../Data/glove.6B.100d.txt")

	emb_words = list(embeddings_index.keys())

	# Create tokenizer
	tok = Tokenizer(num_words = len(emb_words), filters = '', oov_token = len(emb_words)+1)
	tok.fit_on_texts(emb_words)

	# Remove stop words
	my_stop_words = read_stop_words('NLTK_stop_words.txt')
	print ("Removing stop words.")
	x_stop = [rm_stop_words(s, my_stop_words) for s in x]

	# tokenize texts
	x_seq = tok.texts_to_sequences(x_stop)
	x_dat = pad_sequences(x_seq, maxlen=MAX_SEQUENCE_LENGTH)

	# Make word embedding layer
	word_index = tok.word_index
	embedding_layer = make_embedding_layer(word_index, embeddings_index, "rand")
	
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
		#model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

		# print("Compiling model...")
		model = Model(sequence_input, preds)
		model.compile(loss='binary_crossentropy',
					optimizer='adam',
					metrics=['accuracy']) #, precision, recall]
					
		return model

	# fit CNN models
	cnn_mod = create_model()
	print("Fitting CNN A model.")
	cnn_mod.fit(x_dat, y, epochs=10, batch_size=10, verbose=0)

	# save
	cnn_mod.save("../Results/cnna_mod.h5")
	print '\tSaved model...'
	joblib.dump(tok, "../Results/cnna_tok.pkl", compress = 1)
	print '\tSaved tokenizer...'
	return(None)


## Run demo code
if __name__ == '__main__':
	seed = 1

	# Define global params for length of texts
	MAX_SEQUENCE_LENGTH = 1000
	# word vector dimension
	EMBEDDING_DIM = 100

	# Regex defs
	html_ex = re.compile(r'</?\w+>') # works
	alt_html_ex = re.compile(r'<.*?>')
	numbering_ex = re.compile(r'\b\d\.') # works
	spacing_ex = re.compile(r'\s{2,}') # works
	num_ex = re.compile(r'\d')
	basic_backslash_tag_ex = re.compile(r'\$\\+backslash\$n')
	backslash_mod_ex = re.compile(r'{\$\\+backslash\$\w+\\+}')
	# '\\\\u1234' but also '\\\\u123a'
	endash_type1_ex = re.compile(r'\\\\u\d{4}') # works
	endash_type2_ex = re.compile(r'\\\\u\d{3}\w')
	# '\\\\\\\\%'
	slash_percent_ex = re.compile(r'\\\\\\\\%')
	# Need regex for '\xNN' style features
	x_tags_ex = re.compile(r'\\x[\w\d]{2}')
	# Need regex for '\\\\\\\\(symbol){letter}' and replace with letter
	foreign_symbol_ex = re.compile(r'\\\\\\\\.{\w}')
	brackets_letter_ex = re.compile(r'{\w}')
	letter_ex = re.compile(r'\w')
	weird_symbol_ex = re.compile(r'\\\\\\\\.{\\\\\\\\\w}')
	slash_letter_ex = re.compile(r'\w}')


	# Create output directory
	res_dir = "../Results"
	if not os.path.exists(res_dir):
		os.makedirs(res_dir)

	# Load texts
	print("Loading training texts.")
	p_df = pd.read_csv("../Data/positives.csv") 
	n_df = pd.read_csv("../Data/negatives.csv")
	#*** Specify your own positive/negative texts here... e.g. ***#
	# p_df = pd.read_csv("../Data/<my_pos_texts>.csv") 
	# n_df = pd.read_csv("../Data/<my_neg_texts>.csv") 

	# Combine titles and abstracts
	p_df["TitleAbstract"] = p_df["Title"] + ' ' + p_df["Abstract"]
	n_df["TitleAbstract"] = n_df["Title"] + ' ' + n_df["Abstract"]

	p_txts = p_df['TitleAbstract'].tolist()
	n_txts = n_df['TitleAbstract'].tolist()

	# Make the number of negative texts equal to the number of positives
	n_txts = random.sample(n_txts, len(p_txts))

	# Preprocess texts
	print("Processing text data.")
	p_txts = text_preprocessing(p_txts)
	n_txts = text_preprocessing(n_txts)

	# Generate training data and labels
	x = p_txts + n_txts
	y = np.append(np.ones(len(p_txts)), np.zeros(len(n_txts)))

	# LR model fit
	train_save_lr(x, y)

	# CNN model fit
	train_save_cnn(x, y)

