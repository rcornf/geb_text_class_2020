#!usr/bin/python

'''
Python code to demonstrate the application of saved text-classification models 
(LR A and CNN A), given a specified input csv of unclassified texts.
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

from sklearn.externals import joblib

from keras.models import load_model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences


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


# Load and apply lra model
def load_apply_lr(u):
	# Load saved model and vecgtorizer
	print("Loading lr model.")
	lr_mod = joblib.load("../Results/lra_mod.pkl")
	vectorizer = joblib.load("../Results/lra_vect.pkl")

	# Stem texts
	u_stem = [stem_words(s) for s in u]

	# Vectorize
	u_dat = vectorizer.transform(u_stem)

	# Predict
	print("Predicting relevance of texts.")
	return(lr_mod.predict_proba(u_dat)[:,1].tolist())


# Load and apply cnna model
def load_apply_cnn(u):
	# Load saved model and tokenizer
	print("Loading cnn model.")
	cnn_mod = load_model("../Results/cnna_mod.h5")
	tok = joblib.load("../Results/cnna_tok.pkl")

	# Remove stop words
	my_stop_words = read_stop_words('NLTK_stop_words.txt')
	# # Modify text to remove stop words
	print ("Removing stop words.")
	u_txt = [rm_stop_words(s, my_stop_words) for s in u]

	# Tokenize texts
	u_seq = tok.texts_to_sequences(u_txt)
	u_dat = pad_sequences(u_seq, maxlen=MAX_SEQUENCE_LENGTH)
	
	# Apply model
	print("Predicting relevance of texts.")
	return(list(chain.from_iterable(cnn_mod.predict(u_dat).tolist())))


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


	# Load texts
	print("Loading texts.")
	u_df = pd.read_csv("../Data/unclassifieds.csv")
	#*** Specify your own unclassified texts here... e.g. ***#
	# u_df = pd.read_csv("../Data/<my_unclass_texts>.csv")

	# Combine titles and abstracts
	u_df["TitleAbstract"] = u_df["Title"] + ' ' + u_df["Abstract"]
	u_txts = u_df['TitleAbstract'].tolist()

	# Preprocess texts
	print("Processing texts.")
	u_txts = text_preprocessing(u_txts)

	# Apply lr classifier
	u_df["lr_p"] = load_apply_lr(u_txts)
	# Apply cnn classifier
	u_df["cnn_p"] = load_apply_cnn(u_txts)

	u_df.to_csv("../Results/classifieds.csv")

