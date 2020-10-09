#!/usr/bin/python

''' Python code to repeat cross-validation of the best models but using different 
sizes of training set and testing inclusion of true-negatives.
'''

# Import required modules/packages
import os
import os.path
import numpy as np
import pandas as pd
import random
import datetime
import re

from string import punctuation

from nltk.stem.lancaster import LancasterStemmer 

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

##
# Functions
##

# Function to stem words using a Lancaster stemmer function
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
def read_stop_words(file):
	with open('../../Data/Stop_words/' + file, 'r') as f:
		lines = [line.rstrip('\n') for line in f.readlines()]
	return frozenset(lines)


def replace_foreign_symbol(input_list):
	for i in range(len(input_list)):
		to_replace = re.findall(foreign_symbol_ex, input_list[i])
		
		if len(to_replace) > 0:
			for j in to_replace:
				section = re.findall(brackets_letter_ex, j)
				letter = re.findall(letter_ex, section[0])
				input_list[i] = input_list[i].replace(j, letter[0])

	return (input_list)


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
	return [regex.sub(replacement, str(element)) for element in array]


# Function for removing punctution from input string 
def rm_punctuation(string, replacement=' '): 
    for p in set(list(punctuation)): 
        string = string.replace(p, replacement)

    string = ' '.join(string.split())  # Remove excess whitespace
    return string

def lower_case(input_list):
	for i in range(len(input_list)):
		if isinstance(input_list[i], str):
			input_list[i] = input_list[i].lower()

	return(input_list)

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
	# output = [s.replace('\\\\&lt;', ' ') for s in output]
	# output = [s.replace('\\\\&gt;', ' ') for s in output]
	output = sub_all(numbering_ex, output)
	output = sub_all(num_ex, output)
	output = [rm_punctuation(s) for s in output]
	output = [s.lower() for s in output]
	# output = sub_all(x_tags_ex, output, replacement = ' ')
	# output = [s.replace('\xc2\xa0', ' ') for s in output]

	# test rm of copyright txt...
	output = [s.split(' \xc2\xac\xc2\xa9')[0] for s in output]
	
	return (output)


def split_data(x, y, perc_train):

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
def get_datasets(p_df, n_df, attribute):
	# Code to concat Title and Abstract if attribute == 'Title_Abstract'
	if attribute == 'TitleAbstract':
		p_df['TitleAbstract'] = p_df['Title'] + ' ' + p_df['Abstract']
		n_df['TitleAbstract'] = n_df['Title'] + ' ' + n_df['Abstract']

	positives = p_df[attribute].tolist()
	negatives = n_df[attribute].tolist()

	random.seed(seed)
	negatives = random.sample(negatives, len(positives))

	return {'positives': positives, 'negatives': negatives}


# Function to conduct 10 fold cros validation whilst also adjusting training size
def npos_cv(df):
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	i = 1
	df["Fold_id"] = np.nan

	for npos in [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]:
		for rs in range(10):
			if (npos==1) & (rs>0):
				break
			# print("p_"+str(npos)+"_"+str(rs))
			df["p_"+str(npos)+"_"+str(rs)] = np.nan
		# df["p_"+str(npos)] = np.nan

	for (tr, te) in kfold.split(df["x"], df["y"]):
		#df.iloc[te]["Fold_id"] = i
		df.loc[te, "Fold_id"] = i
		te_df = df.iloc[te]
		tr_df = df.iloc[tr]

		for npos in [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]:
			for rs in range(10):
				if npos > tr_df.shape[0]/2:
					break

				elif npos == 1:
					tr_samp_df = tr_df
					if rs>0:
						break

				else:
					# Sample np positives and np negatives from tr_df
					p_samp = tr_df.loc[tr_df["y"]==1].sample(n=npos, replace=False, random_state = rs)
					n_samp = tr_df.loc[tr_df["y"]==0].sample(n=npos, replace=False, random_state = rs)
					tr_samp_df = pd.concat([p_samp[["x", "y"]], 
											n_samp[["x", "y"]]])

				# Vectorize
				vect = None
				vect = TfidfVectorizer(analyzer = "word", ngram_range = (1,2), max_df = 0.85)
				x_tr = vect.fit_transform(tr_samp_df["x"])
				x_te = vect.transform(te_df["x"])

				# Train LR
				lr_mod = None
				lr_mod = LogisticRegression(penalty='l2', class_weight='balanced').fit(x_tr, tr_samp_df["y"])
				
				# Predict
				df.loc[te, "p_"+str(npos)+"_"+str(rs)] = lr_mod.predict_proba(x_te)[:,1].tolist()
		i += 1

	return(df)


def weight_calc(tr_df, method, w, n_new_neg):
	tr_df["tmp_w"] = np.nan

	if method == 1:
		tr_df.loc[tr_df["Set"]=="original", "tmp_w"] = 1-w
		tr_df.loc[tr_df["Set"]=="new", "tmp_w"] = w

	elif method == 4:
		n_neg = sum(tr_df["Relevance"]==0)
		n_orig_neg = n_neg-n_new_neg
		tr_df.loc[tr_df["Relevance"]==1, "tmp_w"] = 1
		tr_df.loc[(tr_df["Set"]=="original") & (tr_df["Relevance"]==0), "tmp_w"] = (n_neg*(1-w))/n_orig_neg
		tr_df.loc[(tr_df["Set"]=="new") & (tr_df["Relevance"]==0), "tmp_w"] = (n_neg*w)/n_new_neg

	weights = tr_df["tmp_w"].tolist()

	return(weights)


def augment_data_test_model(orig_texts_df, new_texts_df):

	# Split data for training and testing - only keep training data
	x_tr_orig, x_te_orig, y_tr_orig, y_te_orig = split_data(x = orig_texts_df["x"], 
															y = orig_texts_df["y"], 
															perc_train = 0.8)

	# LRA processing, stemming
	tr_orig_df = pd.DataFrame({"TitleAbstract" : x_tr_orig,
								"Relevance" : y_tr_orig,
								"Set" : "original"})

	n_orig = tr_orig_df.shape[0]

	new_texts_df["TitleAbstract"] = [stem_words(txt) for txt in new_texts_df["TitleAbstract"].tolist()]

	# List unique searches in new texts_df
	searches = list(set(new_texts_df["Search"]))

	w_var = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	meth_var = [4] #1, 
	for w in w_var:
		for m in meth_var:
			new_texts_df["p_"+str(m)+"_"+str(w)] = np.nan

	n_reps = len(w_var)*len(meth_var)
	nrows = len(searches)*n_reps
	nan_ls = [np.nan]*nrows
	cache_df = pd.DataFrame({"Search" 		: np.repeat(searches,n_reps),
							"N_orig_pos" 	: [n_orig/2]*nrows,
							"N_orig_neg" 	: [n_orig/2]*nrows,
							"N_new_pos" 	: nan_ls,
							"N_new_neg" 	: nan_ls,
							"w" 			: nan_ls,
							"method"		: nan_ls,
							"w_orig_pos" 	: nan_ls,
							"w_new_pos" 	: nan_ls,
							"w_orig_neg" 	: nan_ls,
							"w_new_neg" 	: nan_ls})

	rw_cnt = 0

	# For each search
	for s in searches:
		# Identify split - i.e. which new texts to train/test with
		te_df = new_texts_df.loc[new_texts_df["Search"]==s,]
		tr_new_df = new_texts_df.loc[new_texts_df["Search"]!=s,]
		
		# Drop any texts present in test from train
		# And drop duplicates in train
		if "EID" in new_texts_df.columns:
			tr_new_df = tr_new_df[~tr_new_df["EID"].isin(te_df["EID"].tolist())]
			tr_new_df = tr_new_df.drop_duplicates(subset = ["EID"])
		if "UT" in new_texts_df.columns:
			tr_new_df = tr_new_df[~tr_new_df["UT"].isin(te_df["UT"].tolist())]
			tr_new_df = tr_new_df.drop_duplicates(subset = ["UT"])

		n_new_neg = sum(tr_new_df["Relevance"]==0)
		n_new_pos = sum(tr_new_df["Relevance"]==1)

		# Make df to store texts, label, new/orig
		tr_df = pd.concat([tr_orig_df[["TitleAbstract", "Relevance", "Set"]], 
							tr_new_df[["TitleAbstract", "Relevance", "Set"]]])

		# Vectorize texts using best specifications... - training and test
		vect = TfidfVectorizer(analyzer = "word", ngram_range = (1,2), max_df = 0.85)
		tr_txt = vect.fit_transform(tr_df["TitleAbstract"])
		te_txt = vect.transform(te_df["TitleAbstract"])

		# Set weights assigned to original v new texts
		for w in w_var:
			for m in meth_var:
				# Set weight col to nan
				tr_df["Weight_"+str(m)+"_"+str(w)] = np.nan
				tr_df["Weight_"+str(m)+"_"+str(w)] = weight_calc(tr_df, m, w, n_new_neg)

				# Add to cache...
				cache_df.loc[rw_cnt,"method"] = m
				cache_df.loc[rw_cnt,"w"] = w
				cache_df.loc[rw_cnt,"w_orig_pos"] = tr_df.loc[(tr_df["Set"]=="original")&(tr_df["Relevance"]==1),"Weight_"+str(m)+"_"+str(w)].tolist()[0]
				cache_df.loc[rw_cnt,"w_new_pos"] = tr_df.loc[(tr_df["Set"]=="new")&(tr_df["Relevance"]==1),"Weight_"+str(m)+"_"+str(w)].tolist()[0]
				cache_df.loc[rw_cnt,"w_orig_neg"] = tr_df.loc[(tr_df["Set"]=="original")&(tr_df["Relevance"]==0),"Weight_"+str(m)+"_"+str(w)].tolist()[0]
				cache_df.loc[rw_cnt,"w_new_neg"] = tr_df.loc[(tr_df["Set"]=="new")&(tr_df["Relevance"]==0),"Weight_"+str(m)+"_"+str(w)].tolist()[0]

				cache_df.loc[rw_cnt,"N_new_pos"] = n_new_pos
				cache_df.loc[rw_cnt,"N_new_neg"] = n_new_neg	

				# Fit model
				lr_mod_w = None
				lr_mod_w = LogisticRegression(penalty='l2', class_weight='balanced').fit(tr_txt, tr_df["Relevance"], sample_weight = tr_df["Weight_"+str(m)+"_"+str(w)])
				# Predict
				new_texts_df.loc[new_texts_df["Search"]==s, "p_"+str(m)+"_"+str(w)] = lr_mod_w.predict_proba(te_txt)[:,1].tolist()
				rw_cnt += 1

	return(new_texts_df, cache_df)


##
# Variables
##

# Define regular expressions for finding unwanted symbols and tags
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


##
# Main Code
##

seed = 1

# Load texts
lpi_p_df = pd.read_csv("../Data/Text/prepared_lpi_positives.csv")
lpi_n_df = pd.read_csv("../Data/Text/prepared_lpi_negatives.csv")

predicts_p_df = pd.read_csv("../Data/Text/prepared_predicts_positives.csv")
predicts_n_df = pd.read_csv("../Data/Text/prepared_predicts_negatives.csv")

# Initial text data collation
lpi_x = get_datasets(lpi_p_df, lpi_n_df, "TitleAbstract")
lpi_y = np.append(np.ones(len(lpi_x['positives'])), np.zeros(len(lpi_x['negatives'])))
lpi_x = lpi_x["positives"] + lpi_x["negatives"]

predicts_x = get_datasets(predicts_p_df, predicts_n_df, "TitleAbstract")
predicts_y = np.append(np.ones(len(predicts_x['positives'])), np.zeros(len(predicts_x['negatives'])))
predicts_x = predicts_x["positives"] + predicts_x["negatives"]

## Npos 
# Stemming
lpi_x = [stem_words(txt) for txt in lpi_x]
predicts_x = [stem_words(txt) for txt in predicts_x]

lpi_npos_df = pd.DataFrame({"x" : lpi_x,
						"y" : lpi_y})
predicts_npos_df = pd.DataFrame({"x" : predicts_x,
							"y" : predicts_y})

lpi_npos_df = npos_cv(lpi_npos_df)
predicts_npos_df = npos_cv(predicts_npos_df)

lpi_npos_df.to_csv("../Results/lpi_npos_rs10.csv", index = False)
predicts_npos_df.to_csv("../Results/predicts_npos_rs10.csv", index = False)


## Augmentation
lpi_new_df = pd.read_csv("../Results/lpi_man_class.csv")
predicts_new_df = pd.read_csv("../Results/predicts_man_class.csv")

# Process newly classified texts
lpi_new_df["TitleAbstract"] = text_preprocessing(lpi_new_df["Title"] + " " + lpi_new_df["Abstract"])
lpi_new_df["Set"] = ["new"]*lpi_new_df.shape[0]

predicts_new_df["TitleAbstract"] = text_preprocessing(predicts_new_df["Title"] + " " + predicts_new_df["Abstract"])
predicts_new_df["Set"] = ["new"]*predicts_new_df.shape[0]

lpi_new_df, lpi_cache_df = augment_data_test_model(lpi_npos_df[["x","y"]], lpi_new_df)
predicts_new_df, predicts_cache_df = augment_data_test_model(predicts_npos_df[["x","y"]], predicts_new_df)

lpi_new_df.to_csv("../Results/lpi_man_class_augment.csv", index = False)
predicts_new_df.to_csv("../Results/predicts_man_class_augment.csv", index = False)
lpi_cache_df.to_csv("../Results/lpi_man_class_augment_cache.csv", index = False)
predicts_cache_df.to_csv("../Results/predicts_man_class_augment_cache.csv", index = False)

