#!/usr/bin/python

'''Python code to compare words in GloVe and training texts.
'''

# Load packages
import os
import sys
import pandas as pd
import random
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer 
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Functions
# Load GloVe (words only)
# Function loading GloVe word vectors
def load_glove_words(filepath):
	import pickle

	print("Loading words from glove data...")
	words = []
	f = open(filepath)
	for line in f:
		values = line.split()
		word = values[0]
		words.append(word)
	f.close()

	print("done.")

	# Return the GloVe words
	return(words)

# Function to load title+abstract texts, convert to list of words, return unique words
def glove_text_compare(positive_df, negative_df, glove_words):
	positive_df['TitleAbstract'] = positive_df['Title'] + ' ' + positive_df['Abstract']
	negative_df['TitleAbstract'] = negative_df['Title'] + ' ' + negative_df['Abstract']

	positives = positive_df['TitleAbstract'].tolist()
	negatives = negative_df['TitleAbstract'].tolist()

	random.seed(seed)
	negatives_samp = random.sample(negatives, len(positives))

	texts = positives + negatives_samp

	word_list = [i for input_string in texts for i in input_string.split(' ')]

	word_set = set(word_list)

	word_dif = list(word_set.difference(set(glove_words)))

	return(texts, word_list, word_set, word_dif)


# Main Code
seed = 1

# Load csv files
lpi_p_df = pd.read_csv('../Data/Text/prepared_lpi_positives.csv')
lpi_n_df = pd.read_csv('../Data/Text/prepared_lpi_negatives.csv')

predicts_p_df = pd.read_csv('../Data/Text/prepared_predicts_positives.csv')
predicts_n_df = pd.read_csv('../Data/Text/prepared_predicts_negatives.csv')

glove_word_list = load_glove_words('../Data/glove_100d.txt')
print ('Number of words in GloVe: ' + str(len(glove_word_list)))

lpi_texts, all_lpi_words, set_lpi_words, lpi_train_dif = glove_text_compare(positive_df = lpi_p_df, 
																negative_df = lpi_n_df, 
																glove_words = glove_word_list)

predicts_texts, all_predicts_words, set_predicts_words, predicts_train_dif = glove_text_compare(positive_df = predicts_p_df,
																				negative_df = predicts_n_df,
																				glove_words = glove_word_list)

print ('Number of words in LPI training texts: ' + str(len(all_lpi_words))) 
print ('Number of words in LPI training texts but not GloVe: ' + str(len(lpi_train_dif)))

print ('\nNumber of words in PREDICTS training texts: ' + str(len(all_predicts_words))) 
print ('Number of words in PREDICTS training texts but not GloVe: ' + str(len(predicts_train_dif)))

print ('\nLPI words not in GloVe: ' + str(sorted(lpi_train_dif)))
print ('\nPREDICTS words not in GloVe: ' + str(sorted(predicts_train_dif)))


##########
# Check for coverage of discriminatory words as found in LR A model
# Load LR A features
lpi_feats_df = pd.read_csv('../Results/Model_metrics/LR/lpi_feats_coeffs.csv')
predicts_feats_df = pd.read_csv('../Results/Model_metrics/LR/predicts_feats_coeffs.csv')

# Stem GloVe words (Lancaster)
stemmed_emb = [LancasterStemmer().stem(unicode(i, 'utf-8')) for i in glove_word_list]

# Rm na
# lpi_feats_df = lpi_feats_df.dropna(subset = ["149_TitleAbstract_features"])
# predicts_feats_df = predicts_feats_df.dropna(subset = ["149_TitleAbstract_features"])

# Sort by coeffs
lpi_feats_df = lpi_feats_df.sort_values(by = "LR_A_coefficients", 
										axis = 0, 
										ascending = False)

predicts_feats_df = predicts_feats_df.sort_values(by = "LR_A_coefficients", 
													axis = 0, 
													ascending = False)

# Extract top 50
lpi_50_feats = lpi_feats_df["LR_A_features"][0:50].tolist()
predicts_50_feats = predicts_feats_df["LR_A_features"][0:50].tolist()

# Set comparison
len(set(lpi_50_feats).difference(set(stemmed_emb))) # 5
len(set(lpi_50_feats).intersection(set(stemmed_emb))) # 45

len(set(predicts_50_feats).difference(set(stemmed_emb))) # 2
len(set(predicts_50_feats).intersection(set(stemmed_emb))) # 48

# absent stems are combinations of two words
##########

# Compare stemmed versions of non-glove words to top 50 words
stemmed_lpi_dif = [LancasterStemmer().stem(unicode(i, 'utf-8')) for i in lpi_train_dif]
stemmed_predicts_dif = [LancasterStemmer().stem(unicode(i, 'utf-8')) for i in predicts_train_dif]

lpi_match = [len(set(lpi_50_feats).intersection(set([x]))) for x in stemmed_lpi_dif]
predicts_match = [len(set(predicts_50_feats).intersection(set([x]))) for x in stemmed_predicts_dif]

lpi_non_glove = pd.DataFrame(data = {"Word" : lpi_train_dif,
									"Stem" : stemmed_lpi_dif,
									"Match" : lpi_match})
predicts_non_glove = pd.DataFrame(data = {"Word" : predicts_train_dif,
										"Stem" : stemmed_predicts_dif,
										"Match" : predicts_match})

lpi_non_glove_stop = lpi_non_glove.loc[lpi_non_glove["Match"]==1, "Word"].tolist()
predicts_non_glove_stop = predicts_non_glove.loc[predicts_non_glove["Match"]==1, "Word"].tolist()

# Find all word occurances where word is not in GloVe but stem is in top 50
all_lpi_missing = [word for word in all_lpi_words if word in lpi_non_glove.loc[lpi_non_glove['Match'] != 0, "Word"].tolist()]
# 10 words total - ['seasonalized', 'catchability', 'nestings', 'catchability', 'periodicities', 'catchability', 'catchability', 'abundanc', 'populational', 'popan']
all_predicts_missing = [word for word in all_predicts_words if word in predicts_non_glove.loc[predicts_non_glove['Match'] != 0, "Word"].tolist()]
# 8 words total - ['pollinations', 'richnesses', 'conservational', 'faunistic', 'faunistic', 'conservational', 'vegetational', 'faunistic']


# In [104]: lpi_non_glove[lpi_non_glove['Match'] != 0]
# Out[104]: 
#       Match    Stem           Word
# 406       1  period  periodicities
# 943       1   abund       abundanc
# 2000      1    nest       nestings
# 2009      1   catch   catchability
# 2122      1  season   seasonalized
# 2789      1     pop   populational
# 3032      1     pop          popan

# In [105]: predicts_non_glove[predicts_non_glove['Match'] != 0]
# Out[105]: 
#       Match     Stem            Word
# 711       1     faun       faunistic
# 829       1   pollin    pollinations
# 895       1  conserv  conservational
# 2021      1    veget    vegetational
# 2674      1     rich      richnesses


# Split texts in train/test
def split_data(x, perc_train):

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

	# # Assign y values to train or test
	# y_train = [y[i] for i in train_indexes]
	# y_test = [y[i] for i in test_indexes]

	# Return training and test data
	return x_train, x_test #, y_train, y_test

lpi_x_tr, lpi_x_te = split_data(lpi_texts, 0.8)
predicts_x_tr, predicts_x_te = split_data(predicts_texts, 0.8)


# Rm non-glove stop-words

# Function to remove stop words from text strings
def rm_stop_words(input_string, stop_words):
	# Split input by spaces and remove words in stop_words
	word_list = [i for i in input_string.split(' ') if i not in stop_words]
	return (' ').join(word_list)

lpi_x_tr_stop = [rm_stop_words(x, lpi_non_glove_stop) for x in lpi_x_tr]
lpi_x_te_stop = [rm_stop_words(x, lpi_non_glove_stop) for x in lpi_x_te]

predicts_x_tr_stop = [rm_stop_words(x, predicts_non_glove_stop) for x in predicts_x_tr]
predicts_x_te_stop = [rm_stop_words(x, predicts_non_glove_stop) for x in predicts_x_te]


# Process based on LRA - Lancaster stemm, 

# Function to stem words using a provided stemmer function
def stem_words(input_string):
	# convert to unicode from utf-8
	word_list = [unicode(i, 'utf-8') for i in input_string.split(' ')]
	stem_list = [LancasterStemmer().stem(i) for i in word_list]
	# Join strings
	stem_str = (' ').join(stem_list)
	# Convert back to utf-8??
	return stem_str.encode('utf8')

lpi_x_tr_stem = [stem_words(x) for x in lpi_x_tr]
lpi_x_te_stem = [stem_words(x) for x in lpi_x_te]

predicts_x_tr_stem = [stem_words(x) for x in predicts_x_tr]
predicts_x_te_stem = [stem_words(x) for x in predicts_x_te]

lpi_x_tr_stop_stem = [stem_words(x) for x in lpi_x_tr_stop]
lpi_x_te_stop_stem = [stem_words(x) for x in lpi_x_te_stop]

predicts_x_tr_stop_stem = [stem_words(x) for x in predicts_x_tr_stop]
predicts_x_te_stop_stem = [stem_words(x) for x in predicts_x_te_stop]


# Load models and vectorizers
lpi_models_df = pd.read_csv("../Results/Model_metrics/LR/lpi_models_to_use.csv")
lpi_model_spec = lpi_models_df.loc[0,:]

lpi_mod_fp = '../Results/Models/LR/model_lpi_' + str(lpi_model_spec['Model_number']) + '_' + str(lpi_model_spec['Attribute']) + '_' + str(lpi_model_spec['Stop_words']).rstrip('.txt') + '_' + str(lpi_model_spec['Stemmer']) + '_' + str(lpi_model_spec['Weighting']) + '_' + str(lpi_model_spec['Feature']) + '_' + str(lpi_model_spec['Ngram_range']) + '_' + str(lpi_model_spec['Date_Time']) + '.pkl'
lpi_vect_fp = '../Results/Models/LR/vectorizer_lpi_' + str(lpi_model_spec['Model_number']) + '_' + str(lpi_model_spec['Attribute']) + '_' + str(lpi_model_spec['Stop_words']).rstrip('.txt') + '_' + str(lpi_model_spec['Stemmer']) + '_' + str(lpi_model_spec['Weighting']) + '_' + str(lpi_model_spec['Feature']) + '_' + str(lpi_model_spec['Ngram_range']) + '_' + str(lpi_model_spec['Date_Time']) + '.pkl'
		
lpi_lra = joblib.load(lpi_mod_fp)
lpi_vect = joblib.load(lpi_vect_fp)


precicts_models_df = pd.read_csv("../Results/Model_metrics/LR/predicts_models_to_use.csv")
predicts_model_spec = precicts_models_df.loc[0,:]

predicts_mod_fp = '../Results/Models/LR/model_predicts_' + str(predicts_model_spec['Model_number']) + '_' + str(predicts_model_spec['Attribute']) + '_' + str(predicts_model_spec['Stop_words']).rstrip('.txt') + '_' + str(predicts_model_spec['Stemmer']) + '_' + str(predicts_model_spec['Weighting']) + '_' + str(predicts_model_spec['Feature']) + '_' + str(predicts_model_spec['Ngram_range']) + '_' + str(predicts_model_spec['Date_Time']) + '.pkl'
predicts_vect_fp = '../Results/Models/LR/vectorizer_predicts_' + str(predicts_model_spec['Model_number']) + '_' + str(predicts_model_spec['Attribute']) + '_' + str(predicts_model_spec['Stop_words']).rstrip('.txt') + '_' + str(predicts_model_spec['Stemmer']) + '_' + str(predicts_model_spec['Weighting']) + '_' + str(predicts_model_spec['Feature']) + '_' + str(predicts_model_spec['Ngram_range']) + '_' + str(predicts_model_spec['Date_Time']) + '.pkl'

predicts_lra = joblib.load(predicts_mod_fp)
predicts_vect = joblib.load(predicts_vect_fp)	


# Vectorize texts
lpi_x_tr_stem_v = lpi_vect.transform(lpi_x_tr_stem)
lpi_x_te_stem_v = lpi_vect.transform(lpi_x_te_stem)

predicts_x_tr_stem_v = predicts_vect.transform(predicts_x_tr_stem)
predicts_x_te_stem_v = predicts_vect.transform(predicts_x_te_stem)

lpi_x_tr_stop_stem_v = lpi_vect.transform(lpi_x_tr_stop_stem)
lpi_x_te_stop_stem_v = lpi_vect.transform(lpi_x_te_stop_stem)

predicts_x_tr_stop_stem_v = predicts_vect.transform(predicts_x_tr_stop_stem)
predicts_x_te_stop_stem_v = predicts_vect.transform(predicts_x_te_stop_stem)


# Classify original and modified (train_test) texts 
lpi_x_tr_stem_p = lpi_lra.predict_proba(lpi_x_tr_stem_v)[:,1].tolist()
lpi_x_te_stem_p = lpi_lra.predict_proba(lpi_x_te_stem_v)[:,1].tolist()

predicts_x_tr_stem_p = predicts_lra.predict_proba(predicts_x_tr_stem_v)[:,1].tolist()
predicts_x_te_stem_p = predicts_lra.predict_proba(predicts_x_te_stem_v)[:,1].tolist()

lpi_x_tr_stop_stem_p = lpi_lra.predict_proba(lpi_x_tr_stop_stem_v)[:,1].tolist()
lpi_x_te_stop_stem_p = lpi_lra.predict_proba(lpi_x_te_stop_stem_v)[:,1].tolist()

predicts_x_tr_stop_stem_p = predicts_lra.predict_proba(predicts_x_tr_stop_stem_v)[:,1].tolist()
predicts_x_te_stop_stem_p = predicts_lra.predict_proba(predicts_x_te_stop_stem_v)[:,1].tolist()

lpi_tr_df = pd.DataFrame({"TitleAbstract" : lpi_x_tr,
							"TitleAbstract_stop" : lpi_x_tr_stop,
							"TitleAbstract_stem" : lpi_x_tr_stem,
							"TitleAbstract_stop_stem" : lpi_x_tr_stop_stem,
							"p_orig" : lpi_x_tr_stem_p,
							"p_stop" : lpi_x_tr_stop_stem_p})
lpi_te_df = pd.DataFrame({"TitleAbstract" : lpi_x_te,
							"TitleAbstract_stop" : lpi_x_te_stop,
							"TitleAbstract_stem" : lpi_x_te_stem,
							"TitleAbstract_stop_stem" : lpi_x_te_stop_stem,
							"p_orig" : lpi_x_te_stem_p,
							"p_stop" : lpi_x_te_stop_stem_p})

predicts_tr_df = pd.DataFrame({"TitleAbstract" : predicts_x_tr,
							"TitleAbstract_stop" : predicts_x_tr_stop,
							"TitleAbstract_stem" : predicts_x_tr_stem,
							"TitleAbstract_stop_stem" : predicts_x_tr_stop_stem,
							"p_orig" : predicts_x_tr_stem_p,
							"p_stop" : predicts_x_tr_stop_stem_p})
predicts_te_df = pd.DataFrame({"TitleAbstract" : predicts_x_te,
							"TitleAbstract_stop" : predicts_x_te_stop,
							"TitleAbstract_stem" : predicts_x_te_stem,
							"TitleAbstract_stop_stem" : predicts_x_te_stop_stem,
							"p_orig" : predicts_x_te_stem_p,
							"p_stop" : predicts_x_te_stop_stem_p})

lpi_tr_df.to_csv("../Results/glove_stop_lpi_tr_df.csv", index = False)
lpi_te_df.to_csv("../Results/glove_stop_lpi_te_df.csv", index = False)
predicts_tr_df.to_csv("../Results/glove_stop_predicts_tr_df.csv", index = False)
predicts_te_df.to_csv("../Results/glove_stop_predicts_te_df.csv", index = False)



def compare_to_lra(lra_feat_df, stemmed_emb, all_word_set, word_dif_set, dataset):

	lra_feats_df = lra_feats_df.sort_values(by = "LR_A_coefficients", 
											axis = 0, 
											ascending = False)

	lra_top_50 = lra_feats_df["LR_A_features"][0:50].tolist()
	lra_bot_50 = lra_feats_df["LR_A_features"][-50:].tolist()

	stem_word_dif = [LancasterStemmer().stem(unicode(i, 'utf-8')) for i in word_dif_set]
	top_match = [len(set(lra_top_50).intersection(set([x]))) for x in stem_word_dif]
	bot_match = [len(set(lra_bot_50).intersection(set([x]))) for x in stem_word_dif]

	non_glove_df = pd.DataFrame(data = {"Word" : word_dif_set,
									"Stem" : stem_word_dif,
									"Top_Match" : top_match,
									"Bot_Match" : bot_match})

	top_miss = [word for word in all_word_set if word in non_glove_df.loc[non_glove_df['Top_Match'] != 0, "Word"].tolist()]
	bot_miss = [word for word in all_word_set if word in non_glove_df.loc[non_glove_df['Bot_Match'] != 0, "Word"].tolist()]

	return (non_glove_df, top_miss, bot_miss)

	
# Extract top 50
lpi_bot_50_feats = lpi_feats_df["LR_A_features"][-50:].tolist()
predicts_bot_50_feats = predicts_feats_df["LR_A_features"][-50:].tolist()

# Set comparison
len(set(lpi_bot_50_feats).difference(set(stemmed_emb))) # 0
len(set(lpi_bot_50_feats).intersection(set(stemmed_emb))) # 50

len(set(predicts_bot_50_feats).difference(set(stemmed_emb))) # 0
len(set(predicts_bot_50_feats).intersection(set(stemmed_emb))) # 50


# Compare stemmed versions of non-glove words to top 50 words
# stemmed_lpi_dif = [LancasterStemmer().stem(unicode(i, 'utf-8')) for i in lpi_train_dif]
# stemmed_predicts_dif = [LancasterStemmer().stem(unicode(i, 'utf-8')) for i in predicts_train_dif]

lpi_bot_match = [len(set(lpi_bot_50_feats).intersection(set([x]))) for x in stemmed_lpi_dif]
predicts_bot_match = [len(set(predicts_bot_50_feats).intersection(set([x]))) for x in stemmed_predicts_dif]

lpi_bot_non_glove = pd.DataFrame(data = {"Word" : lpi_train_dif,
									"Stem" : stemmed_lpi_dif,
									"Match" : lpi_bot_match})
predicts_bot_non_glove = pd.DataFrame(data = {"Word" : predicts_train_dif,
										"Stem" : stemmed_predicts_dif,
										"Match" : predicts_bot_match})

# Find all word occurances where word is not in GloVe but stem is in top 50
all_bot_lpi_missing = [word for word in all_lpi_words if word in lpi_bot_non_glove.loc[lpi_bot_non_glove['Match'] != 0, "Word"].tolist()]
# ['mutus','muts','muts','generalistic','microbivores','microbivore',
# 'microbivores','microbivores','phylogenetical','pheromonal','pheromonal',
# 'pheromonal','expressional','pheromonal','varients']
all_bot_predicts_missing = [word for word in all_predicts_words if word in predicts_bot_non_glove.loc[predicts_bot_non_glove['Match'] != 0, "Word"].tolist()]
# ['fited','geneity','genotyped','genotyped','genotyped','fitnesses','fitnesses',
# 'fitnesses','fitnesses','divergently','genotyped']



#       Match        Stem            Word
# 407       1         gen    generalistic
# 1360      1      microb     microbivore
# 2181      1     express    expressional
# 2220      1        vary        varients
# 2596      1      microb    microbivores
# 2867      1         mut           mutus
# 3044      1         mut            muts
# 3207      1    pheromon      pheromonal
# 3724      1  phylogenet  phylogenetical


#       Match     Stem         Word
# 70        1      fit    fitnesses
# 725       1  genotyp    genotyped
# 2222      1   diverg  divergently
# 2433      1      fit        fited
# 2534      1      gen      geneity
