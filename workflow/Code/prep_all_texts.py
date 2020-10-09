#!/usr/bin/python

''' 
Python script to pre-process texts and generate negatives from downloaded records
Creates files for use in model training etc

Arguments: 	Provide either 0 or 2. 
			If zero are provided then the script processes the LPI positives and 125000 Ecology texts from NCBI.
			If providing 2, first should be the raw positive texts (csv), the second should be the raw unclassified texts (csv).

Inputs:		2 csv files, positives, unclassifieds/pseudonegatives.
Outputs:	2 csv files, positives, pseudonegatives
'''

# Import required functions/modules 
import os
import sys
import glob
import pandas as pd
import random
import datetime
import numpy as np
import re
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer


#####
# Functions
#####

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
	return [regex.sub(replacement, str(element)) for element in array]


# Function for removing punctution from input string 
def rm_punctuation(string, replacement=' '):    
    for p in set(list(punctuation)): # - set(list(exclude)):
        string = string.replace(p, replacement)

    string = ' '.join(string.split())  # Remove excess whitespace
    return string


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
	
	return (output)


# Function to load and preprocess text data for subsequent analysis
def process_text_data(in_df):

	# Remove duplicates
	mod_df = in_df.drop_duplicates(subset = ['Journal', 'Title', 'Abstract']) #'Source'

	# Remove rows where there are NA/NaN in all columns
	mod_df = mod_df.dropna(how='all')

	# Drop articles with missing titles or abstracts
	mod_df = mod_df.dropna(subset = ['Title'])
	mod_df = mod_df.dropna(subset = ['Abstract'])
	# Reset index
	mod_df = mod_df.reset_index()

	# Convert title and abstract columns to lists 
	title_list = mod_df['Title'].tolist()
	abstract_list = mod_df['Abstract'].tolist()

	# Process text to standard, lowercase
	proc_title_list = text_preprocessing(title_list)
	proc_abstract_list = text_preprocessing(abstract_list)

	if 'PMID' in mod_df.columns:
		# Create df to store this processed text
		out_df = pd.DataFrame({'Journal' : mod_df['Journal'],
								'Title' : proc_title_list,
								'Abstract' : proc_abstract_list,
								'PMID' : mod_df['PMID']})

	else:
		# Create df to store this processed text
		out_df = pd.DataFrame({'Journal' : mod_df['Journal'],
								'Title' : proc_title_list,
								'Abstract' : proc_abstract_list})

	# Drop those where title and abstract are same
	out_df = out_df.drop_duplicates(subset = ['Title', 'Abstract'])

	return out_df



# Function to calulate simlarity metrics and return max score per classified text
def calc_similarity(positive_df, check_df, attribute):
	# Vectorize positive texts
	pos_tfidf = vectorizer.fit_transform(positive_df[attribute].tolist())

	# Convert check_df to list
	check_list = check_df[attribute].tolist()

	max_scores = []

	# Split check_df into smaller chunks
	tmp_indexes = np.arange(0, check_df.shape[0]/1000+1)
	tmp_indexes = np.multiply(tmp_indexes, 1000)
	tmp_indexes = tmp_indexes.tolist()
	tmp_indexes.append(check_df.shape[0]+1)

	# For each chunk, calc cosine simlarity and add max scores to list 
	for i in range(len(tmp_indexes)-1):
		tmp_list = check_list[tmp_indexes[i]: tmp_indexes[i+1]]
		
		# Vectorize classified texts
		check_tfidf = vectorizer.transform(tmp_list)
		# Caclulate cosine similarity matrix/array
		cos_sim_array = np.multiply(pos_tfidf, check_tfidf.T).A
		# Identify maximum similarity score per classified text
		max_cos_sim_array = np.max(cos_sim_array, axis = 0)
		max_scores.append(max_cos_sim_array)

	# unpack arrays
	max_scores = np.concatenate(max_scores, axis = 0)	

	return max_scores



# Function to process similarity scores and add to df
def prepare_texts(pos_df, check_df, LPI_df = None):

	print ('Initial number of records in dowloaded papers: ' + str(check_df.shape[0]))

	attributes = ['Title', 'Abstract']

	# Loop through both attributes
	for attrib in attributes:

		print 'Calculating similarity scores for ' + str(attrib)

		####
		# Section used to find non-perfect matches, however these have been identified
		# by PMID and manually removde through subsequent code
		####

		# Calculate max similarity socres 
		# sim_scores1 = calc_similarity(pos_df, check_df, attrib)
		
		# Create new column name based on attribute
		# col_name1 = 'Cosine_similarity_' + str(attrib) + '_positives'
		
		# Add similarity score to df
		# check_df[col_name1] = sim_scores1
		

		# if LPI_df is not None:
			# sim_scores2 = calc_similarity(LPI_df, check_df, attrib)
			# col_name2 = 'Cosine_similarity_' + str(attrib) + '_LPI_bib'
			# check_df[col_name2] = sim_scores2


		perf_match_pos = []
		perf_match_LPI_bib = []

		for i in check_df[attrib].tolist():

			if (i in pos_df[attrib].tolist()):
				perf_match_pos.append(1)
			else:
				perf_match_pos.append(0)


			if LPI_df is not None:
				if i in LPI_df[attrib].tolist():
					perf_match_LPI_bib.append(1)
				else:
					perf_match_LPI_bib.append(0)

		# Create new column name based on attribute
		col_name3 = 'Perfect_match_' + str(attrib) + '_positives'
		
		# Add match info score to df
		check_df[col_name3] = perf_match_pos		

		if LPI_df is not None:
			# Create new column name based on attribute
			col_name4 = 'Perfect_match_' + str(attrib) + '_LPI_bib'

			# Add match info score to df
			check_df[col_name4] = perf_match_LPI_bib

			# Add column indicating if perfect match to either LPI library
			col_name5 = 'Perfect_match_' + str(attrib)
			check_df[col_name5] = check_df[col_name3] + check_df[col_name4]


		elif LPI_df is None:
			# Add column indicating if perfect match to either LPI library
			col_name5 = 'Perfect_match_' + str(attrib)
			check_df[col_name5] = check_df[col_name3]

	check_df['Perfect_match_either'] = check_df['Perfect_match_Title'] + check_df['Perfect_match_Abstract']

	# Remove perfect matches
	non_matched_df = check_df.loc[check_df['Perfect_match_either'].astype('int') == 0,]

	# Rm checked paper for lpi
	if LPI_df is not None:
		non_matched_df = non_matched_df.loc[non_matched_df['PMID'].astype('int') != 20121838,]
		non_matched_df = non_matched_df.loc[non_matched_df['PMID'].astype('int') != 17531041,]


	non_matched_df = non_matched_df.reset_index()

	print ('Number of records in dowloaded texts after removal of duplicates etc.: ' + str(non_matched_df.shape[0]))

	# Reduce dfs to columns of interest/importance
	pos_df = pos_df[['Title', 'Abstract']]
	non_matched_df = non_matched_df[['Title', 'Abstract', 'PMID']]

	# Remove records with title/abstracts which are too short
	pos_df = pos_df.loc[pos_df['Abstract'].str.len()>=300,]
	pos_df = pos_df.loc[pos_df['Title'].str.len()>=10,]

	non_matched_df = non_matched_df.loc[non_matched_df['Abstract'].str.len()>=300,]
	non_matched_df = non_matched_df.loc[non_matched_df['Title'].str.len()>=10,]

	####
	# Need to modify here if don't want to sample negatives from a set of unclassfied papers.
	# e.g.
	# neg_df = non_matched_df
	# neg_df = neg_df.reset_index(drop=True)
	#### 

	# Sample negatives
	neg_df = non_matched_df.sample(n = 5000, random_state = 1)

	# Reset inexes
	pos_df = pos_df.reset_index(drop=True)
	neg_df = neg_df.reset_index(drop=True)

	print ('Number of positives: ' + str(pos_df.shape[0]))
	print ('Number of negatives: ' + str(neg_df.shape[0]))

	return pos_df, neg_df



#####
# Variables
#####

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
slash_percent_ex = re.compile(r'\\\\\\\\%')

foreign_symbol_ex = re.compile(r'\\\\\\\\.{\w}')
brackets_letter_ex = re.compile(r'{\w}')
letter_ex = re.compile(r'\w')

weird_symbol_ex = re.compile(r'\\\\\\\\.{\\\\\\\\\w}')
slash_letter_ex = re.compile(r'\w}')

# Create vectorizer
vectorizer = TfidfVectorizer(analyzer = 'word')


#####
# Main code
#####

if __name__ == '__main__':

	# print (len(sys.argv))
	
	if (len(sys.argv) == 3):
		print ('\nPreparing specified input files.')
		# Assign file arguments to objects 
		p_file = str(sys.argv[1])
		d_file = str(sys.argv[2])
		print ('Positive records: ' + p_file)
		print ('Downloaded records: ' + d_file)

	elif (len(sys.argv) == 1):
		print('\nNo input files provided, running default preparation of LPI texts.')
		p_file = 'lpi_positives.csv'
		d_file = 'negatives.csv'
		print ('Positive records: ' + p_file)
		print ('Downloaded records: ' + d_file)

	elif (len(sys.argv) == 2):
		print('\nSingle input file provided, preprocessing only.')
		file = str(sys.argv[1])
		tmp_df = pd.read_csv('../Data/Text/' + file)
		tmp_df = process_text_data(tmp_df)
		print('Saving processed texts to csv.')
		tmp_df.to_csv('../Data/Text/prepared_' + file) 
		sys.exit('Text preprocessing completed!')

	elif ((len(sys.argv) != 1) | (len(sys.argv) != 3) | (len(sys.argv) != 2)):
		sys.exit('Error: incorrect number of files specified!')

	# Load specified files...
	p_df = pd.read_csv('../Data/Text/' + p_file)
	d_df = pd.read_csv('../Data/Text/' + d_file)

	# Process text
	print ('Preprocessing texts...')
	p_df = process_text_data(p_df)
	d_df = process_text_data(d_df)

	# Prepare texts
	print ('Checking for, and removing, positive texts in downloaded texts...')
	# If lpi data
	if (p_file.find('lpi') != -1):
		lpi_bibtex_df = pd.read_csv('../Data/Text/LPI_bibtex_info.csv')

		# prep using lpi_bibtex_df as extra info
		p_df, n_df = prepare_texts(pos_df = p_df, check_df = d_df, LPI_df = lpi_bibtex_df)
		# Save outputs
		print ('Saving prepared LPI text files.')
		p_df.to_csv('../Data/Text/prepared_' + str(p_file), encoding = 'utf-8', index = False)
		n_df.to_csv('../Data/Text/prepared_lpi_negatives.csv', encoding = 'utf-8', index = False)
		
	elif (p_file.find('predicts') != -1):
		# prep using positives alone
		p_df, n_df = prepare_texts(pos_df = p_df, check_df = d_df, LPI_df = None)		
		# Save outputs
		print ('Saving prepared PREDICTS text files.')
		p_df.to_csv('../Data/Text/prepared_' + str(p_file), encoding = 'utf-8', index = False)
		n_df.to_csv('../Data/Text/prepared_predicts_negatives.csv', encoding = 'utf-8', index = False)

	else:
		# Determine name of dataset using input positives
		dataset = str(sys.argv[1]).split('_')[0]
		# Prepare texts 
		p_df, n_df = prepare_texts(pos_df = p_df, check_df = d_df, LPI_df = None)
		# Save outputs
		print ('Saving prepared ' + str(dataset) + ' text files.')
		p_df.to_csv('../Data/Text/prepared_' + str(p_file), encoding = 'utf-8', index = False)
		n_df.to_csv('../Data/Text/prepared_' + str(dataset) + '_negatives.csv', encoding = 'utf-8', index = False)

	sys.exit('Text preparation complete!')



