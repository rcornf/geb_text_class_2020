#!/usr/bin/python

'''Python script for identifying features and their coefficients in the logistic models.
'''

# Import necessary packages
import numpy as np
import pandas as pd 

from sklearn.externals import joblib


# Function to load lr models given database/model info, extract features/weights and save in df
def extract_feats_coeffs(dataset):
	# Create empty dictionary to store features and coefficiens
	feat_coeff_dict = {}

	if dataset == 'lpi':
		# Load models specs 
		model_df = pd.read_csv('../Results/Model_metrics/LR/lpi_models_to_use.csv')

	elif dataset == 'predicts':
		model_df = pd.read_csv('../Results/Model_metrics/LR/predicts_models_to_use.csv')

	model_spec = model_df.loc[0,:]
	model_type = 'LR_A'

	mod_fp = '../Results/Models/LR/model_' + str(dataset) + '_' + str(model_spec['Model_number']) + '_' + str(model_spec['Attribute']) + '_' + str(model_spec['Stop_words']).rstrip('.txt') + '_' + str(model_spec['Stemmer']) + '_' + str(model_spec['Weighting']) + '_' + str(model_spec['Feature']) + '_' + str(model_spec['Ngram_range']) + '_' + str(model_spec['Date_Time']) + '.pkl'
	vect_fp = '../Results/Models/LR/vectorizer_' + str(dataset) + '_' + str(model_spec['Model_number']) + '_' + str(model_spec['Attribute']) + '_' + str(model_spec['Stop_words']).rstrip('.txt') + '_' + str(model_spec['Stemmer']) + '_' + str(model_spec['Weighting']) + '_' + str(model_spec['Feature']) + '_' + str(model_spec['Ngram_range']) + '_' + str(model_spec['Date_Time']) + '.pkl'
		
	lr_model = joblib.load(mod_fp)
	vectorizer_obj = joblib.load(vect_fp)

	# Extract feature names and coefficients
	features = np.array(vectorizer_obj.get_feature_names())
	print(len(features))
	coeffs = lr_model.coef_[0]

	feat_coeff_dict[model_type + '_features'] = features
	feat_coeff_dict[model_type + '_coefficients'] = coeffs

	out_df = pd.DataFrame.from_dict(feat_coeff_dict, orient = 'index').T

	return(out_df)

# Extract features and coefficients
lpi_feats_coeffs_df = extract_feats_coeffs('lpi')
predicts_feats_coeffs_df = extract_feats_coeffs('predicts')

# Save dfs
lpi_feats_coeffs_df.to_csv('../Results/Model_metrics/LR/lpi_feats_coeffs.csv', index = False, encoding = 'utf-8')
predicts_feats_coeffs_df.to_csv('../Results/Model_metrics/LR/predicts_feats_coeffs.csv', index = False, encoding = 'utf-8')

