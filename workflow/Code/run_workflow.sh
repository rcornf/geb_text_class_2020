#!/usr/bin/bash

# Bash code to run manuscript workflow 
printf "Running GEB manuscript workflow."

printf "\n\nPreprocessing and preparing LPD texts."
# python prep_all_texts.py lpi_positives.csv negatives.csv

printf "\n\nPreprocessing and preparing PEDICTS texts."
# python prep_all_texts.py predicts_positives.csv negatives.csv

printf "\n\nFitting various models using LPD texts."
# python train_classifiers.py prepared_lpi_positives.csv prepared_lpi_negatives.csv

printf "\n\nAnalysing the models trained using LPD texts."
Rscript model_analysis_lpi.R

printf "\n\nFitting best models on PREDICTS texts."
# python train_predicts_classifiers.py

printf "\n\nAnalysing models trained using PREDICTS texts."
Rscript model_analysis_predicts.R

printf "\n\nAssessing the model performance on real-world searches."
Rscript search_analysis.R

printf "\n\nDetermining effect of training set size and addition of true-negatives to performance of the best Logistic model."
# Rscript man_class_process.R
# python lr_npos_augment_fit.py
Rscript lr_npos_augment_anaylsis.R

printf "\n\nIdenitfying most highly weighted features of the best Logistic model."
# python lr_feature_extraction.py
Rscript word_clouds.R

# Resampling...
printf "\n\nResampling from pseudo-negatives and assessing impact on model performance."
# python resample_models.py
Rscript resamp_analysis.R

printf "\n\nAssessing coverage of GloVe words for scientific articles used."
# python glove_text_coverage.py
Rscript lra_pred_glove_stop.R

