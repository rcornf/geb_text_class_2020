####
# R script for analysing the AUC scores etc of the resampled models
####

# Clear environment
rm(list = ls())
graphics.off()

# Load packages
# cvAUC
require(cvAUC)
# H measure
require(hmeasure)
# ROCR
require(ROCR)
# ggplot
require(ggplot2)

# Functions
# Function to convert string of numbers to vector
str_to_vector <- function(input_str){
    # Remove [ and ] and ,
    str <- gsub("\\[|\\]|,", "", input_str)
    
    # Separate by ' '
    str <- strsplit(str, ' ')[[1]]
    
    # Create vector to store output
    vect <- rep(99, length(str))
    
    # Convert characters to floats and store in vect
    for (i in 1:length(str)){
        vect[i] <- as.numeric(str[i])
    }
    
    return (vect)
}

# Function to calculate metrics for cv predictions given an input df
cv_metrics_calc_resamp <- function(input_df){
    
    # Convert prob/class cols to character
    input_df$CV_Predicted_Probs <- as.character(input_df$CV_Predicted_Probs)
    input_df$CV_True_Class <- as.character(input_df$CV_True_Class)
    
    input_df['UniqueID'] <- paste(as.character(input_df$Attribute), as.character(input_df$Seed_number), as.character(input_df$Stop_words), sep = '_')
    
    # Create list to store model metrics
    model_metrics_list <- list()
    
    
    # Go through each model 
    for (i in unique(input_df$UniqueID)){
        sub_df <- subset(input_df, input_df$UniqueID == i)
        
        # Create list to store stats for this model
        model_metrics_list[[i]] <- list()
        
        # Create lists/vectors to store values from each fold for cvAUC
        y_pred_list <- list()
        y_true_list <- list()
        fold_id_list <- list()
        
        y_pred_vect <- c()
        y_true_vect <- c()
        fold_id_vect <- c()
        
        h_vect <- c()
        
        # Loop through each cv fold
        for (j in 1:dim(sub_df)[1]){
            
            # Create list to store fold metrics
            model_metrics_list[[i]][[paste('Fold', j, sep = '_')]] <- list()
            
            # Extract useful values from row of interes
            tmp_pred <- str_to_vector(sub_df$CV_Predicted_Probs[j])
            tmp_true <- str_to_vector(sub_df$CV_True_Class[j])
            tmp_fold <- rep(j, length(tmp_true))
            
            # Add values to lists/ vectors for later analysis
            y_pred_list[[j]] <- tmp_pred
            y_true_list[[j]] <- tmp_true
            fold_id_list[[j]] <- tmp_fold
            
            y_pred_vect <- c(y_pred_vect, tmp_pred)
            y_true_vect <- c(y_true_vect, tmp_true)
            fold_id_vect <- c(fold_id_vect, tmp_fold)
            
            model_metrics_list[[i]][[paste('Fold', j, sep = '_')]][['Predictions']] <- tmp_pred
            model_metrics_list[[i]][[paste('Fold', j, sep = '_')]][['True_Labels']] <- tmp_true
            
            if (j == 1){
                # Extract predictions and true lables
                tmp_pred_test <- str_to_vector(sub_df$Test_Predicted_Probs[j])
                tmp_true_test <- str_to_vector(sub_df$Test_True_Class[j])
                
                pred_test <- prediction(tmp_pred_test, tmp_true_test)
                
                model_metrics_list[[i]][['Test_AUC']] <- AUC(tmp_pred, tmp_true)
            }
        }
        
        # Calculate cv AUC metrics and add to list
        model_metrics_list[[i]][['cvAUC']] <- cvAUC(y_pred_list, y_true_list, folds = fold_id_list)
        model_metrics_list[[i]][['cicvAUC']] <- ci.cvAUC(y_pred_vect, y_true_vect, folds = fold_id_vect)
        
        # Calculate av ROC
        cv_pred <- prediction(y_pred_list, y_true_list)
        model_metrics_list[[i]][['cv_ROC']] <- performance(cv_pred, 'tpr', 'fpr')
    }
    
    return (model_metrics_list) 
}

# Function to create AUC/H df
create_resamp_AUC_H_df <- function(input_list, input_df){
    # List columns in input_df
    cols <- colnames(input_df)
    
    # Remove unnecessary columns
    # "N_training_docs", "Classifier", "Cost_function", "CV_folds", "Fold",  "CV_Predicted_Probs", "CV_True_Class", "N_test_docs", "Test_Predicted_Probs" "Test_True_Class"
    indices_rm <- which(cols == 'N_training_docs' | cols == 'Cost_function' | cols == 'CV_folds' |
                            cols == 'Fold' | cols == 'CV_Predicted_Probs' | cols == 'CV_True_Class' | cols == 'N_test_docs' |
                            cols == 'Test_Predicted_Probs' | cols == 'Test_True_Class')
    
    cols <- cols[-indices_rm]

    # Add avAUC and avH cols
    cols <- c(cols, c("avAUC"))
    
    # Create df
    AUC_H_df <- data.frame(matrix(nrow = length(input_list), ncol = length(cols)))
    names(AUC_H_df)  <- cols
    
    df_indices <- seq(1, dim(input_df)[1], 10)
    
    # Extract relevant info from input_list/input_df
    for (i in 1:length(input_list)){
        for (j in 1:length(cols)){
            
            if (cols[j] == 'avAUC'){
                AUC_H_df[i,j] <- input_list[[i]]$cicvAUC$cvAUC
                next
            }
            else{
                AUC_H_df[i,j] <- as.character(input_df[df_indices[i], which(colnames(input_df) == cols[j])])
            }
        }
    }
    
    return (AUC_H_df)
}

min_mean_se_max_resamp <- function(x) {
    df <- data.frame('ymin' = min(x$avAUC),
                     'q2_5' = as.numeric(quantile(x$avAUC, 0.025)),
                     'lower' = mean(x$avAUC) - sd(x$avAUC)/sqrt(length(x$avAUC)),
                     'lo_CI' = mean(x$avAUC) - 1.96*(sd(x$avAUC)/sqrt(length(x$avAUC))),
                     'middle' = mean(x$avAUC),
                     'q50' = as.numeric(quantile(x$avAUC, 0.5)),
                     'up_CI' = mean(x$avAUC) + 1.96*(sd(x$avAUC)/sqrt(length(x$avAUC))),
                     'upper' = mean(x$avAUC) + sd(x$avAUC)/sqrt(length(x$avAUC)),
                     'q97_5' = as.numeric(quantile(x$avAUC, 0.975)),
                     'ymax' = max(x$avAUC))
    
    return(df)
}


####
# Main code
####
# Load resampled data
lr_lpi_resamp_df <- read.csv('../Results/Model_metrics/LR/lpi_resample_metrics.csv')
lr_predicts_resamp_df <- read.csv('../Results/Model_metrics/LR/predicts_resample_metrics.csv')
nn_lpi_resamp_df <- read.csv('../Results/Model_metrics/NN/lpi_resample_metrics.csv')
nn_predicts_resamp_df <- read.csv('../Results/Model_metrics/NN/predicts_resample_metrics.csv')

# Calculate metrics
lr_lpi_resamp_metr_list <- cv_metrics_calc_resamp(input_df = lr_lpi_resamp_df)
lr_predicts_resamp_metr_list <- cv_metrics_calc_resamp(input_df = lr_predicts_resamp_df)
nn_lpi_resamp_metr_list <- cv_metrics_calc_resamp(input_df = nn_lpi_resamp_df)
nn_predicts_resamp_metr_list <- cv_metrics_calc_resamp(input_df = nn_predicts_resamp_df)

# Export to df
lr_lpi_resamp_auc_df <- create_resamp_AUC_H_df(input_list = lr_lpi_resamp_metr_list, input_df = lr_lpi_resamp_df)
lr_predicts_resamp_auc_df <- create_resamp_AUC_H_df(input_list = lr_predicts_resamp_metr_list, input_df = lr_predicts_resamp_df)
nn_lpi_resamp_auc_df <- create_resamp_AUC_H_df(input_list = nn_lpi_resamp_metr_list, input_df = nn_lpi_resamp_df)
nn_predicts_resamp_auc_df <- create_resamp_AUC_H_df(input_list = nn_predicts_resamp_metr_list, input_df = nn_predicts_resamp_df)


# Add dataset col
lr_lpi_resamp_auc_df['Dataset'] <- 'LPD'
lr_predicts_resamp_auc_df['Dataset'] <- 'PREDICTS'
nn_lpi_resamp_auc_df['Dataset'] <- 'LPD'
nn_predicts_resamp_auc_df['Dataset'] <- 'PREDICTS'

lr_lpi_resamp_auc_df['Model'] <- 'LR A'
lr_predicts_resamp_auc_df['Model'] <- 'LR A'
nn_lpi_resamp_auc_df['Model'] <- 'CNN A'
nn_predicts_resamp_auc_df['Model'] <- 'CNN A'

# Rbind
resamp_auc_df <- rbind(lr_lpi_resamp_auc_df[c('Dataset', 'Model', 'avAUC')], 
                          lr_predicts_resamp_auc_df[c('Dataset', 'Model', 'avAUC')],
                          nn_lpi_resamp_auc_df[c('Dataset', 'Model', 'avAUC')],
                       nn_predicts_resamp_auc_df[c('Dataset', 'Model', 'avAUC')])

resamp_bp_df <- ddply(resamp_auc_df, .(Dataset, Model), min_mean_se_max_resamp)

# resamp_bp_df$ymin
# 0.9696655 0.9815266 0.9734725 0.9880587

# Load original model scores
lr_lpi_orig_mod_df <- read.csv('../Results/Model_metrics/LR/lpi_models_to_use.csv')[1,]
lr_predicts_orig_mod_df <- read.csv('../Results/Model_metrics/LR/predicts_models_to_use.csv')[1,]
nn_lpi_orig_mod_df <- read.csv('../Results/Model_metrics/NN/lpi_models_to_use.csv')[1,]
nn_predicts_orig_mod_df <- read.csv('../Results/Model_metrics/NN/predicts_models_to_use.csv')[1,]

lr_lpi_orig_mod_df['Dataset'] <- 'LPD'
lr_predicts_orig_mod_df['Dataset'] <- 'PREDICTS'
nn_lpi_orig_mod_df['Dataset'] <- 'LPD'
nn_predicts_orig_mod_df['Dataset'] <- 'PREDICTS'

lr_lpi_orig_mod_df['Model'] <- 'LR A'
lr_predicts_orig_mod_df['Model'] <- 'LR A'
nn_lpi_orig_mod_df['Model'] <- 'CNN A'
nn_predicts_orig_mod_df['Model'] <- 'CNN A'

orig_mod_df <- rbind(lr_lpi_orig_mod_df[c('Dataset', 'Model', 'avAUC')], 
                        lr_predicts_orig_mod_df[c('Dataset', 'Model', 'avAUC')],
                     nn_lpi_orig_mod_df[c('Dataset', 'Model', 'avAUC')],
                     nn_predicts_orig_mod_df[c('Dataset', 'Model', 'avAUC')])


# Merge original and resampled dfs
resamp_bp_df <- merge(x = resamp_bp_df, y = orig_mod_df[c('Model', 'Dataset', 'avAUC')], by = c('Model', 'Dataset'))

resamp_bp_df$Model <- factor(resamp_bp_df$Model, 
                             levels = c("LR A", "CNN A"))
# Plot variation in avAUC
resamp_plt <- ggplot(data = resamp_bp_df) +
    geom_errorbar(aes(x = Model, ymin = q2_5, ymax = q97_5, width = 0, colour = Dataset), 
                  size = 1,
                  position = position_dodge(width = 0.3),
                  show.legend = F) +
    geom_point(aes(x = Model, y = q50, colour = Dataset), 
               pch = 16, size = 3.5, position = position_dodge(width = 0.3)) +
    geom_point(aes(x = Model, y = avAUC, group = Dataset), 
               pch = 18, colour = 'black', size = 3.5, alpha = 0.7,
               position = position_dodge(width = 0.3)) +
    ylab('Average AUC') +
    xlab('Model') +
    geom_vline(xintercept = 1.5, lty = 'dashed', colour = 'grey50') +
    scale_y_continuous(breaks = c(0.95, 1.00),
                       limits = c(0.95, 1.00)) +
    scale_color_manual(name = 'Indicator dataset',
                       values = c("#E69F00", "#009E73"),
                       breaks = c('LPD', 'PREDICTS')) +
    theme_bw() + 
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 20),
          legend.text = element_text(size = 16),
          legend.title = element_text(size = 18),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

resamp_plt

# ggsave(plot = resamp_plt, filename = 'resamp_plt.pdf',
#        path = '../Results/Figs',
#        width = 8, height = 5, dpi = 300, device = "pdf")


####
# Standard stats
####

range(resamp_auc_df[which(resamp_auc_df$Model == 'LR A' & resamp_auc_df$Dataset == 'LPD'),]$avAUC)
mean(resamp_auc_df[which(resamp_auc_df$Model == 'LR A' & resamp_auc_df$Dataset == 'LPD'),]$avAUC)
sd(resamp_auc_df[which(resamp_auc_df$Model == 'LR A' & resamp_auc_df$Dataset == 'LPD'),]$avAUC)

range(resamp_auc_df[which(resamp_auc_df$Model == 'LR A' & resamp_auc_df$Dataset == 'PREDICTS'),]$avAUC)
mean(resamp_auc_df[which(resamp_auc_df$Model == 'LR A' & resamp_auc_df$Dataset == 'PREDICTS'),]$avAUC)
sd(resamp_auc_df[which(resamp_auc_df$Model == 'LR A' & resamp_auc_df$Dataset == 'PREDICTS'),]$avAUC)


range(resamp_auc_df[which(resamp_auc_df$Model == 'CNN A' & resamp_auc_df$Dataset == 'LPD'),]$avAUC)
mean(resamp_auc_df[which(resamp_auc_df$Model == 'CNN A' & resamp_auc_df$Dataset == 'LPD'),]$avAUC)
sd(resamp_auc_df[which(resamp_auc_df$Model == 'CNN A' & resamp_auc_df$Dataset == 'LPD'),]$avAUC)

range(resamp_auc_df[which(resamp_auc_df$Model == 'CNN A' & resamp_auc_df$Dataset == 'PREDICTS'),]$avAUC)
mean(resamp_auc_df[which(resamp_auc_df$Model == 'CNN A' & resamp_auc_df$Dataset == 'PREDICTS'),]$avAUC)
sd(resamp_auc_df[which(resamp_auc_df$Model == 'CNN A' & resamp_auc_df$Dataset == 'PREDICTS'),]$avAUC)





