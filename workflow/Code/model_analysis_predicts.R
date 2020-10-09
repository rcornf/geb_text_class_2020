#%%%%%%%%%%
# Rscript to analyse LPI models
#%%%%%%%%%%

# Clear working env
rm(list = ls())
graphics.off()

# Import required packages
# ROCR
require(ROCR)
# cvAUC
require(cvAUC)
# H measure
require(hmeasure)
# ggplot
require(ggplot2)
# Load packages for grouping plots
require(grid)
require(gridExtra)
# ddply
require(plyr)
# Colour pallette
require(RColorBrewer)
# cowplot
require(cowplot)


#%%%%%
# Functions
#%%%%%

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
predicts_metrics_calc <- function(input_df){
    
    # Convert prob/class cols to character
    input_df$CV_Predicted_Probs <- as.character(input_df$CV_Predicted_Probs)
    input_df$CV_True_Class <- as.character(input_df$CV_True_Class)
    
    input_df['UniqueID'] <- paste(as.character(input_df$Attribute), as.character(input_df$Model_number), sep = '_')
    
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
            
            if (j == 1){
                # Extract predictions and true lables
                test_pred <- str_to_vector(sub_df$Test_Predicted_Probs[j])
                test_true <- str_to_vector(sub_df$Test_True_Class[j])
                
                pred <- prediction(test_pred, test_true)
                
                # Calculate metrics of interest
                # Add to metrics list
                model_metrics_list[[i]][['Test_ROC']] <- performance(pred, 'tpr', 'fpr')
                model_metrics_list[[i]][['Test_AUC']] <- AUC(test_pred, test_true)
         
            }
            
            # Create list to store fold metrics
            model_metrics_list[[i]][[paste('Fold', j, sep = '_')]] <- list()
            
            # Extract useful values from row of interest
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
        }
        
        # Calculate cv AUC metrics and add to list
        model_metrics_list[[i]][['cvAUC']] <- cvAUC(y_pred_list, y_true_list, folds = fold_id_list)
        model_metrics_list[[i]][['cicvAUC']] <- ci.cvAUC(y_pred_vect, y_true_vect, folds = fold_id_vect)
        
        # Calculate av ROC
        cv_pred <- prediction(y_pred_list, y_true_list)
        model_metrics_list[[i]][['cv_ROC']] <- performance(cv_pred, 'tpr', 'fpr')
        
        # Calculate total train and test time
        model_metrics_list[[i]]['Train_test_time'] <- sum(sub_df$Duration)
    }
    
    return (model_metrics_list) 
}

# Function to create average AUC df
create_predicts_AUC_df <- function(input_list, input_df){
    # List columns in input_df
    cols <- colnames(input_df)
    
    # Remove unnecessary columns
    # "N_training_docs", "Cost_function", "CV_folds", "Fold",  "CV_Predicted_Probs", "CV_True_Class", "N_test_docs", "Test_Predicted_Probs" "Test_True_Class"
    indices_rm <- which(cols == 'N_training_docs' | cols == 'Cost_function' | cols == 'CV_folds' |
                            cols == 'Fold' | cols == 'CV_Predicted_Probs' | cols == 'CV_True_Class' | cols == 'N_test_docs' |
                            cols == 'Test_Predicted_Probs' | cols == 'Test_True_Class' | cols == 'Duration')
    
    cols <- cols[-indices_rm]
    
    # Add avAUC and avH cols
    cols <- c(cols, c("avAUC", "AUC_loCI", "AUC_upCI", "Test_AUC", "Train_test_time")) # "avH", "AUC_loCI", "AUC_upCI",
    
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
            if (cols[j] == 'AUC_loCI'){
                AUC_H_df[i,j] <- input_list[[i]]$cicvAUC$ci[1]
                next
            }
            if (cols[j] == 'AUC_upCI'){
                AUC_H_df[i,j] <- input_list[[i]]$cicvAUC$ci[2]
                next
            }
            if (cols[j] == 'Test_AUC'){
                AUC_H_df[i,j] <- input_list[[i]]$Test_AUC
                next
            }
            if (cols[j] == 'Train_test_time'){
                AUC_H_df[i,j] <- input_list[[i]]$Train_test_time
                next
            }
            
            else{
                AUC_H_df[i,j] <- as.character(input_df[df_indices[i], which(colnames(input_df) == cols[j])])
            }
        }
    }
    
    return (AUC_H_df)
}

# Function to calculate precision, recall and F1 of the selected models using a threshold of 0.5 and the threshold which optimises F1
acc_prec_rec_F1_extract <- function(input_df, lr_df, nn_df, lr_list, nn_list){
    for (i in 1:dim(input_df)[1]){
        
        tmp_name <- paste(input_df$Attribute[i], input_df$Model_number[i], sep = '_')
        
        thresholds <- seq(0, 1, length.out = 1001)
        TP <- c(rep(0, length(thresholds)))
        FP <- c(rep(0, length(thresholds)))
        FN <- c(rep(0, length(thresholds)))
        TN <- c(rep(0, length(thresholds)))
        
        # For cv data, 
        for (j in 1:10){
            
            if (input_df$Classifier[i] == 'LogisticRegression'){
                tmp_pred <- lr_list[[tmp_name]][[paste('Fold', j, sep = '_')]][['Predictions']]
                
                tmp_labels <- lr_list[[tmp_name]][[paste('Fold', j, sep = '_')]][['True_Labels']]
                
            }
            else if (input_df$Classifier[i] == 'Neural Network'){
                tmp_pred <- nn_list[[tmp_name]][[paste('Fold', j, sep = '_')]][['Predictions']]
                
                tmp_labels <- nn_list[[tmp_name]][[paste('Fold', j, sep = '_')]][['True_Labels']]
                
            }
            
            tp_vec <- tmp_pred*tmp_labels
            
            for (k in 1:length(thresholds)){
                tmp_tp <- sum(tp_vec>thresholds[k])
                tmp_fp <- sum(tmp_pred>thresholds[k])-tmp_tp
                tmp_fn <- sum(tmp_labels)-tmp_tp
                tmp_tn <- length(tp_vec) - (tmp_tp+tmp_fp+tmp_fn)
                
                TP[k] <- TP[k]+tmp_tp
                FP[k] <- FP[k]+tmp_fp
                FN[k] <- FN[k]+tmp_fn
                TN[k] <- TN[k]+tmp_tn
            }
        }
        
        # Calculate metrics for thresholds
        F1 <- (2*TP)/((2*TP)+FP+FN)
        Recall <- TP/(TP+FN)
        Precision <- TP/(TP+FP)
        Accuracy <- (TP+TN)/(TP+TN+FP+FN)
        # F1 <- 2*((Precision*Recall)/(Precision+Recall))
        
        # Calc av prec, recall, F1 using t=0.5
        input_df$t_1[i] <- 0.5
        input_df$cv_Recall_1[i] <- Recall[which(thresholds == 0.5)]
        input_df$cv_Precision_1[i] <- Precision[which(thresholds == 0.5)]
        input_df$cv_F1_1[i] <- F1[which(thresholds == 0.5)]
        input_df$cv_Accuracy_1[i] <- Accuracy[which(thresholds == 0.5)]
        
        # Find max F1 and corresponding t
        # Calc av prec, rec, F1 for this
        input_df$t_2[i] <- min(thresholds[which(F1 == max(F1))])
        input_df$cv_Recall_2[i] <- Recall[which(thresholds == input_df$t_2[i])]
        input_df$cv_Precision_2[i] <- Precision[which(thresholds == input_df$t_2[i])]
        input_df$cv_F1_2[i] <- F1[which(thresholds == input_df$t_2[i])]
        input_df$cv_Accuracy_2[i] <- Accuracy[which(thresholds == input_df$t_2[i])]
        
        # Test set values using 0.5 and optimal thresholds
        # Depending on the source model type, extract row od test data from relevant df 
        if (input_df$Classifier[i] == 'LogisticRegression'){
            test_row <- subset(lr_df, lr_df$Attribute == input_df$Attribute[i] & 
                                   lr_df$Model_number == input_df$Model_number[i])[1,]
        }
        
        else if (input_df$Classifier[i] == 'Neural Network'){
            test_row <- subset(nn_df, nn_df$Attribute == input_df$Attribute[i] & 
                                   nn_df$Model_number == input_df$Model_number[i])[1,]
        }
        
        # Extract predictions and true values
        test_pred <- str_to_vector(test_row$Test_Predicted_Probs)
        test_true <- str_to_vector(test_row$Test_True_Class)
        
        
        # multiply together
        test_tp_vec <- test_pred*test_true
        
        # calc tp, fp, fn
        test_tp_1 <- sum(test_tp_vec>0.5)
        test_fp_1 <- sum(test_pred>0.5)-test_tp_1
        test_fn_1 <- sum(test_true)-test_tp_1
        test_tn_1 <- length(test_tp_vec) - (test_tp_1+test_fp_1+test_fn_1)
        
        # Calculate metrics
        F1_1 <- (2*test_tp_1)/((2*test_tp_1)+test_fp_1+test_fn_1)
        Recall_1 <- test_tp_1/(test_tp_1+test_fn_1)
        Precision_1 <- test_tp_1/(test_tp_1+test_fp_1)
        Accuracy_1 <- (test_tp_1+test_tn_1)/(test_tp_1+test_tn_1+test_fp_1+test_fn_1)
        
        # Output  
        input_df$test_Recall_1[i] <- Recall_1
        input_df$test_Precision_1[i] <- Precision_1
        input_df$test_F1_1[i] <- F1_1
        input_df$test_Accuracy_1[i] <- Accuracy_1
        
        # calc tp, fp, fn using alternative t
        test_tp_2 <- sum(test_tp_vec>input_df$t_2[i])
        test_fp_2 <- sum(test_pred>input_df$t_2[i])-test_tp_2
        test_fn_2 <- sum(test_true)-test_tp_2
        test_tn_2 <- length(test_tp_vec) - (test_tp_2+test_fp_2+test_fn_2)
        
        # Calculate metrics
        F1_2 <- (2*test_tp_2)/((2*test_tp_2)+test_fp_2+test_fn_2)
        Recall_2 <- test_tp_2/(test_tp_2+test_fn_2)
        Precision_2 <- test_tp_2/(test_tp_2+test_fp_2)
        Accuracy_2 <- (test_tp_2+test_tn_2)/(test_tp_2+test_tn_2+test_fp_2+test_fn_2)
        
        # Output
        input_df$test_Recall_2[i] <- Recall_2
        input_df$test_Precision_2[i] <- Precision_2
        input_df$test_F1_2[i] <- F1_2
        input_df$test_Accuracy_2[i] <- Accuracy_2
        
    }
    
    return(input_df)
}

# Function to create df of cv AUC scores for specified models 
create_cvAUC_df <- function(input_list, input_df){
    # Create df (ncol = 4 (Attribute, Model_number, Fold, AUC), nrow = df*10)
    out_df <- data.frame(matrix(nrow = dim(input_df)[1]*10, ncol = 5))
    names(out_df) <- c('Attribute', 'Model_number', 'Classifier', 'Fold', 'AUC')
    
    out_df$Fold <- seq(1, 10, by = 1)
    
    input_df['Unique_ID'] <- paste(as.character(input_df$Attribute), as.character(input_df$Model_number), sep = '_')
    
    df_indexes <- seq(1, dim(input_df)[1]*10, by = 10)
    
    for (i in 1:dim(input_df)[1]){
        out_df$Attribute[df_indexes[i]:(df_indexes[i]+9)] <- as.character(input_df$Attribute[i])
        out_df$Model_number[df_indexes[i]:(df_indexes[i]+9)] <- input_df$Model_number[i]
        out_df$Classifier[df_indexes[i]:(df_indexes[i]+9)] <- input_df$Classifier[i]
        out_df$AUC[df_indexes[i]:(df_indexes[i]+9)] <- input_list[[input_df$Unique_ID[i]]]$cvAUC$fold.AUC
    }
    
    return(out_df)
}

#%%%%%
# Main Code
#%%%%%

# Load predicts data
lr_predicts_df <- read.csv('../Results/Model_metrics/LR/predicts_cv_metrics.csv')
nn_predicts_df <- read.csv('../Results/Model_metrics/NN/predicts_cv_metrics.csv')

# Calculate metrics
lr_predicts_metr_list <- predicts_metrics_calc(input_df = lr_predicts_df)
nn_predicts_metr_list <- predicts_metrics_calc(input_df = nn_predicts_df)

# Add metrics to df
lr_predicts_AUC_df <- create_predicts_AUC_df(input_list = lr_predicts_metr_list, input_df = lr_predicts_df)
nn_predicts_AUC_df <- create_predicts_AUC_df(input_list = nn_predicts_metr_list, input_df = nn_predicts_df)

# Save for use in resamp_analysis
write.csv(lr_predicts_AUC_df, '../Results/Model_metrics/LR/predicts_models_to_use.csv')
write.csv(nn_predicts_AUC_df, '../Results/Model_metrics/NN/predicts_models_to_use.csv')

# Extra metrics
predicts_extra_metrics_df <- rbind(lr_predicts_AUC_df[c('Attribute', 'Model_number', 'Classifier')], 
                                   nn_predicts_AUC_df[c('Attribute', 'Model_number', 'Classifier')])

predicts_extra_metrics_df <- acc_prec_rec_F1_extract(input_df = predicts_extra_metrics_df, 
                                                     lr_df = lr_predicts_df, 
                                                     nn_df = nn_predicts_df, 
                                                     lr_list = lr_predicts_metr_list, 
                                                     nn_list = nn_predicts_metr_list)
# write.csv(predicts_extra_metrics_df, '../Results/Model_metrics/extra_metrics_predicts.csv')

min(min(predicts_extra_metrics_df$cv_Recall_1), 
    min(predicts_extra_metrics_df$cv_Precision_1), 
    min(predicts_extra_metrics_df$cv_F1_1),
    min(predicts_extra_metrics_df$cv_Accuracy_1),
    min(predicts_extra_metrics_df$test_Recall_1),
    min(predicts_extra_metrics_df$test_Precision_1),
    min(predicts_extra_metrics_df$test_F1_1),
    min(predicts_extra_metrics_df$test_Accuracy_1))
# LRA/NNA exceed 0.89 for all
# min(predicts_extra_metrics_df[c(1,4),c(5:8,10:21)])
predicts_extra_metrics_df[c(1,4),c(3,5:8,10:21)]

#           Classifier cv_Recall_1 cv_Precision_1   cv_F1_1 cv_Accuracy_1 cv_Recall_2
# 1 LogisticRegression   0.9549763      0.9549763 0.9549763     0.9556593   0.9810427
# 4     Neural Network   0.9691943      0.9008811 0.9337900     0.9323221   0.9526066
#   cv_Precision_2   cv_F1_2 cv_Accuracy_2 test_Recall_1 test_Precision_1 test_F1_1
# 1      0.9430524 0.9616725     0.9614936     0.9385965        0.9553571 0.9469027
# 4      0.9284065 0.9403509     0.9404901     0.9736842        0.9098361 0.9406780
#   test_Accuracy_1 test_Recall_2 test_Precision_2 test_F1_2 test_Accuracy_2
# 1       0.9441860     0.9473684        0.9230769 0.9350649       0.9302326
# 4       0.9348837     0.9561404        0.9159664 0.9356223       0.9302326


# Compare selected models for significant difference
# Extract fold AUC scores
lr_predicts_cv_AUC_df <- create_cvAUC_df(input_list = lr_predicts_metr_list, input_df = lr_predicts_AUC_df)
nn_predcits_cv_AUC_df <- create_cvAUC_df(input_list = nn_predicts_metr_list, input_df = nn_predicts_AUC_df)

# Combine dfs and give unique IDs
predicts_cv_AUC_df <- rbind(lr_predicts_cv_AUC_df, 
                            nn_predcits_cv_AUC_df)
predicts_cv_AUC_df['UniqueID'] <- paste(as.character(predicts_cv_AUC_df$Attribute), 
                                        as.character(predicts_cv_AUC_df$Model_number), 
                                        as.character(predicts_cv_AUC_df$Classifier), 
                                        sep = '_')

# ANOVA
# summary(aov(AUC ~ UniqueID, data = predicts_cv_AUC_df))
# summary(glm(AUC ~ UniqueID, data = predicts_cv_AUC_df))
# # Significant 
# 
# kruskal.test(x=predicts_cv_AUC_df$AUC, g=factor(predicts_cv_AUC_df$UniqueID))
# library(pgirmess)
# kruskalmc(predicts_cv_AUC_df$AUC, factor(predicts_cv_AUC_df$UniqueID), probs = 0.05)
# 
# summary(glm(AUC ~ UniqueID, data = predicts_cv_AUC_df, family = "binomial"))  # NS
# summary(glm(AUC ~ UniqueID, data = predicts_cv_AUC_df, family = "quasibinomial")) # Sig
# 
# summary(glm(AUC ~ UniqueID, data = predicts_cv_AUC_df[(predicts_cv_AUC_df$Model_number==149)|
#                                                            (predicts_cv_AUC_df$Model_number==18),], family = "binomial"))
# summary(glm(AUC ~ UniqueID, data = predicts_cv_AUC_df[(predicts_cv_AUC_df$Model_number==149)|
#                                                           (predicts_cv_AUC_df$Model_number==18),], family = "quasibinomial"))
# 
# TukeyHSD(aov(AUC ~ UniqueID, data = predicts_cv_AUC_df))

# Load lpi data summary for plots
lr_lpi_models_df <- read.csv('../Results/Model_metrics/LR/lpi_models_to_use.csv')
nn_lpi_models_df <- read.csv('../Results/Model_metrics/NN/lpi_models_to_use.csv')

# Rbind dfs
bp_df <- rbind(lr_lpi_models_df[c('Attribute', 'Model_number', 'Classifier', 'avAUC', 'AUC_loCI', 'AUC_upCI', 'Test_AUC')],
               nn_lpi_models_df[c('Attribute', 'Model_number', 'Classifier', 'avAUC', 'AUC_loCI', 'AUC_upCI', 'Test_AUC')],
               lr_predicts_AUC_df[c('Attribute', 'Model_number', 'Classifier', 'avAUC', 'AUC_loCI', 'AUC_upCI', 'Test_AUC')],
               nn_predicts_AUC_df[c('Attribute', 'Model_number', 'Classifier', 'avAUC', 'AUC_loCI', 'AUC_upCI', 'Test_AUC')])

# Add dataset and model columns
bp_df['Model'] <- c('LR A', 'LR B', 'LR C', 'CNN A', 'CNN B', 'CNN C')
bp_df['Dataset'] <- c(rep('LPD', 6), rep('PREDICTS', 6))

# Combined plots
bp_df$Model <- factor(bp_df$Model,
                      levels = c('LR A', 'LR B', 'LR C', 
                                 'CNN A', 'CNN B', 'CNN C'))

bp <- ggplot(data = bp_df) +
    geom_point(aes(x = Model, y = avAUC, colour = Dataset), 
               pch = 16, size = 3.5, position = position_dodge(width = 0.5)) + 
    geom_errorbar(aes(x = Model, ymin = AUC_loCI, ymax = AUC_upCI, 
                      width = 0, colour = Dataset), 
                  size = 1, show.legend = F,
                  position = position_dodge(width = 0.5)) + 
    geom_point(aes(x = Model, y = Test_AUC, colour = Dataset, group = Dataset), 
               pch = 18, size = 3.5, alpha = 0.7,
               position = position_dodge(width = 0.5)) + 
    xlab('Model') +
    ylab('AUC') +
    scale_x_discrete(labels = c('LR A', 'LR B', 'LR C', 'CNN A', 'CNN B', 'CNN C')) +
    scale_y_continuous(breaks = c(0.90, 0.95, 1.00), limits = c(0.90, 1.00)) +
    geom_vline(xintercept = 3.5, lty = 'dashed', colour = 'grey50') + 
    scale_color_manual(name = 'Indicator dataset',
                       values = c("#E69F00", "#009E73"),
                       breaks = c('LPD', 'PREDICTS')) +
    theme_bw() + 
    theme(axis.text = element_text(size = 16), 
          axis.title = element_text(size = 20), 
          legend.text = element_text(size = 16), 
          legend.title = element_text(size = 18),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) # 22 

plot(bp)

# Make "Results/Figs" dir
fig_dir <- "../Results/Figs"
if(!dir.exists(fig_dir)){
    dir.create(fig_dir)
}

# ggsave(plot = bp, filename = 'cvAUC.pdf', path = fig_dir,
#        width = 12, height = 6, dpi = 300, device = "pdf")

