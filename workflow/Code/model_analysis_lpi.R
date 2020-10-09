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
cv_metrics_calc <- function(input_df){
    
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
create_AUC_H_df <- function(input_list, input_df){
    # List columns in input_df
    cols <- colnames(input_df)
    
    # Remove unnecessary columns
    # "N_training_docs", "Cost_function", "CV_folds", "Fold",  "CV_Predicted_Probs", "CV_True_Class", "N_test_docs", "Test_Predicted_Probs" "Test_True_Class"
    indices_rm <- which(cols == 'N_training_docs' | cols == 'Cost_function' | cols == 'CV_folds' |
                            cols == 'Fold' | cols == 'CV_Predicted_Probs' | cols == 'CV_True_Class' | cols == 'N_test_docs' |
                            cols == 'Test_Predicted_Probs' | cols == 'Test_True_Class' | cols == 'Duration')
    
    cols <- cols[-indices_rm]
    
    # Add avAUC and avH cols
    cols <- c(cols, c("avAUC", "AUC_loCI", "AUC_upCI", "Train_test_time")) # "avH", "AUC_loCI", "AUC_upCI",
    
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


# Function to analyse influence of ngram range
ngram_range_plots <- function(input_df){
    word_df <- subset(input_df, input_df$Feature == 'word')
    word_df$ID <- seq(1,4)
    plot(word_df$ID, word_df$avAUC, xlab = 'Ngram range', xaxt = 'n', ylab = 'avAUC')
    axis(1, at=1:4, labels = c('(1,1)', '(1,2)', '(1,3)', '(2,2)'))
    
    word_glm <- glm(avAUC ~ Ngram_range, data = word_df)
    word_aov <- aov(avAUC ~ Ngram_range, data = word_df)
    print(summary(word_glm))
    print(summary(word_aov))
    print(TukeyHSD(word_aov))
    
    word_av_df <- ddply(word_df, 'Ngram_range', summarise, mn = mean(avAUC))
    word_av_df$Ngram_range <- factor(word_av_df$Ngram_range, levels = c('(1, 1)', '(1, 2)', '(1, 3)', '(2, 2)'))
    plot(word_av_df$Ngram_range, word_av_df$mn)
    
    char_df <- subset(input_df, input_df$Feature == 'char')
    char_df$ID <- seq(1,4)
    plot(char_df$ID, char_df$avAUC, xlab = 'Ngram range', xaxt = 'n', ylab = 'avAUC')
    axis(1, at=1:4, labels = c('(3,4)', '(3,6)', '(3,8)', '(3,10)'))
    
    char_glm <- glm(avAUC ~ Ngram_range, data = char_df)
    char_aov <- aov(avAUC ~ Ngram_range, data = char_df)
    print(summary(char_glm))
    print(summary(char_aov))
    print(TukeyHSD(char_aov))
    
    char_av_df <- ddply(char_df, 'Ngram_range', summarise, mn = mean(avAUC))
    char_av_df$Ngram_range <- factor(char_av_df$Ngram_range, levels = c('(3, 4)', '(3, 6)', '(3, 8)', '(3, 10)'))
    plot(char_av_df$Ngram_range, char_av_df$mn)
}

# Function to calculate test set metrics for chosen models
test_metrics_calc <- function(metrics_df, metrics_list, models_df){
 
    for (m in 1:dim(models_df)[1]){
        # Select row of metrics df which is relevant
        tmp_rows <- subset(metrics_df, metrics_df$Attribute == models_df$Attribute[m] & 
                               metrics_df$Model_number == models_df$Model_number[m])
        
        tmp_row <- tmp_rows[1,]
        # print (tmp_row)
        
        i <- paste(as.character(models_df$Attribute[m]), as.character(models_df$Model_number[m]), sep = '_')
        # print (i)
        
        # Extract predictions and true lables
        tmp_pred <- str_to_vector(tmp_row$Test_Predicted_Probs)
        tmp_true <- str_to_vector(tmp_row$Test_True_Class)
        
        pred <- prediction(tmp_pred, tmp_true)
        
        # Calculate metrics of interest
        # Add to metrics list
        metrics_list[[i]][['Test_ROC']] <- performance(pred, 'tpr', 'fpr')

        metrics_list[[i]][['Test_AUC']] <- AUC(tmp_pred, tmp_true)
        
    }
    return(metrics_list)
}

# Function to add test auc to df of chosen models
add_test_AUC <- function(models_df, metrics_list){
    models_df['Test_AUC'] <- NA
    
    for (row in 1:dim(models_df)[1]){
        nam <- paste(as.character(models_df$Attribute[row]), as.character(models_df$Model_number[row]), sep = '_')
        models_df$Test_AUC[row] <- metrics_list[[nam]]$Test_AUC
    }
    return(models_df)
}


# Function to automatically plot effects sizes...
effects_plt_binom <- function(input_glm){
    
    eff_df <- data.frame(matrix(nrow = length(summary(input_glm)$coefficients[,1]), ncol = 10))
    
    factors <- names(input_glm$model)[-1]
    
    names(eff_df) <- c('Coeff_Name', 'Estimate', 'Lo_CI', 'Up_CI', 'y', 'p', 'Sig', 'Variable', 'Factor', 'Baseline')
    
    eff_df$Coeff_Name <-  names(input_glm$coefficients)
    
    # Tukey and GLM estimates are equal
    eff_df$Estimate <- summary(input_glm)$coefficients[,1]
    
    # Remove 1st row (intercept)
    eff_df <- eff_df[-1,]
    
    # Add variables and Factors
    for (i in 1:length(eff_df$Coeff_Name)){
        for (j in factors){
            if (length(grep(j, eff_df$Coeff_Name[i])) == 1){
                eff_df$Variable[i] <- strsplit(eff_df$Coeff_Name[i], j)[[1]][2]
                # Add factor
                eff_df$Factor[i] <- j
            }
            else {
                next
            }
        }
    }
    
    # Add baselines
    eff_df$Baseline[which(eff_df$Factor == 'Attribute')] <- 'Title'
    eff_df$Baseline[which(eff_df$Factor == 'Stop_words'| eff_df$Factor == 'Stemmer')] <- 'None'
    eff_df$Baseline[which(eff_df$Factor == 'Feature')] <- 'word'
    eff_df$Baseline[which(eff_df$Factor == 'Weighting')] <- 'tf'
    eff_df$Baseline[which(eff_df$Factor == 'Unknown_words')] <- 'remove'
    
    # Order of coefficients in eff_df should be same as in model summary....
    confints <- confint(input_glm)
    
    eff_df$Lo_CI <- confints[,1][-1]
    eff_df$Up_CI <- confints[,2][-1]
    
    # Define y co-ords
    eff_df$y <- seq(0.75, 0, length.out = dim(eff_df)[1])
    
    # Neaten variable names for consistency with report text
    # Both
    eff_df$Variable[which(eff_df$Variable == 'TitleAbstract')] <- 'Title+Abstract'
    eff_df$Variable[which(eff_df$Variable == 'NLTK_stop_words.txt')] <- 'NLTK'
    # LR
    eff_df$Variable[which(eff_df$Variable == 'english')] <- 'English'
    eff_df$Variable[which(eff_df$Variable == 'over_0.85_df')] <- 'df>0.85'
    eff_df$Variable[which(eff_df$Variable == 'porter')] <- 'Porter'
    eff_df$Variable[which(eff_df$Variable == 'lancaster')] <- 'Lancaster'
    eff_df$Variable[which(eff_df$Variable == 'wnl')] <- 'Lemmatizer'
    eff_df$Variable[which(eff_df$Variable == 'char')] <- 'Character-based'
    # eff_df$Variable[which(eff_df$Variable == 'TfidfVectorizer')] <- 'tf.idf' # expression(tf%.%idf)
    # NN
    eff_df$Variable[which(eff_df$Variable == 'zeros')] <- 'Zeros'
    eff_df$Variable[which(eff_df$Variable == 'rand')] <- 'Random'
    
    # Neaten factor names
    # Both
    eff_df$Factor[which(eff_df$Factor == 'Attribute')] <- 'Data type'
    eff_df$Factor[which(eff_df$Factor == 'Stop_words')] <- 'Stop words'

    # NN
    eff_df$Factor[which(eff_df$Factor == 'Unknown_words')] <- 'Unknown words'
    
    
    # Add significance to effects df
    eff_df$Sig[which(eff_df$Lo_CI < 0 & eff_df$Up_CI > 0)] <- 'NS'
    eff_df$Sig[which(eff_df$Up_CI < 0)] <- 'Negative'
    eff_df$Sig[which(eff_df$Lo_CI > 0)] <- 'Positive'
    
    
    horiz_sep <- c()
    # Find y locations to add horiz lines to separate factors
    for (i in 1:(dim(eff_df)[1]-1)){
        if (eff_df$Factor[i] != eff_df$Factor[i+1]){
            horiz_sep <- c(horiz_sep, (eff_df$y[i]+eff_df$y[i+1])/2) 
        }
    }
    
    # Vector of shapes to use for plotting the different factors
    if ('Unknown words' %in% eff_df$Factor){
        shapes <- c(21, 24, 4)
        height_ <- 0.05
        label_ <- 'b.'
    } else if ('Weighting' %in% eff_df$Factor){
        shapes <- c(21, 22, 23, 24, 25)
        height_ <- 0.02
        label_ <- 'a.'
    }
    
    y_labs <- eff_df$Variable
    
    # Create plot
    p <- ggplot() +
        geom_vline(xintercept = 0, lty = 'dashed', colour = 'grey50') + 
        geom_point(aes(x = Estimate, y = y, colour = Sig, fill = Sig, shape = Factor), 
                   data = eff_df, size = 4) + 
        ###***
        # geom_errorbarh(aes(xmin = Lo_CI, xmax = Up_CI, y = y, colour = Sig, height = height_), data = eff_df) + 
        geom_errorbarh(aes(xmin = Lo_CI, xmax = Up_CI, y = y, colour = Sig, height = 0), 
                       size = 1, data = eff_df) + 
        geom_hline(yintercept = horiz_sep, lty = 'dotted', lwd = 0.4) + 
        ylim(0, 0.75) +
        xlab(expression(paste(beta, ' coefficient estimate'))) +
        ylab('Level') +
        scale_y_continuous(labels = y_labs, breaks = eff_df$y) +
        scale_colour_manual(name = 'Significance',
                            values = c('black', brewer.pal(9, 'YlGn')[7], 'darkred'),  
                            breaks = c('NS', 'Positive', 'Negative'),
                            guide = F) +
        scale_fill_manual(name = 'Significance',
                          values = c('black', brewer.pal(9, 'YlGn')[7], 'darkred'),  
                          breaks = c('NS', 'Positive', 'Negative'), 
                          guide = F) +
        scale_shape_manual(name = 'Factor',
                           values = as.vector(shapes), 
                           breaks = as.vector(unique(eff_df$Factor))) +
        theme_bw() +
        ggtitle(label_) +
        theme(axis.text = element_text(size = 20),  # 16
              axis.title = element_text(size = 24), # 20
              legend.text = element_text(size = 20), # 16
              legend.title = element_text(size = 22), # 18
              panel.grid.minor.y=element_blank(),
              panel.grid.major.y=element_blank(),
              plot.title = element_text(size = 20, face = "bold"))  #18
    
    # plot(p)
    return(p)
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

# Load lpi model data
lr_df <- read.csv('../Results/Model_metrics/LR/lpi_cv_metrics.csv')
nn_df <- read.csv('../Results/Model_metrics/NN/lpi_cv_metrics.csv')

# Calculate metrics
lr_metr_list <- cv_metrics_calc(input_df = lr_df)
nn_metr_list <- cv_metrics_calc(input_df = nn_df)

# Create dfs with AUC cols
lr_full_AUC_df <- create_AUC_H_df(input_list = lr_metr_list, input_df = lr_df)
nn_AUC_df <- create_AUC_H_df(input_list = nn_metr_list, input_df = nn_df)

# Identify ngram range settings to analyse lr models
# library(plyr)
# ngram_range_plots(input_df = lr_full_AUC_df)
# Sig diff for words, bigrams worse 
# No sig diff for char
# Select (1, 2) and (3, 8), best on average
# ngram_range_plots(input_df = lr_hpc_AUC_df)

# Select lr models based on feature specifics
lr_AUC_df <- subset(lr_full_AUC_df, 
                    lr_full_AUC_df$Ngram_range == '(3, 8)' |  #  '(3, 4)'
                    lr_full_AUC_df$Ngram_range == '(1, 2)')


# Set baseline variables...
lr_AUC_df$Attribute <- factor(lr_AUC_df$Attribute, levels = c('Title', 'Abstract', 'TitleAbstract'))
lr_AUC_df$Stop_words <- factor(lr_AUC_df$Stop_words, levels = c('None', 'NLTK_stop_words.txt', 'over_0.85_df'))
lr_AUC_df$Stemmer <- factor(lr_AUC_df$Stemmer, levels = c('None', 'porter', 'lancaster', 'wnl'))
lr_AUC_df$Weighting <- factor(lr_AUC_df$Weighting, levels = c('tf', 'tf-idf'))
lr_AUC_df$Feature <- factor(lr_AUC_df$Feature, levels = c('word', 'char'))
lr_AUC_df$Ngram_range <- factor(lr_AUC_df$Ngram_range, levels = c('(1, 2)', '(3, 8)')) # '(3, 4)'


nn_AUC_df$Attribute <- factor(nn_AUC_df$Attribute, levels = c('Title', 'Abstract', 'TitleAbstract'))
nn_AUC_df$Stop_words <- factor(nn_AUC_df$Stop_words, levels = c('None', 'NLTK_stop_words.txt'))
nn_AUC_df$Unknown_words <- factor(nn_AUC_df$Unknown_words, levels = c('remove', 'zeros', 'rand'))

# Model avAUC as response
# Following code is not run here as only if all classifiers have been fitted will 
# it work properly
lr_glm_AUC_quasibinom <- glm(avAUC ~ Attribute + Stop_words + Stemmer + Feature + Weighting, data = lr_AUC_df, family = "quasibinomial")
nn_glm_AUC_quasibinom <- glm(avAUC ~ Attribute + Stop_words + Unknown_words, data = nn_AUC_df, family = "quasibinomial")

# Predicted avAUC from baseline vs other models of interest
predict.glm(lr_glm_AUC_quasibinom, data.frame("Attribute" = c("Title","TitleAbstract", "Title"),
                                              "Stop_words" = rep("None", 3),
                                              "Stemmer" = rep("None", 3),
                                              "Feature" = c("word", "word", "char"),
                                              "Weighting" = rep("tf", 3)),
            type = "response")

anova(lr_glm_AUC_quasibinom, test = "F")
anova(nn_glm_AUC_quasibinom, test = "F")


# Plot effects sizes
# TODO: rm ends of error bars, thicker...
lr_glm_AUC_quasibinom_plt <- effects_plt_binom(input_glm = lr_glm_AUC_quasibinom)
nn_glm_AUC_quasibinom_plt <- effects_plt_binom(input_glm = nn_glm_AUC_quasibinom)

quasibinom_x_min <- min(ggplot_build(lr_glm_AUC_quasibinom_plt)$layout$panel_scales_x[[1]]$range$range[1],
                   ggplot_build(nn_glm_AUC_quasibinom_plt)$layout$panel_scales_x[[1]]$range$range[1])
quasibinom_x_max <- max(ggplot_build(lr_glm_AUC_quasibinom_plt)$layout$panel_scales_x[[1]]$range$range[2],
                   ggplot_build(nn_glm_AUC_quasibinom_plt)$layout$panel_scales_x[[1]]$range$range[2])

quasibinom_plt <- plot_grid(lr_glm_AUC_quasibinom_plt + xlim(c(quasibinom_x_min,
                                                               quasibinom_x_max)) + 
                                theme(axis.title.x = element_blank(),
                                                    axis.title.y = element_blank()),
                       nn_glm_AUC_quasibinom_plt + xlim(c(quasibinom_x_min,
                                                          quasibinom_x_max)) +
                           theme(axis.title.y = element_blank(),
                                                    axis.title.x = element_blank()), 
                       align = "v", nrow = 2, rel_heights = c(9/14, 5/14))

quasibinom_plt <- grid.arrange(arrangeGrob(quasibinom_plt, 
                                      left = textGrob('Level', gp=gpar(fontsize = 20), rot = 90), # for poster: fontsize = 24
                                      bottom = textGrob(expression(paste(beta, ' coefficient estimate')), gp=gpar(fontsize = 20)))) #24
# Make "Results/Figs" dir
fig_dir <- "../Results/Figs"
if(!dir.exists(fig_dir)){
    dir.create(fig_dir)
}

# ggsave(plot = quasibinom_plt, filename = 'quasibinom_plt1.pdf',
#        path = '../../Results/Figs', 
#        width = 12, height = 9, dpi = 300, device = "pdf")

# Residual plots
# pdf('../Results/Figs/lr_lpi_residuals.pdf')
# par(mfrow=c(2,2))
# plot(lr_glm_AUC_quasibinom)
# dev.off()

# pdf('../Results/Figs/nn_lpi_residuals.pdf')
# par(mfrow=c(2,2))
# plot(nn_glm_AUC_quasibinom)
# dev.off()


# Select models
# Edit models to use - 10th July 2020
# Use best, best without text processing steps, and best without abstract
# Demo importance of abstract compared to text-proc
lr_models_to_use_df  <- subset(lr_AUC_df, lr_AUC_df$avAUC == max(lr_AUC_df$avAUC) |
                                   (lr_AUC_df$Attribute == 'TitleAbstract' & 
                                        lr_AUC_df$Stop_words == 'None' & 
                                        lr_AUC_df$Stemmer == 'None' & 
                                        lr_AUC_df$Feature == 'word' & 
                                        lr_AUC_df$Weighting == 'tf') |
                                   (lr_AUC_df$Attribute == 'Title' & 
                                        lr_AUC_df$Stop_words == 'over_0.85_df' & 
                                        lr_AUC_df$Stemmer == 'lancaster' & 
                                        lr_AUC_df$Feature == 'word' & 
                                        lr_AUC_df$Weighting == 'tf-idf'))

nn_models_to_use_df <- subset(nn_AUC_df, nn_AUC_df$avAUC == max(nn_AUC_df$avAUC) |
                                  (nn_AUC_df$Attribute == 'TitleAbstract' &
                                       nn_AUC_df$Stop_words == 'None' &
                                       nn_AUC_df$Unknown_words == 'remove') |
                              (nn_AUC_df$Attribute == 'Title' &
                                   nn_AUC_df$Stop_words == 'NLTK_stop_words.txt' &
                                   nn_AUC_df$Unknown_words == 'rand'))


# Calculate test AUC/metrics
lr_metr_list <- test_metrics_calc(metrics_df = lr_df, metrics_list = lr_metr_list, models_df = lr_models_to_use_df)
nn_metr_list <- test_metrics_calc(metrics_df = nn_df, metrics_list = nn_metr_list, models_df = nn_models_to_use_df)

# Add test AUC to models_dfs
lr_models_to_use_df <- add_test_AUC(models_df = lr_models_to_use_df, metrics_list = lr_metr_list)
nn_models_to_use_df <- add_test_AUC(models_df = nn_models_to_use_df, metrics_list = nn_metr_list)

# Calculate average ROC

# Sort dfs by avAUC (A models at top)
lr_models_to_use_df <- lr_models_to_use_df[order(-lr_models_to_use_df$avAUC),]
nn_models_to_use_df <- nn_models_to_use_df[order(-nn_models_to_use_df$avAUC),]
# Store data about avAUC, CIs etc
# write.csv(lr_models_to_use_df, '../Results/Model_metrics/LR/lpi_models_to_use.csv')
# write.csv(nn_models_to_use_df, '../Results/Model_metrics/NN/lpi_models_to_use.csv')


# Calculate additional metrics and save
# Create empty df, cols = Classifier, Attribute, Model_num, t1, Prec1, Rec1, F11, t2, Prec2, Rec2, F12
extra_metrics_df <- rbind(lr_models_to_use_df[c('Attribute', 'Model_number', 'Classifier')], 
                        nn_models_to_use_df[c('Attribute', 'Model_number', 'Classifier')])

extra_metrics_df <- acc_prec_rec_F1_extract(input_df = extra_metrics_df, 
                                            lr_df = lr_df, 
                                            nn_df = nn_df, 
                                            lr_list = lr_metr_list, 
                                            nn_list = nn_metr_list)
# write.csv(extra_metrics_df, '../Results/Model_metrics/extra_metrics_lpi.csv')

min(min(extra_metrics_df$cv_Recall_1), 
    min(extra_metrics_df$cv_Precision_1), 
    min(extra_metrics_df$cv_F1_1),
    min(extra_metrics_df$cv_Accuracy_1),
    min(extra_metrics_df$test_Recall_1),
    min(extra_metrics_df$test_Precision_1),
    min(extra_metrics_df$test_F1_1),
    min(extra_metrics_df$test_Accuracy_1))

# extra_metrics_df[c(1,4),]
# Attribute Model_number         Classifier           cv_Recall_2    cv_Precision_2
# 534 TitleAbstract          149 LogisticRegression   0.9761431      0.9370229
# 10  TitleAbstract           18     Neural Network   0.9403579      0.9131274
#     cv_F1_2       cv_Accuracy_2 test_Recall_2        test_Precision_2 test_F1_2
# 534 0.9561831     0.9555336     0.9461538            0.984            0.9647059
# 10  0.9265426     0.9258893     0.8230769            1.000            0.9029536
#           test_Accuracy_2
# 534       0.9645669
# 10        0.9094488

