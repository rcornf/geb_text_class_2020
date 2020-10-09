##%%%%%%%%%%
# Rscript to process manual classification of papers for subsequent analysis on 
# the impact of adding these texts to the training data....
##%%%%%%%%%%

rm(list = ls())
graphics.off()


##%%%%%
# Functions
##%%%%%

# Function to load files and process manual classification data for analysis
load_process <- function(file_list, database){
    # Create df to store output
    data_df <- data.frame()
    
    for (f in file_list){
        if (database == "lpd"){
            tmp_df <- read.csv(paste(lpd_input_dir, f, sep = ""),
                               stringsAsFactors = F)
        }
        else if (database == "predicts"){
            tmp_df <- read.csv(paste(predicts_input_dir, f, sep = ""),
                               stringsAsFactors = F)
        }
        # Use mod columns if present
        if ("Relevance_std_mod" %in% names(tmp_df)){
            tmp_df$Relevance_std <- tmp_df$Relevance_std_mod
            tmp_df$Timespan_std <- tmp_df$Timespan_std_mod
        }
        if ("Relevance_ranked_mod" %in% names(tmp_df)){
            tmp_df$Relevance_ranked <- tmp_df$Relevance_ranked_mod
            tmp_df$Timespan_ranked <- tmp_df$Timespan_ranked_mod
        }
        
        # Retain columns of interest
        if (database == "lpd"){
            tmp_df <- tmp_df[, lpd_keep_cols]
            # Rm rows where papers are in training data
            if (sum(tmp_df$Present_in_training_docs == 1, na.rm = T)>=1){
                idx <- which(tmp_df$Present_in_training_docs == 1)
                tmp_df <- tmp_df[-idx,]
            }
            # Add search rank
            tmp_df$Search_rank <- tmp_df$Scopus_index
            # Add search term col
            tmp_df$Search <- strsplit(strsplit(f, "_scopus")[[1]][1], "mod_")[[1]][2]
        }
        else if (database == "predicts"){
            tmp_df <- tmp_df[, predicts_keep_cols]
            # Rm rows where papers are in training data
            if (sum(tmp_df$In_predicts_training_docs == 1, na.rm = T)>=1){
                idx <- which(tmp_df$In_predicts_training_docs == 1)
                tmp_df <- tmp_df[-idx,]
            }
            tmp_df$Search_rank <- tmp_df$WoK_Index
            # Add search term col
            tmp_df$Search <- strsplit(strsplit(f, "_1")[[1]][1], "mod_")[[1]][2]
        }
        
        # Binary values for classifications
        # If '1' present, set to 1, else set to 0
        tmp_df$Relevance <- tmp_df$Relevance_std_bin <- tmp_df$Relevance_ranked_bin <- NA
        
        tmp_df$Relevance_std_bin[!is.na(tmp_df$Timespan_std)] <- 0
        tmp_df$Relevance_std_bin[grep("1", tmp_df$Relevance_std)] <- 1
        
        tmp_df$Relevance_ranked_bin[!is.na(tmp_df$Timespan_ranked)] <- 0
        tmp_df$Relevance_ranked_bin[grep("1", tmp_df$Relevance_ranked)] <- 1
        
        # Create a single, relevant column 
        tmp_df$Relevance <- tmp_df$Relevance_std_bin
        tmp_df$Relevance[is.na(tmp_df$Relevance)] <- tmp_df$Relevance_ranked_bin[is.na(tmp_df$Relevance)]
        
        # Drop rows where papers weren't classified
        tmp_df <- tmp_df[!is.na(tmp_df$Relevance),]
        # Rbind 
        data_df <- rbind(data_df, tmp_df)
    }
    return(data_df)
}


#%%%%
# Constants
#%%%%

lpd_input_dir <- "../Results/Scopus_lpi_manual/"
predicts_input_dir <- "../Results/WoK_predicts/"

lpd_files <- list.files(path = lpd_input_dir,
                        pattern = 'mod_')
predicts_files <- list.files(path = predicts_input_dir,
                             pattern = "mod_")

lpd_keep_cols <- c("Authors", "Title", "Year", "Source.title", "DOI", 
                   "Link", "Abstract", "Document.Type", "Source", "EID", "p", 
                   "Relevance_std", "Relevance_ranked", "Timespan_std", "Timespan_ranked",
                   "Scopus_index", "Present_in_training_docs")
predicts_keep_cols <- c("AU", "SO", "TI", "AB", "Relevance_std", "Timespan_std", 
                        "Relevance_ranked", "Timespan_ranked", "PY", "UT",
                        "WoK_Index", "p", "In_predicts_training_docs")


#%%%%
# Main code
#%%%%

# Load and process data
lpd_data <- load_process(file_list = lpd_files, database = "lpd")
predicts_data <- load_process(file_list = predicts_files, database = "predicts")

lpd_df <- lpd_data[,c("Search", "Title", "Abstract", "DOI", "EID", "Scopus_index", "p", "Relevance")]
predicts_df <- predicts_data[,c("Search", "TI", "AB", "UT", "WoK_Index", "p", "Relevance")]

names(lpd_df) <- c("Search", "Title", "Abstract", "DOI", "EID", "SE_index", "p", "Relevance")
names(predicts_df) <- c("Search", "Title", "Abstract", "UT", "SE_index", "p", "Relevance")

# Save data to csv
write.csv(lpd_df, "../Results/lpi_man_class.csv")
write.csv(predicts_df, "../Results/predicts_man_class.csv")






