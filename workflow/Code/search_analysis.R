#%%%%%%%%%
# R script to analyse data searches and manual classification for LPD and PREDICTS
#%%%%%%%%%

# Clear env
rm(list = ls())
graphics.off()


# Load libraries
require(ggplot2)
require(readxl)
require(plyr)
require(reshape)
require(grid)
require(gridExtra)
require(gtable)
require(cowplot)
require(RColorBrewer)
require(lme4)
library(Cairo)


#%%%%
# Functions
#%%%%

# Function to return counts covering both std and ranked classifications
obtain_counts <- function(input_df){
    # First subset to unranked assessments
    sub1 <- subset(input_df, !is.na(input_df$Relevance_std_bin))
    # Find total number assessed by time
    n_read1 <- c(sum(sub1$Timespan_std == 5, na.rm = T), 
                 sum(sub1$Timespan_std %in% c(5, 10), na.rm = T), 
                 sum(sub1$Timespan_std %in% c(5, 10, 15), na.rm = T))
    # Find relevant papers by time
    n_rel1 <- c(sum(sub1$Relevance_std_bin[which(sub1$Timespan_std == 5)] == 1), 
                sum(sub1$Relevance_std_bin[which(sub1$Timespan_std %in% c(5, 10))] == 1), 
                sum(sub1$Relevance_std_bin[which(sub1$Timespan_std %in% c(5, 10, 15))] == 1))
    
    # Repeat for ranked
    sub2 <- subset(input_df, !is.na(input_df$Relevance_ranked_bin))
    # Find total number assessed by time
    n_read2 <- c(sum(sub2$Timespan_ranked == 5, na.rm = T), 
                 sum(sub2$Timespan_ranked %in% c(5, 10), na.rm = T), 
                 sum(sub2$Timespan_ranked %in% c(5, 10, 15), na.rm = T))
    # Find relevant papers by time
    n_rel2 <- c(sum(sub2$Relevance_ranked_bin[which(sub2$Timespan_ranked == 5)] == 1), 
                sum(sub2$Relevance_ranked_bin[which(sub2$Timespan_ranked %in% c(5, 10))] == 1), 
                sum(sub2$Relevance_ranked_bin[which(sub2$Timespan_ranked %in% c(5, 10, 15))] == 1))
    
    out_df <- data.frame('Search' = input_df$Search[1],
                         'Timespan' = c(5, 10, 15), 
                         'Std_N' = n_read1, 
                         'Std_rel' = n_rel1, 
                         'Std_ratio' = n_rel1/n_read1,
                         'Ranked_N' = n_read2,
                         'Ranked_rel' = n_rel2,
                         'Ranked_ratio' = n_rel2/n_read2,
                         'N_search_results' = dim(input_df)[1])
    
    return(out_df)
}


# Function to obtain ggplot legend
g_legend <- function(a.gplot){ 
    tmp <- ggplot_gtable(ggplot_build(a.gplot)) 
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box") 
    legend <- tmp$grobs[[leg]] 
    return(legend)} 


# Function to load files and process data for analysis
load_process <- function(file_list, database){
    # Create dfs to store output
    data_df <- data.frame()
    corr_df <- data.frame(matrix(ncol = 3, nrow = length(file_list)))
    colnames(corr_df) <- c('Search', 'Rho', 'p_value')
    
    i <- 1
    
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
        tmp_df$Relevance_std_bin <- tmp_df$Relevance_ranked_bin <- NA
        
        tmp_df$Relevance_std_bin[!is.na(tmp_df$Timespan_std)] <- 0
        tmp_df$Relevance_std_bin[grep("1", tmp_df$Relevance_std)] <- 1
        
        tmp_df$Relevance_ranked_bin[!is.na(tmp_df$Timespan_ranked)] <- 0
        tmp_df$Relevance_ranked_bin[grep("1", tmp_df$Relevance_ranked)] <- 1
        
        # Add Model ranks and cumulative sums of papers found using the different techniques
        tmp_df <- tmp_df[order(-tmp_df$p),]
        tmp_df$Model_rank <- seq(1, dim(tmp_df)[1])
        tmp_df$Model_cumsum <- cumsum(tmp_df$Relevance_ranked_bin)
            
        tmp_df <- tmp_df[order(tmp_df$Search_rank),]
        tmp_df$Search_cumsum <- cumsum(tmp_df$Relevance_std_bin)
        
        # Rbind 
        data_df <- rbind(data_df, tmp_df)
        
        spearman_ <- cor.test(tmp_df$Search_rank, 
                              tmp_df$Model_rank,
                              method = 'spearman')
        corr_df$Search[i] <- tmp_df$Search[1]
        corr_df$Rho[i] <- as.numeric(spearman_$estimate)
        corr_df$p_value[i] <- as.numeric(spearman_$p.value)
        
        i <- i + 1
        
    }
    
    # Generate df of counts
    count_df <- ddply(data_df, .(Search), obtain_counts)
    
    if (database == "lpd"){
        # Neaten df naming
        # Create Class and Realm cols
        count_df$Class <- sapply(strsplit(as.character(count_df$Search), '_'), "[", 1)
        count_df$Realm <- sapply(strsplit(as.character(count_df$Search), '_'), "[", 2)
    
        # Fix lower case letters
        substr(count_df$Class, 1, 1) <- toupper(substr(count_df$Class, 1, 1))
        substr(count_df$Realm, 1, 1) <- toupper(substr(count_df$Realm, 1, 1))
    
        count_df$Class[which(count_df$Class == 'Amphib')] <- 'Amphibian'
        
        # Melt data for analysis
        melt_count_df <- melt(count_df, id.vars = c("Search", "Timespan", "Class", "Realm"), 
                        measure.vars = c("Std_rel", "Ranked_rel"))
        colnames(melt_count_df) <- c("Search", "Timespan", "Class", "Realm", "Model", "Yes")
        
        melt_N_df <- melt(count_df, id.vars = c("Search", "Timespan", "Class", "Realm"), 
                          measure.vars = c("Std_N", "Ranked_N"))
        colnames(melt_N_df) <- c("Search", "Timespan", "Class", "Realm", "Model", "N")
    }
    else if (database == "predicts"){
        # Need to fix biome labels...
        count_df$Biome <- c(rep("Desert", 3), rep("Flooded grassland", 3),
                            rep("Mangrove", 3), rep("Mediterranean", 3),
                            rep("Montane grassland", 3), rep("Boreal forest", 3), 
                            rep("Temperate broadleaf forest", 3), 
                            rep("Temperate coniferous forest", 3),
                            rep("Temperate grassland", 3),
                            rep("Tropical coniferous forest", 3),
                            rep("Topical grassland", 3), rep("Tundra", 3))
        # Melt data for analysis
        melt_count_df <- melt(count_df, id.vars = c("Search", "Timespan", "Biome"), 
                        measure.vars = c("Std_rel", "Ranked_rel"))
        colnames(melt_count_df) <- c("Search", "Timespan", "Biome", "Model", "Yes")
        
        melt_N_df <- melt(count_df, id.vars = c("Search", "Timespan", "Biome"), 
                              measure.vars = c("Std_N", "Ranked_N"))
        colnames(melt_N_df) <- c("Search", "Timespan", "Biome", "Model", "N")
    }
    # Combine count and n dfs
    melt_count_df$Model <- as.character(melt_count_df$Model)
    melt_N_df$Model <- as.character(melt_N_df$Model)
    
    melt_count_df$Model <- sapply(strsplit(melt_count_df$Model, "_"), "[", 1)
    melt_N_df$Model <- sapply(strsplit(melt_N_df$Model, "_"), "[", 1)
    
    melt_df <- merge(melt_count_df, melt_N_df)
    melt_df$No <- melt_df$N - melt_df$Yes
    melt_df$Model <- factor(melt_df$Model, levels = c("Std", "Ranked"))
    
    return(list("data" = data_df, "corr" = corr_df, "counts" = count_df, 
                "melt" = melt_df))
}


fit_glmms <- function(input_data){
    glmms <- list()
    
    for (t in c(5, 10, 15)){
        tmp_glmm <- glmer(data = input_data[input_data$Timespan == t,],
                         formula = cbind(Yes, No) ~ Model + (1|Search),
                         family = "binomial")
        if (t == 5){
            glmms$five <- tmp_glmm
        }
        if (t == 10){
            glmms$ten <- tmp_glmm
        }
        if (t == 15){
            glmms$fifteen <- tmp_glmm
        }
    }
    return(glmms)
}


#%%%%
# Constants
#%%%%

lpd_input_dir <- "../Results/Scopus_lpi/"
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

# Analyse
lpd_binom_glmms <- fit_glmms(input_data = lpd_data$melt)
predicts_binom_glmms <- fit_glmms(input_data = predicts_data$melt)

lapply(lpd_binom_glmms, FUN = function(x){summary(x)})
lapply(predicts_binom_glmms, FUN = function(x){summary(x)})


# Prop relevant ratio - True
hist(lpd_data$counts$Ranked_ratio/lpd_data$counts$Std_ratio)
range(lpd_data$counts$Ranked_ratio/lpd_data$counts$Std_ratio)

hist(predicts_data$counts$Ranked_ratio/predicts_data$counts$Std_ratio)
range(predicts_data$counts$Ranked_ratio/predicts_data$counts$Std_ratio, na.rm = T)


conc_fac_calc <- function(x){
    cc <- coef(summary(x))
    std_prop <- 1/(1+exp(-cc[1,1]))
    rnk_prop <- 1/(1+exp(-(cc[1,1]+cc[2,1])))
    conc_fac <- rnk_prop/std_prop
    
    return(c("std_prop" = std_prop, "rnk_prop" = rnk_prop, "conc_fac" = conc_fac))
}

# Model inferred
lapply(lpd_binom_glmms, FUN = conc_fac_calc)

lapply(predicts_binom_glmms, FUN = conc_fac_calc)

   
#%%%%%
# Ratio plots
#%%%%%


# Get z score/sig from glmm - add to plot...
beta_z_sig_p_extract <- function(glmm_mod){
    beta <- round(summary(glmm_mod)$coefficients[2,1], 3)
    z <- round(summary(glmm_mod)$coefficients[2,3], 3)
    p <- summary(glmm_mod)$coefficients[2,4]
    sig <- ""
    if (p<0.1){
        sig <- "†"
    }
    if (p<0.05){
        sig <- "*"
    }
    if (p<0.01){
        sig <- "**"
    }
    if (p<0.001){
        sig <- "***"
    }
    return(list(beta = beta, z = z, p = p, sig = sig))
}


lpd_rat_plts <- list()
# labels_ <- c('a.', 'b.', 'c.')
# i <- 1
for (t in c(5, 10, 15)){
    tmp_data <- lpd_data$counts[lpd_data$counts$Timespan == t,]
    
    if (t==5){
        m_summ <- beta_z_sig_p_extract(lpd_binom_glmms$five)
        tmp_title <- "Five min."
    }
    if (t==10){
        m_summ <- beta_z_sig_p_extract(lpd_binom_glmms$ten)
        tmp_title <- "Ten min."
    }
    if (t==15){
        m_summ <- beta_z_sig_p_extract(lpd_binom_glmms$fifteen)
        tmp_title <- "Fifteen min."
    }
    
    
    lpd_rat_plts[[paste(t, '_min', sep = '')]] <- ggplot(data = tmp_data) +
        geom_point(aes(x = Std_ratio, 
                       y = Ranked_ratio, 
                       color = Class,
                       fill = Class,
                       shape = Realm),
                   size = 3.5) +
        geom_segment(aes_(x = 0, xend = 1, 
                          y = 0, yend = 1), 
                     colour = 'gray', lty = 2) +
        annotate("text", x = 0.625, y = 0.06, 
                 label = as.expression(bquote(bolditalic(z)~": "~.(sprintf("%.3f", m_summ$z))~.(m_summ$sig))),
                 hjust = 0,
                 size = 6.5) +
        xlab('Search engine hit rate') +
        ylab('Classifier hit rate') +
        ggtitle(tmp_title) +
        scale_x_continuous(breaks = seq(0, 1, by = 0.25),
                           labels = seq(0, 1, by = 0.25),
                           limits = c(0, 1)) +
        scale_y_continuous(breaks = seq(0, 1, by = 0.25),
                           labels = seq(0, 1, by = 0.25),
                           limits = c(0, 1)) +
        scale_shape_manual(name = 'Realm',
                           values = c(21, 22, 23, 24, 25, 4)) + 
        theme_bw() +
        theme(axis.text = element_text(size = 16),
              axis.title = element_text(size = 20),
              legend.text = element_text(size = 16),
              legend.title = element_text(size = 18),
              plot.title = element_text(size = 18, face = "bold")) +
        coord_equal()
    
    i <- i+1
}

# lpd_rat_plts$`5_min`
# lpd_rat_plts$`10_min`
# lpd_rat_plts$`15_min`

# Get legend
lpd_rat_legend <- g_legend(lpd_rat_plts[[1]])


predicts_rat_plts <- list()
# i <- 1
for (t in c(5, 10, 15)){
    tmp_data <- predicts_data$counts[predicts_data$counts$Timespan == t,]
    
    if (t==5){
        m_summ <- beta_z_sig_p_extract(predicts_binom_glmms$five)
        tmp_title <- "Five min."
    }
    if (t==10){
        m_summ <- beta_z_sig_p_extract(predicts_binom_glmms$ten)
        tmp_title <- "Ten min."
    }
    if (t==15){
        m_summ <- beta_z_sig_p_extract(predicts_binom_glmms$fifteen)
        tmp_title <- "Fifteen min."
    }
    predicts_rat_plts[[paste(t, '_min', sep = '')]] <- ggplot(data = tmp_data) +
        geom_point(aes(x = Std_ratio, 
                       y = Ranked_ratio, 
                       color = Biome,
                       fill = Biome),
                   size = 3.5) +
        geom_segment(aes_(x = 0, xend = 1, 
                          y = 0, yend = 1), 
                     colour = 'gray', lty = 2) +
        xlab('Search engine hit rate') +
        ylab('Classifier hit rate') +
        annotate("text", x = 0.625, y = 0.06, 
                 label = as.expression(bquote(bolditalic(z)~": "~.(sprintf("%.3f", m_summ$z))~.(m_summ$sig))),
                 hjust = 0,
                 size = 6.5) +
        scale_x_continuous(breaks = seq(0, 1, by = 0.25),
                           labels = seq(0, 1, by = 0.25),
                           limits = c(0, 1)) +
        scale_y_continuous(breaks = seq(0, 1, by = 0.25),
                           labels = seq(0, 1, by = 0.25),
                           limits = c(0, 1)) +
        theme_bw() +
        theme(axis.text = element_text(size = 16),
              axis.title = element_text(size = 20),
              legend.text = element_text(size = 16),
              legend.title = element_text(size = 18),
              plot.title = element_text(size = 18, face = "bold")) +
        coord_equal()
    
    i <- i+1
}

# predicts_rat_plts$`5_min`
# predicts_rat_plts$`10_min`
# predicts_rat_plts$`15_min`


# Get legend
predicts_rat_legend <- g_legend(predicts_rat_plts[[1]])


all_rat_plts <- plot_grid(lpd_rat_plts[[1]] + theme(axis.title.x = element_blank(),
                                                    axis.text.x = element_blank(),
                                            legend.position = 'none'),
                          lpd_rat_plts[[2]] + theme(axis.title = element_blank(), 
                                                    axis.text = element_blank(),
                                            legend.position = 'none'),
                          lpd_rat_plts[[3]] + theme(axis.title = element_blank(),
                                            axis.text = element_blank(),
                                            legend.position = 'none'),
                      
                      predicts_rat_plts[[1]] + theme(axis.title.x = element_blank(),
                                                 legend.position = 'none'),
                      predicts_rat_plts[[2]] + theme(axis.title.y = element_blank(), 
                                                     axis.text.y = element_blank(),
                                                 legend.position = 'none'),
                      predicts_rat_plts[[3]] + theme(axis.title = element_blank(),
                                                 axis.text.y = element_blank(),
                                                 legend.position = 'none'),
                      
                      labels = c("LPD", "", "", "PREDICTS", "", ""),
                      label_size = 20,
                      hjust = c(0,0,0,0,0,0), vjust = 1.5,
                      axis = "b", align = "hv", nrow = 2)

plot(all_rat_plts)

all_rat_legends <- plot_grid(lpd_rat_legend,
                         predicts_rat_legend,
                         align = "v", nrow = 2)
plot(all_rat_legends)

full_rat_plt <- grid.arrange(plot_grid(all_rat_plts, all_rat_legends, 
                                       rel_widths = c(1, 0.3)))

# ggsave(plot = full_rat_plt, filename = 'hit_rate.pdf',
#        path = '../Results/Figs',
#        width = 20, height = 12, dpi = 300, device = "pdf")

# ggsave(plot = full_rat_plt, filename = 'hit_rate.pdf',
#        path = '../../Results/Figs',
#        width = 20, height = 12, dpi = 300, device = cairo_pdf)

cmb_rat_plts <- list()

for (t in c(5, 10, 15)){
    tmp_data_predicts <- predicts_data$counts[predicts_data$counts$Timespan == t,]
    tmp_data_lpd <- lpd_data$counts[lpd_data$counts$Timespan == t,]
    
    tmp_data_predicts$Dataset <- "PREDICTS"
    tmp_data_lpd$Dataset <- "LPD"
    
    col_nms <- c("Search", "Timespan", "Std_N", "Std_rel", "Std_ratio", 
                 "Ranked_N", "Ranked_rel", "Ranked_ratio", "N_search_results", 
                 "Dataset")
    tmp_data <- rbind(tmp_data_predicts[,col_nms],
                      tmp_data_lpd[,col_nms])
    
    if (t==5){
        p_m_summ <- beta_z_sig_p_extract(predicts_binom_glmms$five)
        l_m_summ <- beta_z_sig_p_extract(lpd_binom_glmms$five)
        tmp_title <- "Five minutes"
    }
    if (t==10){
        p_m_summ <- beta_z_sig_p_extract(predicts_binom_glmms$ten)
        l_m_summ <- beta_z_sig_p_extract(lpd_binom_glmms$ten)
        tmp_title <- "Ten minutes"
    }
    if (t==15){
        p_m_summ <- beta_z_sig_p_extract(predicts_binom_glmms$fifteen)
        l_m_summ <- beta_z_sig_p_extract(lpd_binom_glmms$fifteen)
        tmp_title <- "Fifteen minutes"
    }
    cmb_rat_plts[[paste(t, '_min', sep = '')]] <- ggplot(data = tmp_data) +
        geom_point(aes(x = Std_ratio, 
                       y = Ranked_ratio, 
                       color = Dataset),
                   size = 3.5, alpha = 0.7) +
        geom_segment(aes_(x = 0, xend = 1, 
                          y = 0, yend = 1), 
                     colour = 'gray', lty = 2) +
        xlab('Search engine hit rate') +
        ylab('Classifier hit rate') +
        ggtitle(tmp_title) +
        annotate("text", x = 0.625, y = 0.15, 
                 hjust = 0, size = 6.5,
                 label = as.expression(bquote(bolditalic(beta)~": "))) +
        annotate("text", x = 0.725, y = 0.15,
                 hjust = 0, size = 6.5,
                 colour = "#E69F00",
                 label = paste(sprintf("%.3f", l_m_summ$beta), l_m_summ$sig)) +
        annotate("text", x = 0.725, y = 0.08,
                 hjust = 0, size = 6.5,
                 colour = "#009E73",
                 label = paste(sprintf("%.3f", p_m_summ$beta), p_m_summ$sig)) +
        scale_color_manual(name = 'Indicator dataset',
                           values = c("#E69F00", "#009E73"),
                           breaks = c('LPD', 'PREDICTS')) +
        scale_x_continuous(breaks = seq(0, 1, by = 0.25),
                           labels = seq(0, 1, by = 0.25),
                           limits = c(0, 1)) +
        scale_y_continuous(breaks = seq(0, 1, by = 0.25),
                           labels = seq(0, 1, by = 0.25),
                           limits = c(0, 1)) +
        theme_bw() +
        theme(axis.text = element_text(size = 16),
              axis.title = element_text(size = 20),
              legend.text = element_text(size = 16),
              legend.title = element_text(size = 18),
              plot.title = element_text(size = 18, face = "bold"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank()) +
        coord_equal()
    
    # i <- i+1
}

cmb_rat_plts

cmb_rat_legend <- g_legend(cmb_rat_plts[[1]])


cmb_rat_grd <- plot_grid(cmb_rat_plts[[1]] + theme(axis.title.x = element_blank(),
                                                         legend.position = 'none'),
                         cmb_rat_plts[[2]] + theme(axis.title.y = element_blank(), 
                                                         axis.text.y = element_blank(),
                                                         legend.position = 'none'),
                         cmb_rat_plts[[3]] + theme(axis.title = element_blank(),
                                                         axis.text.y = element_blank(),
                                                         legend.position = 'none'),
                          axis = "b", align = "hv", nrow = 1)

plot(cmb_rat_grd)


cmb_rat_plt <- plot_grid(cmb_rat_grd, cmb_rat_legend, 
                         rel_widths = c(1, 0.12))

ggsave(plot = cmb_rat_plt, filename = 'cmb_hit_rate.pdf',
       path = '../Results/Figs',
       width = 20, height = 6, dpi = 300, device = cairo_pdf)


#%%%%%
# Alternative plotting using effects size/beta coefficients
#%%%%%

create_effect_dataframe <- function(lpd_glms, predicts_glms){
    # Create df to store data 
    mod_eff_df <- data.frame(matrix(ncol = 7, nrow = 6))
    colnames(mod_eff_df) <- c("Dataset", "Timespan", "Beta", "Std_error", "loCI", "upCI", "p")
    # colnames(mod_eff_df) = 
    
    r_count <- 1
    for (d in c("LPD", "PREDICTS")){
        for (t in c("five", "ten", "fifteen")){
            if (d == "LPD"){
                mods <- lpd_glms
            }
            else if (d == "PREDICTS"){
                mods <- predicts_glms
            }
            m <- mods[[t]] #$add
            mod_eff_df[r_count,] <- c(d, t, # dataset, timespan
                                      summary(m)$coefficients[2,1], # estimate
                                      summary(m)$coefficients[2,2], # std error
                                      # confint(m, "ModelRanked")[1], # loCI
                                      # confint(m, "ModelRanked")[2], # upCI
                                      confint.merMod(m, "ModelRanked", method = "Wald")[1], # loCI
                                      confint.merMod(m, "ModelRanked", method = "Wald")[2], # upCI
                                      summary(m)$coefficients[2,4]) # p value
            r_count <- r_count + 1
        }
    }
    # Fromat cols
    mod_eff_df$Dataset <- factor(mod_eff_df$Dataset, levels = c("LPD", "PREDICTS"))
    mod_eff_df$Timespan <- factor(mod_eff_df$Timespan, levels = c("five", "ten", "fifteen"))
    mod_eff_df$Beta <- as.numeric(mod_eff_df$Beta)
    mod_eff_df$loCI <- as.numeric(mod_eff_df$loCI)
    mod_eff_df$upCI <- as.numeric(mod_eff_df$upCI)
    mod_eff_df$p <- as.numeric(mod_eff_df$p)
    
    return(mod_eff_df)
}


eff_df <- create_effect_dataframe(lpd_glms = lpd_binom_glmms, 
                                  predicts_glms = predicts_binom_glmms)


# eff_plt <- ggplot(data = eff_df) +
#     geom_hline(yintercept = 0, lty = 'dashed', colour = "black") +
#     geom_point(aes(x = Timespan, y = Beta, colour = Dataset), 
#                position = position_dodge(width = 0.5),
#                size = 3) +
#     geom_errorbar(aes(x = Timespan, ymin = loCI, ymax = upCI, colour = Dataset,
#                       width = 0.125), position = position_dodge(width = 0.5)) +
#     # geom_vline(xintercept = 1.5, lty = 'dashed', colour = 'grey50') +
#     # geom_vline(xintercept = 2.5, lty = 'dashed', colour = 'grey50') +
#     # annotate(geom = 'text', 
#     #          y = 1.125, 
#     #          x = c(0.825, 1.825, 2.825, 3.075), 
#     #          label = c("***", "***", "**", "*"), 
#     #          size = 8) +
#     ylab(expression(paste(beta, ' coefficient estimate'))) +
#     xlab('Timespan (minutes)') +
#     scale_color_manual(name = 'Indicator dataset',
#                        values = c(brewer.pal(9, 'Oranges')[5], brewer.pal(9, 'Blues')[7]),
#                        breaks = c('LPD', 'PREDICTS')) +
#     scale_x_discrete(labels = c(5, 10, 15),
#                        breaks = c("five", "ten", "fifteen")) +
#     scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1),
#                        limits = c(-0.175, 1.2)) +
#     theme_bw() +
#     theme(axis.text = element_text(size = 16),
#           axis.title = element_text(size = 20),
#           legend.text = element_text(size = 16),
#           legend.title = element_text(size = 18),
#           plot.title = element_text(size = 18, face = "bold")) +
#     coord_flip()
# plot(eff_plt)


#%%%%%%%%%%
# Checking for impact of mis-classified papers on model effect
#%%%%%%%%%%

# Identify LPD and PREDICTS papers to change
predicts_disagree <- c("WOS:000425120600003", "WOS:000258635600009", "WOS:000381233500012")
lpd_disagree_class <- c("2-s2.0-53549126882", "2-s2.0-84969822689", "2-s2.0-77952520908",
                        "2-s2.0-0033607767", "2-s2.0-84856941914")
lpd_disagree_note <- c("2-s2.0-84987800304", "2-s2.0-53549126882", "2-s2.0-84969822689", 
                       "2-s2.0-77952520908", "2-s2.0-0033607767", "2-s2.0-84856941914",
                       "2-s2.0-46849098963")

# Change paper class (new col)
predicts_data$data$Relevance_std_bin_switched <- predicts_data$data$Relevance_std_bin
predicts_data$data$Relevance_ranked_bin_switched <- predicts_data$data$Relevance_ranked_bin

predicts_data$data$Relevance_std_bin_switched[predicts_data$data$UT %in% predicts_disagree] <- as.integer(predicts_data$data$Relevance_std_bin[predicts_data$data$UT %in% predicts_disagree] != 1)
predicts_data$data$Relevance_ranked_bin_switched[predicts_data$data$UT %in% predicts_disagree] <- as.integer(predicts_data$data$Relevance_ranked_bin[predicts_data$data$UT %in% predicts_disagree] != 1)

lpd_data$data$Relevance_std_bin_switched_class <- lpd_data$data$Relevance_std_bin_switched_note <- lpd_data$data$Relevance_std_bin
lpd_data$data$Relevance_ranked_bin_switched_class <- lpd_data$data$Relevance_ranked_bin_switched_note <- lpd_data$data$Relevance_ranked_bin

lpd_data$data$Relevance_std_bin_switched_class[lpd_data$data$EID %in% lpd_disagree_class] <- as.integer(lpd_data$data$Relevance_std_bin[lpd_data$data$EID %in% lpd_disagree_class] != 1)
lpd_data$data$Relevance_ranked_bin_switched_class[lpd_data$data$EID %in% lpd_disagree_class] <- as.integer(lpd_data$data$Relevance_ranked_bin[lpd_data$data$EID %in% lpd_disagree_class] != 1)

lpd_data$data$Relevance_std_bin_switched_note[lpd_data$data$EID %in% lpd_disagree_note] <- as.integer(lpd_data$data$Relevance_std_bin[lpd_data$data$EID %in% lpd_disagree_note] != 1)
lpd_data$data$Relevance_ranked_bin_switched_note[lpd_data$data$EID %in% lpd_disagree_note] <- as.integer(lpd_data$data$Relevance_ranked_bin[lpd_data$data$EID %in% lpd_disagree_note] != 1)


# Function to return counts covering both std and ranked classifications
obtain_counts_switched <- function(input_df){
    if ("UT" %in% colnames(input_df)){
        # First subset to unranked assessments
        sub1 <- subset(input_df, !is.na(input_df$Relevance_std_bin_switched))
        # Find total number assessed by time
        n_read1 <- c(sum(sub1$Timespan_std == 5, na.rm = T), 
                     sum(sub1$Timespan_std %in% c(5, 10), na.rm = T), 
                     sum(sub1$Timespan_std %in% c(5, 10, 15), na.rm = T))
        # Find relevant papers by time
        n_rel1 <- c(sum(sub1$Relevance_std_bin_switched[which(sub1$Timespan_std == 5)] == 1), 
                    sum(sub1$Relevance_std_bin_switched[which(sub1$Timespan_std %in% c(5, 10))] == 1), 
                    sum(sub1$Relevance_std_bin_switched[which(sub1$Timespan_std %in% c(5, 10, 15))] == 1))
        
        # Repeat for ranked
        sub2 <- subset(input_df, !is.na(input_df$Relevance_ranked_bin_switched))
        # Find total number assessed by time
        n_read2 <- c(sum(sub2$Timespan_ranked == 5, na.rm = T), 
                     sum(sub2$Timespan_ranked %in% c(5, 10), na.rm = T), 
                     sum(sub2$Timespan_ranked %in% c(5, 10, 15), na.rm = T))
        # Find relevant papers by time
        n_rel2 <- c(sum(sub2$Relevance_ranked_bin_switched[which(sub2$Timespan_ranked == 5)] == 1), 
                    sum(sub2$Relevance_ranked_bin_switched[which(sub2$Timespan_ranked %in% c(5, 10))] == 1), 
                    sum(sub2$Relevance_ranked_bin_switched[which(sub2$Timespan_ranked %in% c(5, 10, 15))] == 1))
 
        out_df <- data.frame('Search' = input_df$Search[1],
                             'Timespan' = c(5, 10, 15), 
                             'Std_N' = n_read1, 
                             'Std_rel' = n_rel1,
                             'Ranked_N' = n_read2, 
                             'Ranked_rel' = n_rel2,
                             'N_search_results' = dim(input_df)[1])
        return(out_df)
    }
    else if ("EID" %in% colnames(input_df)){
        # First subset to unranked assessments
        sub1 <- subset(input_df, !is.na(input_df$Relevance_std_bin_switched_class))
        # Find total number assessed by time
        n_read1 <- c(sum(sub1$Timespan_std == 5, na.rm = T), 
                     sum(sub1$Timespan_std %in% c(5, 10), na.rm = T), 
                     sum(sub1$Timespan_std %in% c(5, 10, 15), na.rm = T))
        # Find relevant papers by time
        n_rel1 <- c(sum(sub1$Relevance_std_bin_switched_class[which(sub1$Timespan_std == 5)] == 1), 
                    sum(sub1$Relevance_std_bin_switched_class[which(sub1$Timespan_std %in% c(5, 10))] == 1), 
                    sum(sub1$Relevance_std_bin_switched_class[which(sub1$Timespan_std %in% c(5, 10, 15))] == 1))
    
        # Repeat for ranked
        sub2 <- subset(input_df, !is.na(input_df$Relevance_ranked_bin_switched_class))
        # Find total number assessed by time
        n_read2 <- c(sum(sub2$Timespan_ranked == 5, na.rm = T), 
                     sum(sub2$Timespan_ranked %in% c(5, 10), na.rm = T), 
                     sum(sub2$Timespan_ranked %in% c(5, 10, 15), na.rm = T))
        # Find relevant papers by time
        n_rel2 <- c(sum(sub2$Relevance_ranked_bin_switched_class[which(sub2$Timespan_ranked == 5)] == 1), 
                    sum(sub2$Relevance_ranked_bin_switched_class[which(sub2$Timespan_ranked %in% c(5, 10))] == 1), 
                    sum(sub2$Relevance_ranked_bin_switched_class[which(sub2$Timespan_ranked %in% c(5, 10, 15))] == 1))
        
        # And for notes based switches
        # First subset to unranked assessments
        sub3 <- subset(input_df, !is.na(input_df$Relevance_std_bin_switched_note))
        # Find total number assessed by time
        n_read3 <- c(sum(sub3$Timespan_std == 5, na.rm = T), 
                     sum(sub3$Timespan_std %in% c(5, 10), na.rm = T), 
                     sum(sub3$Timespan_std %in% c(5, 10, 15), na.rm = T))
        # Find relevant papers by time
        n_rel3 <- c(sum(sub3$Relevance_std_bin_switched_note[which(sub3$Timespan_std == 5)] == 1), 
                    sum(sub3$Relevance_std_bin_switched_note[which(sub3$Timespan_std %in% c(5, 10))] == 1), 
                    sum(sub3$Relevance_std_bin_switched_note[which(sub3$Timespan_std %in% c(5, 10, 15))] == 1))
        
        # Repeat for ranked
        sub4 <- subset(input_df, !is.na(input_df$Relevance_ranked_bin_switched_note))
        # Find total number assessed by time
        n_read4 <- c(sum(sub4$Timespan_ranked == 5, na.rm = T), 
                     sum(sub4$Timespan_ranked %in% c(5, 10), na.rm = T), 
                     sum(sub4$Timespan_ranked %in% c(5, 10, 15), na.rm = T))
        # Find relevant papers by time
        n_rel4 <- c(sum(sub2$Relevance_ranked_bin_switched_note[which(sub4$Timespan_ranked == 5)] == 1), 
                    sum(sub2$Relevance_ranked_bin_switched_note[which(sub4$Timespan_ranked %in% c(5, 10))] == 1), 
                    sum(sub2$Relevance_ranked_bin_switched_note[which(sub4$Timespan_ranked %in% c(5, 10, 15))] == 1))
        
        out_df <- data.frame('Search' = input_df$Search[1],
                             'Timespan' = c(5, 10, 15), 
                             'Std_N_class' = n_read1, 
                             'Std_rel_class' = n_rel1,
                             'Ranked_N_class' = n_read2,
                             'Ranked_rel_class' = n_rel2,
                             'Std_N_note' = n_read3,
                             'Std_rel_note' = n_rel3,
                             'Ranked_N_note' = n_read4,
                             'Ranked_rel_note' = n_rel4,
                             'N_search_results' = dim(input_df)[1])

        return(out_df)
    }
}

# Create count dfs
predicts_switch_count_df <- ddply(predicts_data$data, .(Search), obtain_counts_switched)
lpd_switch_count_df <- ddply(lpd_data$data, .(Search), obtain_counts_switched)

# Melt to give model input dfs
# Re analyse
# Assess
# Need to fix biome labels...
predicts_switch_count_df$Biome <- c(rep("Desert", 3), rep("Flooded grassland", 3),
                  rep("Mangrove", 3), rep("Mediterranean", 3),
                  rep("Montane grassland", 3), rep("Boreal forest", 3), 
                  rep("Temperate broadleaf forest", 3), 
                  rep("Temperate coniferous forest", 3),
                  rep("Temperate grassland", 3),
                  rep("Tropical coniferous forest", 3),
                  rep("Topical grassland", 3), rep("Tundra", 3))

lpd_switch_count_df$Class <- sapply(strsplit(as.character(lpd_switch_count_df$Search), '_'), "[", 1)
lpd_switch_count_df$Realm <- sapply(strsplit(as.character(lpd_switch_count_df$Search), '_'), "[", 2)

# Fix lower case letters
substr(lpd_switch_count_df$Class, 1, 1) <- toupper(substr(lpd_switch_count_df$Class, 1, 1))
substr(lpd_switch_count_df$Realm, 1, 1) <- toupper(substr(lpd_switch_count_df$Realm, 1, 1))

lpd_switch_count_df$Class[which(lpd_switch_count_df$Class == 'Amphib')] <- 'Amphibian'

# Melt data for analysis
predicts_switch_melt_count_df <- melt(predicts_switch_count_df, id.vars = c("Search", "Timespan", "Biome"), 
                      measure.vars = c("Std_rel", "Ranked_rel"))
colnames(predicts_switch_melt_count_df) <- c("Search", "Timespan", "Biome", "Model", "Yes")

predicts_switch_melt_N_df <- melt(predicts_switch_count_df, id.vars = c("Search", "Timespan", "Biome"), 
                  measure.vars = c("Std_N", "Ranked_N"))
colnames(predicts_switch_melt_N_df) <- c("Search", "Timespan", "Biome", "Model", "N")

# Combine count and n dfs
predicts_switch_melt_N_df$Model <- as.character(predicts_switch_melt_N_df$Model)
predicts_switch_melt_count_df$Model <- as.character(predicts_switch_melt_count_df$Model)

predicts_switch_melt_N_df$Model <- sapply(strsplit(predicts_switch_melt_N_df$Model, "_"), "[", 1)
predicts_switch_melt_count_df$Model <- sapply(strsplit(predicts_switch_melt_count_df$Model, "_"), "[", 1)

predicts_switch_melt_df <- merge(predicts_switch_melt_count_df, predicts_switch_melt_N_df)
predicts_switch_melt_df$No <- predicts_switch_melt_df$N - predicts_switch_melt_df$Yes
predicts_switch_melt_df$Model <- factor(predicts_switch_melt_df$Model, levels = c("Std", "Ranked"))

predicts_switch_glmms <- fit_glmms(input_data = predicts_switch_melt_df)


# LPD switched using Stef's notes
lpd_switch_melt_note_count_df <-  melt(lpd_switch_count_df, id.vars = c("Search", "Timespan", "Class", "Realm"), 
                                        measure.vars = c("Std_rel_note", "Ranked_rel_note"))
colnames(lpd_switch_melt_note_count_df) <- c("Search", "Timespan", "Class", "Realm", "Model", "Yes")

lpd_switch_melt_note_N_df <- melt(lpd_switch_count_df, id.vars = c("Search", "Timespan", "Class", "Realm"), 
                                   measure.vars = c("Std_N_note", "Ranked_N_note"))
colnames(lpd_switch_melt_note_N_df) <- c("Search", "Timespan", "Class", "Realm", "Model", "N")

# Combine count and n dfs
lpd_switch_melt_note_count_df$Model <- as.character(lpd_switch_melt_note_count_df$Model)
lpd_switch_melt_note_N_df$Model <- as.character(lpd_switch_melt_note_N_df$Model)

lpd_switch_melt_note_count_df$Model <- sapply(strsplit(lpd_switch_melt_note_count_df$Model, "_"), "[", 1)
lpd_switch_melt_note_N_df$Model <- sapply(strsplit(lpd_switch_melt_note_N_df$Model, "_"), "[", 1)

lpd_switch_melt_note_df <- merge(lpd_switch_melt_note_count_df, lpd_switch_melt_note_N_df)
lpd_switch_melt_note_df$No <- lpd_switch_melt_note_df$N - lpd_switch_melt_note_df$Yes
lpd_switch_melt_note_df$Model <- factor(lpd_switch_melt_note_df$Model, levels = c("Std", "Ranked"))

lpd_switch_note_glmms <- fit_glmms(input_data = lpd_switch_melt_note_df)


#%%%%%
# Results of switches
#%%%%%
# In all models incorporating model and search, effect of model is always larger following switches
# Suggests my classification do not bias results in favour of classifier ranking.

switched_df <- create_effect_dataframe(lpd_glms = lpd_switch_note_glmms,
                                       predicts_glms = predicts_switch_glmms)

eff_df$Classification <- "Original"
switched_df$Classification <- "Expert corrected"

eff_switched_plt_df <- rbind(eff_df, switched_df)
eff_switched_plt_df$Classification <- factor(eff_switched_plt_df$Classification, levels = c("Original", "Expert corrected"))

eff_switched_plt <- ggplot(data = eff_switched_plt_df) +
    geom_hline(yintercept = 0, lty = 'dashed', colour = "black") +
    geom_point(aes(x = Timespan, y = Beta, colour = Dataset, shape = Classification), 
               position = position_dodge(width = 0.3),
               size = 3) +
    geom_errorbar(aes(x = Timespan, ymin = loCI, ymax = upCI, colour = Dataset, group = Dataset:Classification,
                      width = 0), 
                  size = 1,
                  show.legend = F,
                  position = position_dodge(width = 0.3)) +
    ylab(expression(paste(beta, ' coefficient estimate'))) +
    xlab('Timespan (minutes)') +
    scale_color_manual(name = 'Indicator dataset',
                       values = c("#E69F00", "#009E73"),
                       # values = c(brewer.pal(9, 'Oranges')[5], brewer.pal(9, 'Blues')[7]),
                       breaks = c('LPD', 'PREDICTS')) +
    scale_x_discrete(labels = c(5, 10, 15),
                     breaks = c("five", "ten", "fifteen")) +
    scale_y_continuous(breaks = c(0, 0.5, 1, 1.5, 2.0),
                       limits = c(-0.175, 2.1)) +
    theme_bw() +
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 20),
          legend.text = element_text(size = 16),
          legend.title = element_text(size = 18),
          plot.title = element_text(size = 18, face = "bold"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(colour = guide_legend(order = 1),
           shape = guide_legend(order = 0))# +
plot(eff_switched_plt)

# ggsave(plot = eff_switched_plt, filename = 'classifier_eff_switch_plt.pdf',
#        path = '../Results/Figs',
#        width = 10, height = 6, dpi = 300, device = "pdf")

