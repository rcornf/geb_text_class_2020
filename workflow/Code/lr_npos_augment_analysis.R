##%%%%%%%%%%
# Rscript to analyse results of man_class_augment.py and lra_np_testing.py
# i.e. How does augmenting/altering the size of the standard training data influence
# classifier performance
##%%%%%%%%%%

rm(list = ls())
graphics.off()


# Libraries
library(ggplot2)
library(reshape2)
library(cvAUC)
require(gridExtra)
require(gtable)
require(cowplot)

# Functions
# Function to convert vectors of predictions/true to lists for use in cvAUC function.
data_to_lists <- function(df, id, pred_col, true_col){
    split_ls <- split.data.frame(df, df[id])

    pred_ls <- sapply(split_ls,
                      FUN = function(x){
                          pred <- as.vector(x[pred_col])
                      })
    true_ls <- sapply(split_ls,
                      FUN = function(x){
                          true <- x[true_col]
                      })
    fold_ls <- sapply(split_ls,
                      FUN = function(x){
                          fold <- x[id]
                      })
    return(list("pred_ls" = pred_ls, "true_ls" = true_ls, "fold_ls" = fold_ls))
}


# Function to take an n_pos df and calculate cvAUC etc for each n 
npos_df_process <- function(df){
    # drop columns of all na
    df <- df[,colSums(is.na(df))==0]
    # List all cols of predictons
    pred_col_idxs <- grep("p_", names(df))
    pred_col_names <- names(df)[pred_col_idxs]
    
    out_df <- as.data.frame(matrix(NA, nrow = length(pred_col_names), ncol = 5))
    names(out_df) <- c("Npos", "RS", "cvAUC", "loCI", "upCI")

    rw_cnt <- 1
    # For each prediction col, extract data and calculate cvAUC
    for (nam in pred_col_names){
        cicvauc <- ci.cvAUC(predictions = df[,nam], labels = df$y, folds = df$Fold_id)
        npos <- as.numeric(strsplit(nam, "_")[[1]][2])
        if(npos==1){
            npos <- nrow(df)/2*0.9
        }
        rs <- as.numeric(strsplit(nam, "_")[[1]][3])
        out_df[rw_cnt,] <- c(npos,
                             rs,
                             cicvauc$cvAUC,
                             cicvauc$ci[1],
                             cicvauc$ci[2])
        rw_cnt <- rw_cnt+1
    }
    return(out_df)
}

# Function to calculate AUC per search and w
aug_df_process <- function(df){
    # List all cols of predictons
    pred_col_idxs <- grep("p_4_", names(df))
    pred_col_names <- names(df)[pred_col_idxs]
    n_search <- length(unique(df$Search))
    out_df <- as.data.frame(matrix(NA, nrow = length(pred_col_names)*n_search, ncol = 3))
    names(out_df) <- c("Search", "w", "AUC")
    #
    # For each prediction col/w, extract data and calculate cvAUC
    rw_cnt <- 1

    for (nam in pred_col_names){
        for (s in unique(df$Search)){
            tmp_df <- subset(df, Search == s)
            auc <- cvAUC(tmp_df[nam], tmp_df$Relevance)
            idx <- length(strsplit(nam, "_")[[1]])
            w <- as.numeric(strsplit(nam, "_")[[1]][idx])
            out_df[rw_cnt,] <- list(as.character(tmp_df$Search[1]),
                                    as.numeric(w),
                                    as.numeric(auc$cvAUC))
            rw_cnt <- rw_cnt+1
        }
    }
    
    # Also group by w and get cvAUC, using search as fold..
    out_df2 <- as.data.frame(matrix(NA, nrow = 11, ncol = 4))
    names(out_df2) <- c("w", "cvAUC", "loCI", "upCI")
    rw_cnt <- 1
    
    for (col_nm in pred_col_names){
        cicvauc <- ci.cvAUC(df[,col_nm], df$Relevance, folds = as.numeric(df$Search))
        idx <- length(strsplit(col_nm, "_")[[1]])
        w <- as.numeric(strsplit(col_nm, "_")[[1]][idx])
        out_df2[rw_cnt,] <- c(w, cicvauc$cvAUC, cicvauc$ci[1], cicvauc$ci[2])
        rw_cnt <- rw_cnt+1
    }
    return(list("df1" = out_df, "df2" = out_df2))
}

# Compare weighted augmentation to simply using original texts - original p
orig_AUC <- function(aug_df){
    orig_cvAUC <- cvAUC::ci.cvAUC(aug_df$p,
                                  aug_df$Relevance,
                                  folds = as.numeric(aug_df$Search))
    return(data.frame("avAUC" = orig_cvAUC$cvAUC))
}


# Main Code
# Load data
lpi_npos_df <- read.csv("../Results/lpi_npos_rs10.csv")
predicts_npos_df <- read.csv("../Results/predicts_npos_rs10.csv")

lpi_aug_df <- read.csv("../Results/lpi_man_class_augment.csv")
predicts_aug_df <- read.csv("../Results/predicts_man_class_augment.csv")

lpi_aug_cache_df <- read.csv("../Results/lpi_man_class_augment_cache.csv")
predicts_aug_cache_df <- read.csv("../Results/predicts_man_class_augment_cache.csv")


# N pos analysis
# cvAUC for each pred col,
lpi_npos_auc_df <- npos_df_process(lpi_npos_df)
predicts_npos_auc_df <- npos_df_process(predicts_npos_df) 

# plot N pos v avAUC + CIs...
# # plot(lpi_npos_auc_df$Npos, lpi_npos_auc_df$cvAUC)
# # plot(predicts_npos_auc_df$Npos, predicts_npos_auc_df$cvAUC)
# 
# lpi_npos_plt <- ggplot(lpi_npos_auc_df) +
#     geom_point(aes(x = Npos*2, y = cvAUC, colour = Npos, group = RS), 
#                position = position_dodge(50)) +
#     geom_errorbar(aes(x = Npos*2, ymin = loCI, ymax = upCI, width = 0,
#                       colour = Npos, group = RS), 
#                   position = position_dodge(50)) + 
#     xlab(element_blank()) +
#     ylab("Average AUC") +
#     ggtitle("LPD") +
#     scale_y_continuous(limits = c(0.97,1)) + 
#     scale_x_continuous(breaks = seq(250,1000,250)) +
#     theme_bw() +
#     theme(legend.position = "none",
#           plot.title = element_text(size = 26),
#           axis.text = element_text(size = 20), 
#           axis.title = element_text(size = 24), 
#           panel.grid.minor.y=element_blank())  
# 
# predicts_npos_plt <- ggplot(predicts_npos_auc_df) +
#     geom_point(aes(x = Npos*2, y = cvAUC, colour = Npos, group = RS), 
#                position = position_dodge(50)) +
#     geom_errorbar(aes(x = Npos*2, ymin = loCI, ymax = upCI, width = 0,
#                       colour = Npos, group = RS), 
#                   position = position_dodge(50)) + 
#     xlab(element_blank()) +
#     ggtitle("PREDICTS") +
#     ylab(element_blank()) +
#     scale_y_continuous(limits = c(0.97,1)) + 
#     scale_x_continuous(breaks = seq(250,1000,250)) +
#     theme_bw() +
#     theme(legend.position = "none",
#           plot.title = element_text(size = 26),
#           axis.text = element_text(size = 20), 
#           axis.title = element_text(size = 24), 
#           panel.grid.minor.y=element_blank())  
# 
# 
# npos_plt <- plot_grid(
#     plot_grid(lpi_npos_plt,
#               predicts_npos_plt + theme(axis.text.y = element_blank()),
#               align = "hv", nrow = 1),
#     ggdraw() + 
#         draw_label("N training texts",
#                    size = 24),
#     align="v", rel_heights = c(1,0.1), nrow = 2
# )
# 
# # ggsave(plot = npos_plt, filename = "npos_plt.pdf",
# #        path = "../Results/Figs/",
# #        width = 15, height = 10, dpi = 300, device = "pdf")


# TODO: combine lpd and predicts npos dfs
# Summarise range of avAUC
# Plot
lpi_npos_auc_df$Dataset <- "LPD"
predicts_npos_auc_df$Dataset <- "PREDICTS"

npos_df <- rbind(lpi_npos_auc_df, 
                 predicts_npos_auc_df)


npos_summ_df <- ddply(npos_df, .(Dataset, Npos), 
                      function(x){
                          data.frame("min_cvAUC" = min(x$cvAUC),
                                     "max_cvAUC" = max(x$cvAUC),
                                     "min_loCI" = min(x$loCI),
                                     "max_upCI" = max(x$upCI))
                      })

npos_plt1 <- ggplot(npos_summ_df) +
    geom_errorbar(aes(x = Npos*2, ymin = min_cvAUC, ymax = max_cvAUC, width = 0,
                      colour = Dataset), 
                  size = 3,
                  position = position_dodge(30)) + 
    geom_errorbar(aes(x = Npos*2, ymin = min_loCI, ymax = max_upCI, width = 0,
                      colour = Dataset), 
                  size = 1.25,
                  position = position_dodge(30)) + 
    geom_point(aes(x = Npos*2, y = min_cvAUC, colour = Dataset),
               size = 2.5,
               position = position_dodge(30),
               data = subset(npos_summ_df, min_cvAUC == max_cvAUC)) +
    scale_color_manual(name = 'Indicator dataset',
                       values = c("#E69F00", "#009E73"),
                       breaks = c('LPD', 'PREDICTS')) +#,
    xlab("N training texts") +
    ylab("AUC") +
    scale_y_continuous(limits = c(0.95,1)) +
    scale_x_continuous(breaks = seq(250,1000,250)) +
    theme_bw() +
    theme(
          axis.text = element_text(size = 16), 
          axis.title = element_text(size = 20), 
          legend.text = element_text(size = 16), #  20 
          legend.title = element_text(size = 18),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())  


# Augmentation analysis
# Caclulate AUC per search and w and average AUC over all searches for a given w
lpi_aug_auc <- aug_df_process(lpi_aug_df)
predicts_aug_auc <- aug_df_process(predicts_aug_df)

# Also, merge AUC data with weight per paper data...
lpi_aug_auc$df1 <- merge(lpi_aug_auc$df1, 
                          lpi_aug_cache_df[lpi_aug_cache_df$method==4,],
                          by = c("Search", "w"),
                          all.x = T)

predicts_aug_auc$df1 <- merge(predicts_aug_auc$df1, 
                               predicts_aug_cache_df[predicts_aug_cache_df$method==4,],
                               by = c("Search", "w"),
                               all.x = T)


lpi_orig_auc_df <- orig_AUC(lpi_aug_df)
predicts_orig_auc_df <- orig_AUC(predicts_aug_df)


# lpi_w_plt <- ggplot() + 
#     geom_hline(aes(yintercept = avAUC), 
#                data = lpi_orig_auc_df,
#                lty = 2, size = 1, colour = "darkgrey") +
#     geom_point(aes(x = w, y = cvAUC),
#                size = 3,
#                data = lpi_aug_auc$df2) +
#     geom_errorbar(aes(x = w, ymin = loCI, ymax = upCI, width = 0),
#                   size = 1,
#                   data = lpi_aug_auc$df2) +
#     xlab("w") +
#     ylab("Average AUC") +
#     ggtitle("LPD") +
#     scale_y_continuous(limits = c(0.5,1)) +
#     scale_x_continuous(breaks = seq(0,1,0.2)) +
#     theme_bw() +
#     theme(legend.position = "none",
#           plot.title = element_text(size = 26),
#           axis.text = element_text(size = 20), 
#           axis.title = element_text(size = 24), 
#           panel.grid.minor.y=element_blank())
# 
# predicts_w_plt <- ggplot() + 
#     geom_hline(aes(yintercept = avAUC), 
#                data = predicts_orig_auc_df,
#                lty = 2, size = 1, colour = "darkgrey") +
#     geom_point(aes(x = w, y = cvAUC),
#                size = 3,
#                data = predicts_aug_auc$df2) +
#     geom_errorbar(aes(x = w, ymin = loCI, ymax = upCI, width = 0),
#                   size = 1,
#                   data = predicts_aug_auc$df2) +
#     xlab("w") +
#     ylab("Average AUC") +
#     ggtitle("PREDICTS") +
#     scale_y_continuous(limits = c(0.5,1)) +
#     scale_x_continuous(breaks = seq(0,1,0.2)) +
#     theme_bw() +
#     theme(legend.position = "none",
#           plot.title = element_text(size = 26),
#           axis.text = element_text(size = 20), 
#           axis.title = element_text(size = 24), 
#           panel.grid.minor.y=element_blank())
# 
# # This is no longer used in report!!!!
# w_plt <- plot_grid(plot_grid(lpi_w_plt + theme(axis.title.x = element_blank()),
#                              predicts_w_plt + theme(axis.title = element_blank(),
#                                                     axis.text.y = element_blank()),
#                              axis = "b", align = "hv", nrow = 1),
#                    ggdraw() + 
#                        draw_label("w", size = 24),
#                    align="v", rel_heights = c(1,0.1), nrow = 2)

# ggsave(plot = w_plt, filename = "w_plt.pdf",
#        path = "../Results/Figs/",
#        width = 15, height = 10, dpi = 300, device = "pdf")


# Plot Auc v w in a single pane
lpi_aug_auc$df2$Dataset <- "LPD"
predicts_aug_auc$df2$Dataset <- "PREDICTS"

w_df <- rbind(lpi_aug_auc$df2,
              predicts_aug_auc$df2)

w_plt1 <- ggplot() + 
    geom_hline(aes(yintercept = avAUC),
               data = predicts_orig_auc_df,
               lty = 2, size = 1, colour = "#009E73") +
    geom_hline(aes(yintercept = avAUC),
               data = lpi_orig_auc_df,
               lty = 2, size = 1, colour = "#E69F00") +
    geom_errorbar(aes(x = w, ymin = loCI, ymax = upCI, width = 0, colour = Dataset),
                  size = 1, position = position_dodge(0.025),
                  show.legend = F,
                  data = w_df) +
    geom_point(aes(x = w, y = cvAUC, colour = Dataset),
               size = 3, position = position_dodge(0.025),
               data = w_df) +
    xlab("Relative weight of new negative texts") +
    ylab("AUC") +
    scale_color_manual(name = 'Indicator dataset',
                       values = c("#E69F00", "#009E73"),
                       breaks = c('LPD', 'PREDICTS')) +#,
    scale_y_continuous(limits = c(0.5,1)) +
    scale_x_continuous(breaks = seq(0,1,0.2)) +
    theme_bw() +
    theme(
          axis.text = element_text(size = 16), 
          axis.title = element_text(size = 20), 
          legend.text = element_text(size = 16), #  20 
          legend.title = element_text(size = 18),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())


cmb_grd <- plot_grid(npos_plt1 + theme(legend.position = "none"),
          w_plt1 + theme(legend.position = "none",
                         axis.title.y = element_blank()),
          labels = c("a.", "b."),
          label_size = 20,
          align = "hv")

cmb_leg <- get_legend(w_plt1)

cmb_plt <- plot_grid(cmb_grd, cmb_leg,
                     nrow = 1, rel_widths = c(1,0.15))

ggsave(plot = cmb_plt, filename = 'cmb_npos_aug_plt.pdf',
       path = '../Results/Figs',
       width = 15, height = 5, dpi = 300, device = "pdf")

