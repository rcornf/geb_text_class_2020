#%%%%%%%%%%
## Rscript to analyse changes in LRA predictions when removing high influence 
## words that are absent from GloVe
#%%%%%%%%%%

# Clear env
rm(list = ls())
graphics.off()


# Load data
lpi_tr_df <- read.csv("../Results/glove_stop_lpi_tr_df.csv")
lpi_te_df <- read.csv("../Results/glove_stop_lpi_te_df.csv")
predicts_tr_df <- read.csv("../Results/glove_stop_predicts_tr_df.csv")
predicts_te_df <- read.csv("../Results/glove_stop_predicts_te_df.csv")


library(ggplot2)
library(cowplot)

plt <- plot_grid(ggplot() +
            geom_point(aes(x = p_orig, y = p_stop), alpha = 0.25, data = lpi_tr_df) +
            geom_point(aes(x = p_orig, y = p_stop), alpha = 0.25, color = "red",
                       data = lpi_te_df) +
            xlab("Original relevance") +
            ylab("New relevance") + 
            ggtitle("LPD") +
            xlim(c(0,1)) + 
            ylim(c(0,1)) +
            coord_equal() +
            theme_bw() +
            theme(plot.title = element_text(size = 26),
                  axis.text = element_text(size = 20), 
                  axis.title = element_text(size = 24)),
          ggplot() +
            geom_point(aes(x = p_orig, y = p_stop), alpha = 0.25, data = predicts_tr_df) +
            geom_point(aes(x = p_orig, y = p_stop), alpha = 0.25, color = "red",
                       data = predicts_te_df) +
            ggtitle("PREDICTS") +
            xlab("Original relevance") +
            ylab(element_blank()) + 
            xlim(c(0,1)) + 
            ylim(c(0,1)) +
            coord_equal() +
            theme_bw() +
            theme(plot.title = element_text(size = 26),
                  axis.text = element_text(size = 20), 
                  axis.title = element_text(size = 24)),
          axis = "b", align = "hv", nrow = 1)

ggsave(plot = plt, filename = "glove_stop_plt.pdf",
       path = "../Results/Figs/",
       width = 15, height = 10, dpi = 300, device = "pdf")
