####
# R script to generate wordclouds from logistic model word weightings
####

# Clear env
rm(list = ls())
graphics.off()

# Load packages
require(wordcloud)
require(RColorBrewer)
require(gridExtra)

# Load word and weighting data
lpi_feats_df <- read.csv('../Results/Model_metrics/LR/lpi_feats_coeffs.csv')
predicts_feats_df <- read.csv('../Results/Model_metrics/LR/predicts_feats_coeffs.csv')

# Extract top and bottom ranked features
top_lpi_df <- head(lpi_feats_df[order(-lpi_feats_df$LR_A_coefficients),], n = 50)
bot_lpi_df <- head(lpi_feats_df[order(lpi_feats_df$LR_A_coefficients),], n = 50)

top_predicts_df <- head(predicts_feats_df[order(-predicts_feats_df$LR_A_coefficients),], n = 50)
bot_predicts_df <- head(predicts_feats_df[order(predicts_feats_df$LR_A_coefficients),], n = 50)

# Inverse negative weights
bot_lpi_df$LR_A_coefficients <- -1*bot_lpi_df$LR_A_coefficients
bot_predicts_df$LR_A_coefficients <- -1*bot_predicts_df$LR_A_coefficients


# Word clouds
# png('../Results/Figs/top_lpi_lr_A.png', width = 300, height = 300)
wordcloud(top_lpi_df$LR_A_features, 
          top_lpi_df$LR_A_coefficients, 
          min.freq = 0, random.order = F, 
          colors = brewer.pal(9, 'YlGn')[-c(1:2,9)],
          rot.per = 0.25, 
          scale = c(max(top_lpi_df$LR_A_coefficients),
                    min(top_lpi_df$LR_A_coefficients)))
# dev.off()

# png('../Results/Figs/bot_lpi_lr_A.png', width = 300, height = 300)
wordcloud(bot_lpi_df$LR_A_features, 
          bot_lpi_df$LR_A_coefficients, 
          min.freq = 0, random.order = F, 
          colors = brewer.pal(9, 'OrRd')[-c(1,9)],
          rot.per = 0.25, 
          scale = c(max(bot_lpi_df$LR_A_coefficients),
                    min(bot_lpi_df$LR_A_coefficients)))
# dev.off()


# png('../Results/Figs/top_predicts_lr_A.png', width = 300, height = 300)
wordcloud(top_predicts_df$LR_A_features, 
          top_predicts_df$LR_A_coefficients, 
          min.freq = 0, random.order = F, 
          colors = brewer.pal(9, 'YlGn')[-c(1:2,9)],
          rot.per = 0.25, 
          scale = c(max(top_predicts_df$LR_A_coefficients),
                    min(top_predicts_df$LR_A_coefficients)))
# dev.off()

# png('../Results/Figs/bot_predicts_lr_A.png', width = 300, height = 300)
wordcloud(bot_predicts_df$LR_A_features, 
          bot_predicts_df$LR_A_coefficients, 
          min.freq = 0, random.order = F, 
          colors = brewer.pal(9, 'OrRd')[-c(1,9)],
          rot.per = 0.25, 
          scale = c(max(bot_predicts_df$LR_A_coefficients),
                    min(bot_predicts_df$LR_A_coefficients)))
# dev.off()

