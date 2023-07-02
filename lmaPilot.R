getwd()

library(FSA)
library(DescTools)
library(rcompanion)
library(multcompView)
library(tidyverse)
library(plotrix)
library(gridExtra)
library(ggplot2)
library(reshape2)
library(purrr)
library(data.table)
library(GPArotation)
library(psych)
library(car)
library("Hmisc")
library("xtable")
library("writexl")
library("readxl")
library("rstatix")
library('ggpubr')
library(stringr)
library(broom)
library(cowplot)
library(dplyr)

combineRowPermutations <- function(df) {
  
  getEffortsTuplesCountsDF <- function(df) {
    #generate counts df by efforts_tuples
    df <- df %>% 
      count(efforts_tuples, selected0, selected1, name= "count")
    #create unique string identifier
    df$id <- row.names(df)
    print("df one")
    print(df)
    return (df)
  }
  
  delete_duplicate_subsequent_row <- function(df) {
    df = df[order(df[,'duplicate_generator'],-df[,'count']),]
    df = df[!duplicated(df$duplicate_generator),]
    return (df)
  }
  
  df <- getEffortsTuplesCountsDF(df)
  
  for (i_row in 1:nrow(df)) {
    for (j_row in i_row:nrow(df)) {
      if (df[i_row, "efforts_tuples"] == df[j_row, "efforts_tuples"]) {
        if (df[i_row, "selected0"] == df[j_row, "selected1"] & df[i_row, "selected1"] == df[j_row, "selected0"]) {
          #of matched rows, give min valued select0 column precedence 
          if (df[i_row, "selected0"] < df[j_row, "selected0"]) {
            df[i_row, "count"] <- df[i_row, "count"] + df[j_row, "count"]
          } else {
            df[j_row, "count"] <- df[i_row, "count"] + df[j_row, "count"]
          }
          df[i_row, "duplicate_generator"] <- paste(as.character(i_row), as.character(j_row))
          df[j_row, "duplicate_generator"] <- paste(as.character(i_row), as.character(j_row))
        } 
        
        else {
          df[i_row, "duplicate_generator"] <- NA
          df[j_row, "duplicate_generator"] <- NA
          # of unmatched row, give min valued select0 column precedence
          if (df[i_row, "selected0"] > df[i_row, "selected1"]) {
            df[i_row, c("selected0", "selected1")] <- df[i_row, c("selected1", "selected0")]
          }
        }
        
      }
    }
  }
  print("df two")
  print(df)
  # replace Nas in duplicate_generator with the corresponding id value
  df <- df %>% 
    mutate(duplicate_generator = coalesce(duplicate_generator,id))
  # delete duplicate subsequent permutation's row and eliminate extraneous columns, while sorting & resetting row idx numbers
  print("df three")
  print(df)
  df <- delete_duplicate_subsequent_row(df)
  df$index <- as.numeric(row.names(df))
  df <- df[order(df$index), ]
  rownames(df) <- NULL  
  print("df four")
  print(df)
  df <- df[, !names(df) %in% c("id", "duplicate_generator", "index")]
  return (df)
}

# df <- read.csv(file="lmaPilot.csv", stringsAsFactors=FALSE, header=T)
lma_pilot <- read.csv(file="walking.csv", stringsAsFactors=FALSE, header=T)

lma_pilot <- as.data.frame(lma_pilot)

# mydf <- lma_pilot %>%
#   rowwise() %>% 
#   mutate(combined = paste(sort(c_across(starts_with("efforts"))),collapse = "_"))
# 
# length(unique(mydf$combined))
# View(mydf)

# Create a new column combining effortsLeft and effortsRight by a policy (to eliminate potential permutations)
lma_pilot$efforts_tuples <- paste(pmin(lma_pilot$effortsLeft, lma_pilot$effortsRight), pmax(lma_pilot$effortsLeft, lma_pilot$effortsRight), sep = "_")
lma_pilot <- lma_pilot[,c("id", "motionType", "efforts_tuples", "effortsLeft", "effortsRight", "selected0", "selected1", "qInd")]
lma_pilot <- lma_pilot %>% 
  rename("motion_type" = "motionType")

# Generate counts of distinct elements in the 'column' column
counts <- sort(table(lma_pilot$efforts_tuples), decreasing = TRUE)
# in theory: 48 possible drive+state expressions, 
#C(48, 2)  = 24 * 47 = 1128 comparisons per motion
# C(56, 2) = 1540

# num comparisons: 1,120
length(counts)
# exclude very first entry of 973 counts
hist(counts[-1], main = "Effort Pair Frequency Histogram", xlab = "Count of Comparison Sample Size", ylab = "Frequency",
     breaks = seq(0, max(counts[-1]) + 3, by = 3))


# get counts by efforts_tuplesXselection as well as by motion
lma_pilot_test <- lma_pilot[, !names(lma_pilot) %in% c("motion_type")]
lma_pilot_counts <- combineRowPermutations(lma_pilot_test)
grouped_lma_pilot_counts <- lma_pilot_counts %>% group_by(efforts_tuples)
# Identify groups lacking specific select column values, populate such, and assign 0 to the corresponding count column
grouped_lma_pilot_counts <- grouped_lma_pilot_counts %>%
  complete(selected0=c(0,1), selected1 = c(1,2)) %>%
  mutate(count = replace_na(count, 0))
# Filter out the (1, 1) combination
grouped_lma_pilot_counts <- grouped_lma_pilot_counts %>%
  filter(!(selected0 == 1 & selected1 == 1))
# Normalize counts within each group to sum to 1
grouped_lma_pilot_counts <- grouped_lma_pilot_counts %>%
  mutate(count_normalized = count / sum(count))

#lma_pilot_0 <- combineRowPermutations(lma_pilot_0)
#lma_pilot_1 <- combineRowPermutations(lma_pilot_1)

#combine selected0 and selected1 into one column for all three dfs (orig and motion 0 and motion1) and eliminate permutations
grouped_lma_pilot_counts$selected_motions <- sort(paste(grouped_lma_pilot_counts$selected0, grouped_lma_pilot_counts$selected1, sep="_"))

#grouped_lma_pilot_counts$selected_motions <- paste(pmin(grouped_lma_pilot_counts$selected0, grouped_lma_pilot_counts$selected1), pmax(grouped_lma_pilot_counts$selected0, grouped_lma_pilot_counts$selected1), sep = "_")
# lma_pilot_0$selected_motions <- paste(lma_pilot_0$selected0, lma_pilot_0$selected1, sep="_")
# lma_pilot_1$selected_motions <- paste(lma_pilot_1$selected0, lma_pilot_1$selected1, sep="_")

# plot_motion <- ggplot(grouped_lma_pilot_counts, aes(x = efforts_tuples,  y = count, label = count, fill = selected_motions)) +
#   geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
#   geom_text(position = position_dodge2(width = 0.9, preserve = "single"), angle = 90, vjust=0.25) +
#   ggtitle("Counts of Selected Motion Pairs by Effort Tuples")

# plot_motion_0 <- ggplot(lma_pilot_0, aes(x = efforts_tuples,  y = count, label = count, fill = selected_motions)) +
#   geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
#   geom_text(position = position_dodge2(width = 0.9, preserve = "single"), angle = 90, vjust=0.25) + 
#   ggtitle("Motion 0 Counts of Selected Motion Pairs by Effort Tuples")
# 
# plot_motion_1 <- ggplot(lma_pilot_1, aes(x = efforts_tuples,  y = count, label = count, fill = selected_motions)) +
#   geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
#   geom_text(position = position_dodge2(width = 0.9, preserve = "single"), angle = 90, vjust=0.25) +
#   ggtitle("Motion 1 Counts of Selected Motion Pairs by Effort Tuples")

# combined_plot_motions <- plot_grid(plot_motion_0, plot_motion_1, plot_motion)

# ggsave(paste0("Selected_Motion_Pairs_Grid_Plot_Pilot",".png"), combined_plot_motions)



