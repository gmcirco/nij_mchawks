# Replication materials for Circo & Wheeler 2021
# 
# This script re-creates the round 2 analysis
# This is a slimmed-down version of the original script
# Mainly here to show evaluation of test-train data for paper

# Required libraries for analysis
library(tidyverse)
library(xgboost)
library(caret)
library(ROCR)
library(iml)

set.seed(53453)

# Custom functions for analysis

# Brier Score function
brier <- function(f,a, clip = FALSE){
  
  if(clip == TRUE){
    f_clip <- ifelse(f >= .5, .499, f)
      mean((f_clip - a)^2)
  }
  else
  mean((f - a)^2)

}

# False-Positive Analysis Black vs White
fp_rate <- function(df, yhat, cutoff = .5){
  
  # Calculate difference in FP
  # Black v. White
  df <- df %>%
    mutate(pred = ifelse(yhat >= .5, cutoff, yhat),
           recid = ifelse(pred >= .5,1,0))
  
  # Get FP for white v. black
  suppressMessages(
    fp_out <-
  df %>%
    count(racewhite, recid, y1) %>%
    filter(recid == 1, y1 == 0) %>%
    left_join(count(df, racewhite, name = 'count')) %>%
    mutate(fp_rate = n/count) %>%
    summarise(fp_diff = fp_rate - lead(fp_rate,1)) %>%
    na.omit() %>%
    pull())
  
  if(is_empty(fp_out))
    fp_out <- 0
  
  return(fp_out)
}


# DATA SETUP
#-------------------------#

# Load
# This is the data that was used for forecasting
nij_data <- read_csv("C:/Users/gioc4/Dropbox/nij_forecasting/train_data_matrix2.csv")

# This is the full data used for evaluation
nij_full <- read_csv("C:/Users/gioc4/Dropbox/nij_forecasting/full_data_matrix.csv")

# Filter out recidivism at time 1
# Predictions made for time 2
nij_data <- nij_data %>%
  filter(y1 == 0)

# Isolate the test dataset from the train set
nij_full <- nij_full %>%
  filter(y1 == 0, !id %in% nij_data$id)

# Set up XGBoost models
# Train data
xgboost_train <- nij_data %>%
  select(-id:-yany) %>%
  as.matrix()

# Full Evaluation data
xgboost_test <- nij_full %>%
  select(-id:-yany) %>%
  as.matrix()

# Separate data into test-train datasets
xtrain <- xgb.DMatrix(label = nij_data$y2, data = xgboost_train)

xtest <- xgb.DMatrix(label = nij_full$y2, data = xgboost_test)

# MODEL FITTING
#-------------------------#
# Fit Model with
# optimal parameters from tuning grid

# Model parameters
# set evaluation metric to 'auc'
# fit with caret
param <-
  list(
    max_depth = 6,
    eta = 0.005,
    verbose = 0,
    nthread = 2,
    subsample = .75,
    min_child_weight = 5,
    max_delta_step = 5,
    objective = "binary:logistic",
    eval_metric = 'auc')

# Fit model
# on full data
fit1.full <-
  xgb.train(
    param,
    xtrain,
    nrounds = 2000)

# Variable importance scores
importance <- xgb.importance(feature_names = fit1.full$feature_names, model = fit1.full)

# Top 5
head(importance)

# MODEL PERFORMANCE
#-------------------------#

# Get predictions on final evaluation data
yhat <- predict(fit1.full, xgboost_test)
yhat.male <- predict(fit1.full, subset(xgboost_test, xgboost_test[,"sexf"] == 0))
yhat.female <- predict(fit1.full,subset(xgboost_test, xgboost_test[,"sexf"] == 1))

# Performance
pred <- prediction(yhat, nij_full$y2)

# Calculate ROC and AUC
perf.roc <- performance(pred, "tpr","fpr") 
perf.auc <- performance(pred, measure = "auc")

plot(perf.roc,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))

# Plot ROC & AUC
perf.auc@y.values[[1]]

# STUFF YOU WIN ON
#-------------------------#

# NOTE: These scores are slightly lower than the published results
# likely due to some randomness in the fitting function, despite
# the same seed.

# BRIER SCORE

# Calculate Brier Score on full sample
print(bscore <- brier(yhat, nij_full$y2, clip = TRUE))

# Just males
print(bscore <- brier(yhat.male, filter(nij_full, sexf == 0) %>% pull(y2), clip = TRUE))

# Just females
print(bscore <- brier(yhat.female, filter(nij_full, sexf == 1) %>% pull(y2), clip = TRUE))

# average accuracy
mean(c(0.1596534, 0.1228812))

# FAIR AND ACCURATE

# set cutoff to .499 to get 0 false positives
print(fp_dif <- abs(fp_rate(nij_full, yhat, cutoff = .5)))

# Fairness metric
(1 - bscore)*(1 - fp_dif)

# ADDITONAL PLOTS & DIAGNOSTIC STUFF

# PLOT 1
# Variable importance
label_order <- importance %>%
  dplyr::slice(1:20) %>%
  mutate(Feature = fct_reorder(Feature, Gain))

importance %>%
  dplyr::slice(1:20) %>%
  dplyr::select(-Cover) %>%
  pivot_longer(cols = c(Gain, Frequency), names_to = "measure") %>%
  mutate(Feature = fct_relevel(Feature, levels(label_order$Feature))) %>%
  ggplot() +
  geom_linerange(
    aes(
      xmin = 0,
      xmax = value,
      y = Feature,
      group = measure,
      color = measure),
    position = position_dodge(width = 0.5)) +
  geom_point(
    aes(x = value, 
        y = Feature, 
        color = measure), 
    position = position_dodge(width = 0.5)) +
  labs(x = "Value") +
  theme_bw() +
  theme(legend.title = element_blank(),
        legend.text = element_text(face = 'bold'),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.title.y = element_blank()) +
  colorspace::scale_color_discrete_qualitative(palette = "dynamic")
ggsave("C:/Users/gioc4/Desktop/plot1.svg")

# PLOT 2.
# Accumulated Local Effects

# Get newdata 
# subtracting out response and id columns
newdata <- nij_full %>%
  select(-id:-yany) %>%
  data.frame()

# Function to get XGBoost predictions
pred <- function(model, newdata)  {
  
  X <- as.matrix(newdata)
  
  results <- as.data.frame(predict(model, X))
  return(results)
}

# Function to get ALE
calc_ale <- function(model, df, feature){
  
  mod <- Predictor$new(model, data = df, predict.function = pred)
  eff <- FeatureEffect$new(mod, feature = feature, grid.size = 30, method = "ale")
  
  out <- eff$results %>%
    mutate(feature = feature) %>%
    set_names(c('type','value','estimate','feature'))
  
  return(out)
}

mod <- Predictor$new(fit1.full, data = newdata, predict.function = pred)
eff <- FeatureEffects$new(mod, feature = c("risk_score","employed_jobs"))

rbind.data.frame(eff$results$risk_score,
                 eff$results$employed_jobs)

# Get top 5 predictors in a list
# plot in facets
ale_list <- list()
feature_list <- as.character(label_order[1:5]$Feature)

for (i in 1:5) {
  ale_list[[i]] <-
    calc_ale(model = fit1.full,
             df = newdata,
             feature = feature_list[i])
}

# Plot
do.call(rbind, ale_list) %>%
  ggplot() +
  geom_line(aes(x = estimate, y = value, color = feature), size = 1) +
  facet_wrap(~feature, scales = 'free') +
  colorspace::scale_color_discrete_qualitative(palette = "dynamic") +
  theme_bw() +
  theme(legend.position = 'none',
        strip.background = element_blank(),
        strip.text = element_text(hjust = 0),
        panel.grid = element_blank())
ggsave("C:/Users/gioc4/Desktop/plot2.svg", width = 6, height = 3)


# Shapely Values
# create predictor
predictor.xgboost <- Predictor$new(
  model = fit1.full,
  data = newdata,
  predict.function = pred
)

sort(unlist(pred(fit1.full, newdata)), decreasing = TRUE)

# High probability individual
shapely.fit <- Shapley$new(predictor = predictor.xgboost, x.interest = newdata[4664,])

shp1 <-
  shapely.fit$results %>%
  arrange(desc(abs(phi))) %>%
  mutate(id = paste0("pred = ", round(as.numeric(shapely.fit$y.hat.interest),3) )) %>%
  dplyr::slice(1:5)

# Low probability individual
shapely.fit <- Shapley$new(predictor = predictor.xgboost, x.interest = newdata[619,])

shp2 <-
  shapely.fit$results %>%
  arrange(desc(abs(phi))) %>%
  mutate(id = paste0("pred = ", round(as.numeric(shapely.fit$y.hat.interest),3) )) %>%
  dplyr::slice(1:5)

# rbind and plot
rbind.data.frame(shp1,shp2) %>%
  ggplot() +
  geom_col(aes(x = phi, y = fct_reorder(feature.value, phi), fill = id)) +
  facet_wrap(~id, ncol = 1, scales = "free") +
  labs(x = "Shapely Value") +
  colorspace::scale_fill_discrete_qualitative(palette = "dynamic") +
  theme_bw()+
  theme(legend.position = 'none',
        strip.background = element_blank(),
        axis.title.y = element_blank(),
        strip.text = element_text(hjust = 0, size = 12),
        panel.grid.minor = element_blank())
ggsave("C:/Users/gioc4/Desktop/plot4.svg")


# CONFUSION MATRIX
conf_mat <-
  nij_full %>%
  select(id, y2, sexf) 

# Overall
confu_df1 <-
  conf_mat %>%
  mutate(pred = yhat,
         recid = ifelse(yhat >= .5, 1, 0),
         across(c(y2, recid), function(x){ifelse(x == 0, "no","yes")})) %>%
  count(y2,recid) %>%
  mutate(eval = case_when(
    y2 == "no" & recid == "no" ~ "true negative",
    y2 == "no" & recid == "yes" ~ "false positive",
    y2 == "yes" & recid == "no" ~ "false negative",
    y2 == "yes" & recid == "yes" ~ "true positive"),
    y2 = fct_relevel(y2, "yes"),
    outcome = str_extract(eval, "true|false")) %>%
  group_by(y2) %>%
  mutate(prop = round(n/ sum(n),2) )

confu_df2 <-
  conf_mat %>%
  mutate(pred = yhat,
         recid = ifelse(yhat >= .4, 1, 0),
         across(c(y2, recid), function(x){ifelse(x == 0, "no","yes")})) %>%
  count(y2,recid) %>%
  mutate(eval = case_when(
    y2 == "no" & recid == "no" ~ "true negative",
    y2 == "no" & recid == "yes" ~ "false positive",
    y2 == "yes" & recid == "no" ~ "false negative",
    y2 == "yes" & recid == "yes" ~ "true positive"),
    y2 = fct_relevel(y2, "yes"),
    outcome = str_extract(eval, "true|false")) %>%
  group_by(y2) %>%
  mutate(prop = round(n/ sum(n),2) )

# .5
mat1 <-
  ggplot(confu_df1) +
  geom_tile(aes(x = y2, y = recid, fill  = outcome), color = "white", size = 2) +
  geom_text(aes(x = y2, y = recid, label = prop), color = "white", size = 3) +
  geom_text(aes(x = y2, y = recid, label = eval), color = "white", size = 3, position = position_nudge(y = .125)) +
  colorspace::scale_fill_discrete_qualitative(palette = "harmonic") +
  labs(x = "Arrested", y = "Predicted", title = "Threshold = .5") +
  scale_x_discrete(position = "top") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text = element_text(size = 12),
        plot.title = element_text(face = 'bold'),
        panel.grid = element_blank())

# .35
mat2 <-
  ggplot(confu_df2) +
  geom_tile(aes(x = y2, y = recid, fill  = outcome), color = "white", size = 2) +
  geom_text(aes(x = y2, y = recid, label = prop), color = "white", size = 3) +
  geom_text(aes(x = y2, y = recid, label = eval), color = "white", size = 3, position = position_nudge(y = .125)) +
  colorspace::scale_fill_discrete_qualitative(palette = "harmonic") +
  labs(x = "Arrested", y = "Predicted", title = "Threshold = .35") +
  scale_x_discrete(position = "top") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text = element_text(size = 12),
        plot.title = element_text(face = 'bold'),
        panel.grid = element_blank())


cowplot::plot_grid(mat1, mat2, ncol = 2)

ggsave("C:/Users/gioc4/Desktop/plot3.svg", width = 6, height = 3)
