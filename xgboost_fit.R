library(tidyverse)
library(xgboost)
library(caret)
library(isotone)
library(ROCR)

set.seed(53453)

# Functions
# Brier Score
brier <- function(f,a, clip = FALSE){
  
  if(clip == TRUE){
    f_clip <- ifelse(f >= .5, .499, f)
      mean((f_clip - a)^2)
  }
  else
  mean((f - a)^2)

}

# Black vs White FP Rate
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

# Custom Loss Function
# For Brier Score
# Pass in predictions and training labels
evalbrier <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  bscore <- brier(labels, exp(preds) )
  return(list(metric = "brier", value = bscore))
}

# Load Data
nij_data <- read_csv("C:/Users/gioc4/Dropbox/nij_forecasting/train_data_matrix .csv")

# Split Train/Test Datasets
# 80/20 split
train_data <- sample_frac(nij_data, size = .8)
test_data <- filter(nij_data, !id %in% train_data$id)

# Set up xgboost objects
# Remove id and outcome variables
xgboost_train <- train_data %>%
  select(-id:-yany) %>%
  as.matrix()

xgboost_test <- test_data%>%
  select(-id:-yany) %>%
  as.matrix()

xtrain <- xgb.DMatrix(label = train_data$y1, data = xgboost_train)
xtest <- xgb.DMatrix(label = test_data$y1, data = xgboost_test)

# HYPERPARAMETER TUNING
#-------------------------#

# Find optimal values of 
# subsample
# eta,
# maxdepth

# Set up tuning grid
tuning_grid <- expand.grid(
  nrounds = 1500,
  min_child_weight = 2,
  colsample_bytree = 1,
  subsample = .75,
  eta = 0.01,
  max_depth = 3,
  gamma = 1
)

# Control options
xgb_train_control <- trainControl(
  method = 'cv',
  number = 5,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = 'all',
  allowParallel = TRUE,
  classProbs = TRUE
)

# Find optimal parameters
# WARNING: Takes > 30 min to run
xgb_train_1 <- train(
  x = xgboost_train,
  y = as.factor(ifelse(train_data$y1 == 1,"yes","no")),
  trControl = xgb_train_control,
  tuneGrid = tuning_grid,
  method = 'xgbTree'
)

# Optimal values
# max_depth = 1
# eta = 0.1
# nrounds = 1000,
# subsample = .75
xgb_train_1
plot(xgb_train_1)

# MODEL FITTING
#-------------------------#

watchlist <- list(eval = xtest, train = xtrain)

# Fit Model with
# optimal parameters from tuning grid

# Model parameters
# set evaluation metric to 'auc'
# fit with caret
param <-
  list(
    max_depth = 3,
    eta = 0.01,
    verbose = 0,
    nthread = 2,
    subsample = .75,
    min_child_weight = 2,
    objective = "binary:logistic",
    eval_metric = 'auc')

# Directly optimize brier score
param2 <-
  list(
    max_depth = 3,
    eta = 0.005,
    verbose = 0,
    nthread = 2,
    subsample = .75,
    min_child_weight = 2,
    objective = "binary:logistic",
    eval_metric = evalbrier)

# Fit Model
fit1 <-
  xgb.train(
    param,
    xtrain,
    watchlist = watchlist,
    early_stopping_rounds = 50,
    nrounds = 1000)

fit2 <-
  xgb.train(
    param2,
    xtrain,
    watchlist = watchlist,
    maximize = FALSE,
    early_stopping_rounds = 100,
    nrounds = 1000
  )

# Variable importance scores
importance <- xgb.importance(feature_names = fit1$feature_names, model = fit1)

# Top 5
head(importance)

# MODEL PERFORMANCE
#-------------------------#

# Get predictions on test data
yhat <- predict(fit1, xgboost_test)

# Performance
pred <- prediction(yhat, test_data$y1)

# Calculate ROC and AUC
perf.roc <- performance(pred, "tpr","fpr") 
perf.auc <- performance(pred, measure = "auc")

plot(perf.roc,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))

# Plot ROC & AUC
perf.auc@y.values[[1]]

# Calibration plots

# Isotone scaling using gpava
# Re-calibrate estimates on training data

cal.plot <- data.frame(y = fct_relevel(as.factor(test_data$y1),"1","0"),
                       yhat = yhat)

cal.df <- calibration(y ~ yhat, data = cal.plot, cuts = 20)$data %>%
  na.omit()


ggplot(cal.df ) +
  geom_abline(intercept = 0, slope = 1, color = "darkgrey", size = 1, linetype = 'dashed') +
  geom_line(aes(x = midpoint, y = Percent), color = "darkblue", size = 1) +
  geom_point(aes(x = midpoint, y = Percent), color = "darkblue", size = 2) +
  coord_cartesian(xlim = c(0, 100), ylim = c(0, 100)) +
  theme_classic()


# Plot on test data
bins <- 60

cal_df <- data.frame(y = test_data$y1,yhat = yhat) %>%
  arrange(yhat) %>%
  mutate(bin = ntile(yhat, bins)) %>%
  group_by(bin) %>%
  summarise(Actual = mean(y),
            Predicted = mean(yhat)) %>%
  pivot_longer(-bin, names_to = 'prob')

ggplot(cal_df, aes(x = bin, y = value, color = prob)) +
  geom_line(size = 1) +
  geom_point(aes(shape = prob)) +
  scale_x_continuous(breaks = seq(0,bins, by = 10)) +
  theme_minimal() +
  theme(
    legend.position = c(.2, .9),
    legend.background = element_rect(fill = "white", color = NA),
    legend.text = element_text(size = 10, face = 'bold'),
    legend.title = element_blank(),
    axis.text = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14, color = "black"),
    panel.grid.minor = element_blank()) +
  labs(
    x = "Bin",
    y = "Probability",
    title = "Calibration Plot",
    subtitle = "Actual vs. Predicted") +
  colorspace::scale_color_discrete_qualitative(palette = "harmonic") 



# STUFF YOU WIN ON
#-------------------------#

# Calculate Brier Score
print(bscore <- brier(yhat, test_data$y1, clip = TRUE))

# Difference
# black - white FP
# set cutoff to .499 to get 0 false positives
print(fp_dif <- abs(fp_rate(test_data, yhat, cutoff = .4999)))

# Fairness metric
(1 - bscore)*(1 - fp_dif)

# PREDICT ON TEST DATA
#-------------------------#
final_test_data <- read_csv("C:/Users/gioc4/Dropbox/nij_forecasting/test_data_matrix.csv") 

final_test_data <- final_test_data %>%
  select(-id) %>%
  as.matrix() %>%
  xgb.DMatrix()

final_pred <- predict(fit1, final_test_data)

# Export
write_csv(data.frame(y = final_pred),"C:/Users/gioc4/Desktop/final_pred.csv")
