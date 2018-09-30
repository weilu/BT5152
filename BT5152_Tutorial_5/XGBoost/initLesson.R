library(tidyverse)
library(mlbench)
library(xgboost)
library(DiagrammeR) # for xgb tree plotting
library(caret)

options(dplyr.width = Inf)

data('Ionosphere')

set.seed(1337)

Ionosphere <- Ionosphere[,-2]
Ionosphere$V1 <- as.numeric(as.character(Ionosphere$V1))
Ionosphere$index <- 1:nrow(Ionosphere)

ionosphere <- as.tibble(Ionosphere)

training <- withr::with_seed(42, sample_frac(ionosphere, 0.8))
test <-ionosphere %>%
  anti_join(training, by = 'index') %>%
  dplyr::select(-index)
training <- training %>% dplyr::select(-index)

# adaboost train/test accuracies: [1] 0.9893238 0.8714286

train_label_binary <- ifelse(training$Class == 'good', 0, 1)
test_label_binary <- ifelse(test$Class == 'good', 0, 1)
train_matrix <- xgb.DMatrix(as.matrix(select(training, -Class)), label=train_label_binary)
test_matrix <- xgb.DMatrix(as.matrix(select(test, -Class)), label=test_label_binary)

# by default objective is reg:linear
model_xgb <- xgboost(data=train_matrix, nrounds=20, objective="binary:logistic")
pred_prob <- predict(model_xgb, test_matrix)
pred_xgb <- as.numeric(pred_prob > 0.5)
mean(pred_xgb == test_label_binary)

# by default eval_metric is error for classification
model_xgb_auc <- xgboost(data=train_matrix, nrounds=20, objective="binary:logistic", eval_metric='auc')
pred_xgb_auc <- as.numeric(predict(model_xgb, test_matrix) > 0.5)
mean(pred_xgb_auc == test_label_binary)

model_xgb_gamma_10 <- xgboost(data=train_matrix, nrounds=20, objective="binary:logistic", gamma=10)
pred_xgb_gamma_10 <- as.numeric(predict(model_xgb_gamma_10, test_matrix) > 0.5)
mean(pred_xgb_gamma_10 == test_label_binary)
xgb.importance(model=model_xgb_gamma_10)
xgb.plot.tree(model=model_xgb_gamma_10)

ctrl <- trainControl(method = "cv", number = 5)
# https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
grid <- expand.grid(.max_depth=c(1, 3, 6), .min_child_weight=c(1, 5), .gamma=c(0, 1, 10),
                    .subsample=c(0.8, 1), .colsample_bytree=c(0.8, 1),
                    .nrounds=c(20, 100), .eta=c(0.01, 0.3, 0.6))
model_xgb_tuned <- train(Class ~ ., data=training, method="xgbTree", trControl=ctrl, tuneGrid=grid)
print(model_xgb_tuned$bestTune)
pred_xgb_tuned <- predict(model_xgb_tuned, test)
mean(pred_xgb_tuned == test$Class)

