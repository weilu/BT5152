library(tidyverse)
library(rpart)
library(caret)
library(randomForest)
library(Metrics)
library(ggplot2)

options(dplyr.width = Inf)

data('diamonds')

set.seed(1337)

sample_indexes <- withr::with_seed(42, sample(1:nrow(diamonds), 1000))
diamonds_sample <- diamonds[sample_indexes, ]
data <- as.tibble(diamonds_sample)
data$index <- 1:nrow(diamonds_sample)

training <- withr::with_seed(42, sample_frac(data, 0.8))
test <-data %>%
  anti_join(training, by = 'index') %>%
  dplyr::select(-index)
training <- training %>% dplyr::select(-index)

model <- rpart(price ~., training)
out_train <- predict(model, training)
rpart_mae_train <- mae(training$price, out_train)
out <- predict(model, test)
rpart_mae <- mae(test$price, out)
print(c(rpart_mae_train, rpart_mae,
        rpart_mae - rpart_mae_train))


# build manual bagging models
train_bagging <- function(n_bags, training_data){
  models <- lapply(1:n_bags, function(i) {
    n_sample <- nrow(training_data)
    sample_indexes <- sample(1:n_sample, replace = TRUE)
    return(rpart(price ~., training_data[sample_indexes, ]))
  })
  return(models)
}

# use bagging models to predict
predict_bagging <- function(models, test_data) {
  prices <- sapply(models, function(m) {
    return(predict(m, test_data))
  })
  return(rowSums(prices) / length(models))
}

models <- train_bagging(25, training)
out_bagging_train <- predict_bagging(models, training)
bagging_mae_train <- mae(training$price, out_bagging_train)
out_bagging <- predict_bagging(models, test)
bagging_mae <- mae(test$price, out_bagging)
print(c(bagging_mae_train, bagging_mae,
        bagging_mae - bagging_mae_train))


# use 5-fold cv for model selection
control <- trainControl(method="cv", number=5)
metric <- 'MAE'

# bagged CART using caret
model_treebag <- train(price~., data=training, method="treebag", control=rpart.control(),
                       metric=metric, trControl=control)
out_train_treebag <- predict(model_treebag, training)
treebag_mae_train <- mae(training$price, out_train_treebag)
out_treebag <- predict(model_treebag, test)
treebag_mae <- mae(test$price, out_treebag)
print(c(treebag_mae_train, treebag_mae,
        treebag_mae - treebag_mae_train))

# bagging with random forest
model_rf <- train(price~., data=training, method="rf",
                  metric=metric, trControl=control)
out_train_rf <- predict(model_rf, training)
rf_mae_train <- mae(training$price, out_train_rf)
out_rf <- predict(model_rf, test)
rf_mae <- mae(test$price, out_rf)
print(c(rf_mae_train, rf_mae,
        rf_mae - rf_mae_train))
