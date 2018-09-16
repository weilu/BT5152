library(tidyverse)
library(mlbench)
library(rpart)
library(caret)
library(randomForest)

options(dplyr.width = Inf)

data('Ionosphere')

set.seed(1337)

Ionosphere <- Ionosphere[,-2]
Ionosphere$V1 <- as.numeric(as.character(Ionosphere$V1))
Ionosphere$index <- 1:nrow(Ionosphere)

ionosphere <- as.tibble(Ionosphere)

training <- withr::with_seed(42, sample_frac(Ionosphere, 0.8))
test <-ionosphere %>%
  anti_join(training, by = 'index') %>%
  dplyr::select(-index)
training <- training %>% dplyr::select(-index)

model <- rpart(Class ~., training)

out_train <- predict(model, training, type='class')
accuracy_train <- mean(training$Class == out_train)
out <- predict(model, test, type='class')
accuracy <- mean(test$Class == out)
print(c(accuracy_train, accuracy,
        accuracy_train - accuracy))

# build manual bagging models
train_bagging <- function(n_bags, training_data){
  models <- lapply(1:n_bags, function(i) {
    n_sample <- nrow(training_data)
    sample_indexes <- sample(1:n_sample, replace = TRUE)
    return(rpart(Class ~., training_data[sample_indexes, ]))
  })
  return(models)
}

# use bagging models to predict
predict_bagging <- function(models, test_data) {
  probs <- sapply(models, function(m) {
    return(as.data.frame(predict(m, test_data, type='prob'))[, -1]) # get the predicted prob of the positive class
  })
  avg_probs <- rowSums(probs) / length(models)
  return(factor(avg_probs>0.5, levels=c(T, F), labels=c('good', 'bad')))
}

models <- train_bagging(50, training)
out_bagging <- predict_bagging(models, test)
accuracy_bagging <- mean(test$Class == out_bagging)
out_bagging_train <- predict_bagging(models, training)
accuracy_bagging_training <- mean(training$Class == out_bagging_train)
print(c(accuracy_bagging_training, accuracy_bagging,
        accuracy_bagging_training - accuracy_bagging))


# use 5-fold cv for model selection
control <- trainControl(method="cv", number=5)

# bagged CART using caret
model_treebag <- train(Class~., data=training, method="treebag", trControl=control, control=rpart.control())
out_train_treebag <- predict(model_treebag, training, type='raw')
accuracy_train_treebag <- mean(training$Class == out_train_treebag)
out_treebag <- predict(model_treebag, test, type='raw')
accuracy_treebag <- mean(test$Class == out_treebag)
print(c(accuracy_train_treebag, accuracy_treebag,
        accuracy_train_treebag - accuracy_treebag))

# bagging with random forest
model_rf <- train(Class~., data=training, method="rf", trControl=control)
out_train_rf <- predict(model_rf, training, type='raw')
accuracy_train_rf <- mean(training$Class == out_train_rf)
out_rf <- predict(model_rf, test, type='raw')
accuracy_rf <- mean(test$Class == out_rf)
print(c(accuracy_train_rf, accuracy_rf,
        accuracy_train_rf - accuracy_rf))
