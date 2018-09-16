library(tidyverse)
library(mlbench)
library(rpart)
library(adabag)
library(caret)

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

train_boosting <- function(n_trees, training_data) {
  n_sample <- nrow(training_data)

  #start with equal weights
  weights <- as.numeric(unlist(list(rep(1/n_sample, n_sample))))

  models <- list()
  for(i in 1:n_trees){
    #sample with weights
    sample_indexes <- sample(1:n_sample, prob=weights, replace=TRUE)
    m <- rpart(Class ~., training_data[sample_indexes, ])
    models <- c(models, list(m))

    #calculate new probabilities based on prediction accuracy on training set
    good_dummy <- ifelse(as.character(training_data$Class) == 'good', 1, 0)
    pred_prob <- as.data.frame(predict(m, training_data, type='prob'))$good
    err <- abs(good_dummy - pred_prob)
    weights <- as.numeric(err/sum(err))
  }

  return(models)
}

predict_boosting <- function(models, test_data) {
  probs <- sapply(models, function(m) {
    return(as.data.frame(predict(m, test_data, type='prob'))$good)
  })
  avg_probs <- rowSums(probs) / length(models)
  return(factor(avg_probs>0.5, levels=c(T, F), labels=c('good', 'bad')))
}

models <- train_boosting(3, training)
boosted_out_train <- predict_boosting(models, training)
boosted_accuracy_train <- mean(training$Class == boosted_out_train)
boosted_out <- predict_boosting(models, test)
boosted_accuracy <- mean(test$Class == boosted_out)
print(c(boosted_accuracy_train, boosted_accuracy,
        boosted_accuracy_train - boosted_accuracy))

# ada boosting only works for classification
control <- trainControl(method="cv", number=5)
grid = expand.grid(.mfinal=10, .maxdepth=30)
model_adaboost <- train(Class~., data=training, method='AdaBag',
                        tuneGrid=grid, trControl=control)
adaboosted_out_train <- predict(model_adaboost, training)
adaboosted_accuracy_train <- mean(training$Class == adaboosted_out_train)
adaboosted_out <- predict(model_adaboost, test)
adaboosted_accuracy <- mean(test$Class == adaboosted_out)
print(c(adaboosted_accuracy_train, adaboosted_accuracy,
        adaboosted_accuracy_train - adaboosted_accuracy))


