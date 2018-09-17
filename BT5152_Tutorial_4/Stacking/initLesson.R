# https://github.com/zachmayer/caretEnsemble/issues/228
library(devtools)
devtools::install_github('zachmayer/caretEnsemble')

library(tidyverse)
library(mlbench)
library(rpart)
library(C50)
library(e1071)
library(caret)
library(caretEnsemble)
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


index <- createFolds(training$Class, 5)
control <- trainControl(method="repeatedcv", number=5, repeats=3,
                        index=index, savePredictions='final',
                        classProbs=TRUE, summaryFunction=twoClassSummary)
algos <- c('rpart', 'knn', 'C5.0Tree', 'svmLinear2')
models <- caretList(Class~., data=training, metric="ROC",
                    trControl=control, methodList=algos)
results <- resamples(models)
summary(results)
modelCor(results)

# stack using random forest
stack_control <- trainControl(method="repeatedcv", number=5, repeats=3,
                              classProbs=TRUE, summaryFunction=twoClassSummary)
stack_rf <- caretStack(models, method="rf", metric="ROC", trControl=stack_control)
print(stack_rf)

stacked_out_train <- predict(stack_rf, training)
stacked_out <- predict(stack_rf, test)
stacked_accuracy_train <- mean(training$Class == stacked_out_train)
stacked_accuracy <- mean(test$Class == stacked_out)
print(c(stacked_accuracy_train, stacked_accuracy,
        stacked_accuracy_train - stacked_accuracy))

# Find better cut-off threshold
stacked_out_train_prob <- predict(stack_rf, training, type='prob')
thresholds <- seq(0, 1, .05)
max_train_accuracies <- 0
threshold <- 0
for (i in thresholds) {
  threshold_value <- i
  stacked_out_train <- factor(stacked_out_train_prob<threshold_value,
                              levels=c(T, F), labels=c('good', 'bad'))
  accuracy_train <- mean(stacked_out_train == training$Class)
  cat("Accuracy for threshold value", i, ':', accuracy_train, "\n")
  if (accuracy_train > max_train_accuracies) {
    max_train_accuracies <- accuracy_train
    threshold <- i
  }
}
cat("Best train accuracy training accuracy is", max_train_accuracies, ' achieve at threshold:', threshold, "\n")

stacked_out_prob <- predict(stack_rf, test, type='prob')
stacked_out <- factor(stacked_out_prob<threshold, levels=c(T, F), labels=c('good', 'bad'))
stacked_accuracy <- mean(test$Class == stacked_out)
print(c(max_train_accuracies, stacked_accuracy,
        max_train_accuracies - stacked_accuracy))

